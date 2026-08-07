# Dynamic Model Group Implementation Design

- Status: Implemented reference design
- Governing specification: [Deferred and Dynamic Model Group Specification](DYNAMIC_MODEL_GROUP_SPEC.md)
- Scope: recommended implementation for `pygent.llm`, `pygent.runtime`, history, and Worker integration

## 1. Role of this document

This document contains implementation design, not additional product principles.
Every implementation choice remains subordinate to the repository first principles,
the LLM and Runtime first principles, and the frozen second principles in the
governing specification.

Names, storage layout, lock primitives, and internal APIs may change. The following
correctness properties may not:

- exact pins never resolve current;
- admission-boundary pins cannot overwrite another boundary's pins;
- exact route resources remain resolvable while active pins or retained recoverable
  manifests depend on them;
- replay precedes capacity and live-resource acquisition;
- durable admission cannot commit an execution without its pin manifest;
- owned resources have exactly one close owner.

## 2. Recommended domain model

### 2.1 Definition values

Add an explicit resolution state to `ModelGroupConfig` or use an internal tagged
union with equivalent behavior:

```python
class ModelGroupResolution(str, Enum):
    CONCRETE = "concrete"
    DEFERRED = "deferred"
```

Required invariants:

```text
CONCRETE -> routes non-empty; fallback references routes
DEFERRED -> routes empty; fallback empty; no local invoker
```

`ModelGroupConfig.deferred()` is the public convenience constructor. Internal code
should narrow to a concrete deployment before calling `ModelInvoker.execute()` so a
deferred value cannot accidentally reach Provider execution.

### 2.2 Portable deployment values

Public frozen values:

```python
ModelResourceRef(
    resolver_id, resource_id, revision,
    capacity_owner_id,
    coordinator_domain,
)

ModelResourceBundle(
    resolver_id,
    route_resources,  # route_id -> ModelResourceRef
    capacity_owner_id,
    coordinator_domain,
)

ModelProfileSnapshot(
    deployment_scope_id, group_name, profile, snapshot_id,
    digest,
    resource_bundle_digest,
    model_group,
    resources,
)
```

The published portable record should not itself contain a live source. The registry
may associate the portable record with a non-portable exact-version source in a
separate runtime-owned table.

### 2.3 Admission identity

Every admission boundary receives a stable `admission_id`:

- Root admission: derived from or stored with the logical `execution_id`;
- inherited Child: reuses the active Parent `admission_id`;
- independent-Binding Child: derives a deterministic identity from Root execution,
  module path, module occurrence, and Child execution identity.

Recommended pin key:

```text
(admission_id, deployment_scope_id, model_group_name)
```

This permits two invocations of the same independently bound Child to select
different deployment snapshots while keeping deterministic replay identity.

## 3. Binding scope identity

`RuntimeBinding` should receive a Runtime-issued `deployment_scope_id`. The identity
should be stored in the Runtime's binding state rather than derived from display name
or exposed object address.

Recommended behavior:

```text
runtime.create_binding(...)
    -> create/resolve binding state
    -> issue deployment_scope_id
    -> return RuntimeBinding authority

runtime.bind(module, binding=raw_binding)
    -> create/resolve binding state
    -> issue deployment_scope_id
    -> return BoundModule exposing controlled publication authority
```

This preserves the existing equivalence of both Binding SDK forms. Publication APIs
accept a Runtime-issued handle or BoundModule authority, never only the policy value.

For non-durable process-local scopes, random IDs retained for the Runtime lifetime
are sufficient. If execution history is durable, the scope mapping and its authority
record must be stored or resolved by an external issuer for the same retained-history
lifetime. A recovered scope ID is identity to validate, not a bearer credential.

## 4. Plan compilation

### 4.1 Typed requirements

Add a typed field to `ModuleSpec` or `ExecutionPlan`; do not encode deferred
requirements in arbitrary metadata or only in `resource_keys`.

One possible shape is:

```json
{
  "model_requirements": [
    {
      "module_path": "root.model",
      "model_group": "assistant",
      "resolution": "deferred",
      "capacity_key": "assistant-model",
      "max_concurrency": 8
    }
  ]
}
```

The compiler must partition requirements by effective Binding boundary:

- raw local Children inherit the current partition;
- a pre-bound Child starts a new partition;
- remote nodes carry a capability requirement and do not resolve against the local
  registry;
- shared Module definitions retain one definition identity, but requirements remain
  addressable from every effective admission boundary that can execute them.

Compilation rejects incompatible declarations for one deployment key or physical
capacity key.

### 4.2 Graph closure enforcement

`resolve_model_deployment()` must verify both:

1. the requested group is a typed requirement of the active admission boundary;
2. the corresponding pin exists in that boundary.

This runtime check closes the gap left by Python code that constructs or calls a
ModelCallLayer not declared in the frozen graph.

### 4.3 Schema version

Adding typed requirements changes the plan schema and graph hash. Add a new schema
version while retaining decoders for every fixed-model schema that remains inside the
supported history/Worker compatibility window. Alternatively, ship an explicit
migration before removing a decoder. Update fixtures, Worker protocol tests, history
tests, and documentation atomically.

## 5. Deployment preparation and digest

### 5.1 Preparation boundary

An LLM deployment builder owns:

- Provider and adapter compatibility validation;
- client construction or exact-version factory construction;
- secret resolution;
- per-route resource revision issuance and bundle construction;
- capacity owner and coordinator-domain resolution;
- portable metadata secret exclusion;
- construction of a live source with unambiguous per-entry ownership;
- declaration of resident-only or reconstructable source capability.

Runtime sees only provider-neutral portable values and the generic live-source
contract.

### 5.2 Exact resource bundle

The builder maps every concrete route to an `ExactResourceRef`. Several routes may
reuse one ref, so the common single-endpoint case stays compact. The live source
acquires the exact bundle and constructs or leases the one `ModelInvoker` used by the
existing retry/fallback path. Provider-specific composition stays outside Runtime.

Every resolver contract must make revision meaningful. A minimal resolver remains:

```python
class ModelResourceResolver(Protocol):
    @property
    def resolver_id(self) -> str: ...

    async def acquire(
        self,
        ref: ExactResourceRef,
    ) -> AsyncContextManager[object]: ...
```

`resource_revision` may be a secret-manager version, immutable application resource
generation, deployment artifact digest, or another resolver-enforced revision. It
must not be a label whose target can silently change.

The bundle need not prove two secret byte strings are equal. It must prove that each
route requested the same resolver contract and immutable revision recorded in the
manifest. The builder also records a resolver-issued capacity owner ID and the
coordinator domain that actually enforces it. All routes in one group use that owner;
for a multi-resource bundle it represents an explicit aggregate capacity pool.

### 5.3 Canonical digest

Recommended digest:

```text
sha256:<SHA-256(canonical UTF-8 JSON)>
```

Canonical JSON uses sorted object keys, no insignificant whitespace, strict JSON,
and no NaN or infinity.

Include:

- digest schema version;
- deployment key;
- concrete routes and fallback order;
- capacity key and maximum concurrency;
- exact resource bundle including every route mapping, resolver, and revision;
- portable resource ownership and source capability mode;
- capacity owner ID and coordinator domain;
- explicitly digest-bearing semantic metadata.

Exclude:

- opaque snapshot ID (identity is stored beside, not inside, the content digest);
- timestamps;
- live object identity;
- credentials and secret values;
- observational metadata.

## 6. Registry model

### 6.1 Records

Logical records, whether implemented as dictionaries or durable storage:

```text
profiles[scope_id, group_name, profile] -> current immutable snapshot
defaults[scope_id, group_name] -> profile
admissions[admission_id] -> exact snapshot manifest
resident_sources[snapshot_id] -> borrowed/owned live source
```

Publication performs provider-neutral validation and creates a new opaque snapshot
identity. Reconfiguring one profile changes only that profile's current snapshot;
existing admission manifests retain the complete older snapshot. Developers neither
pass versions nor perform CAS.

### 6.2 Series behavior

Snapshot IDs and digests are framework-owned and never user selection values. The
application saves stable profile names; history and Worker protocols save exact
snapshot manifests. A stale snapshot can satisfy only an admission that already
retains that exact manifest.

### 6.3 Atomic read

Admission reads all current pointers for one boundary under one registry snapshot or
lock and either pins every requirement or none. Readers observe complete immutable
records, never a mixture of route configuration and another version's resource
source.

This is read consistency, not a coordinated release transaction. Different group
keys may contribute independently published versions from that one snapshot. The
first implementation does not add cross-group release sets.

## 7. Admission commit protocol

### 7.1 Non-durable admission

Recommended sequence:

```text
allocate admission_id
    -> enumerate typed requirements for boundary
    -> atomically resolve and retain candidate pins
    -> attach complete pin set to execution/Child record
    -> mark boundary admitted
    -> start boundary work
```

Failure releases all candidate pins. Terminalization releases the boundary's active
pins.

### 7.2 Durable root admission

Durable idempotency must be resolved before a new current snapshot can become the
logical execution's committed pin set.

Recommended logical protocol:

```text
begin durable execution admission using idempotency identity
    -> existing logical execution:
         load and validate its committed manifests; never read current
    -> new logical execution:
         reserve execution identity
         resolve complete current pin set
         commit execution row + admission row + manifests atomically when one
         store owns them
         otherwise use durable intent/commit records and compensating pin release
    -> only a committed admission may start forward()
```

If registry and history cannot share a transaction, the implementation must define
recoverable states such as `PREPARING`, `COMMITTED`, and `ABORTED`. Recovery either
finishes a committed intent with the recorded exact refs or removes an uncommitted
orphan. It never re-resolves current for an existing logical execution.

### 7.3 Independent Child admission

For a durable Parent, the Child admission identity and selected manifests must be
recorded using deterministic module occurrence before Child work begins. Replay of
the same occurrence loads the same Child pins. A later occurrence is a new admission
and may select current.

## 8. Infrastructure and ModelCallLayer

Recommended additive Infrastructure operations:

```python
def resolve_model_deployment(
    model_group: str,
) -> ResolvedModelDeployment: ...

def acquire_model_deployment_lease(
    deployment: ResolvedModelDeployment,
) -> AsyncContextManager[LiveModelDeploymentLease]: ...
```

Concrete fixed groups retain `resolve_model_invoker()`.

Recommended deferred call order:

```text
resolve portable deployment from active admission pins
    -> compute effective tools
    -> construct portable managed-effect request
    -> execute_effect performs history selection
        -> REPLAYED: return committed value; no permit or live lease
        -> EXECUTE/RETRY:
             release runnable execution lease
             acquire physical model capacity
             acquire exact-version live deployment lease
             invoke ModelInvoker with resolved concrete group
             release live lease and capacity
             resume runnable scheduling
```

The implementation acquires model capacity and any resolver lease inside the
`execute_effect` operation for both deferred and fixed managed calls, so replay
returns before either acquisition. The public fixed-model call shape is unchanged.

The effect request includes exact snapshot/resource digests, concrete
routes/fallback, retry, effective generation, tools, message, and relevant Context
projection.

## 9. Capacity ownership

Deferred `capacity_key` and `max_concurrency` are Agent requirement fields and enter
plan identity. Every published snapshot must match them.

`capacity_key` is the stable logical declaration checked against the Agent plan. For
deferred deployments, the actual coordinator gate is keyed by
`(capacity_coordinator_domain, capacity_owner_id)`, never by deployment snapshot.
Publishing a replacement against the same capacity-limited resource therefore reuses its gate;
publishing against a genuinely different resource may use a different owner while
the Agent's declared limit remains unchanged.

If two deployment scopes share a capacity-limited resource, preparation maps them to
the same resolver-issued `capacity_owner_id`. One group still acquires one stable
gate; a group spanning several resources therefore needs an explicit aggregate owner
rather than several Runtime gates. The coordinator domain is part of the portable
deployment contract. Publication rejects conflicting limits for one owner in one
domain. A Runtime-local gate cannot be described as cross-Runtime capacity; that
claim requires an external coordinator.

Capacity acquisition follows the existing no-hold-and-wait rule: release runnable
lease before waiting for model capacity; release model capacity before scheduling
RESUME.

## 10. Live-source lifecycle

### 10.1 Separate counters

Track at least:

- active pins: currently live admission boundaries;
- recoverable references: manifests retained with durable execution history;
- live leases: invoker/client instances currently executing Provider work.

These counters have separate release paths.

### 10.2 Source capability modes

A source should declare one of:

```text
resident-only
    The same live instance must remain available while any active or recoverable pin
    exists. It cannot satisfy process-restart recovery unless an external owner keeps
    it available.

reconstructable
    An idle instance may close after its final live lease, while the exact resource
    revision and factory remain capable of creating a later instance.
```

Ownership is independently `owned` or `borrowed`.

This avoids the invalid rule that every retired invoker closes at zero live leases.
A resident-only owned source closes after its final relevant pin and live lease. A
reconstructable source may close each idle instance while retaining the exact source
contract.

### 10.3 Replacement

After a new snapshot becomes current, the old snapshot is unavailable for ordinary admission but may remain
`AVAILABLE` for pinned work. It becomes `UNAVAILABLE` only after no active
or recoverable reference remains, or after an explicitly reported availability
failure. Purging a durable execution record releases its recoverable reference and
the corresponding resource-retention obligation.

Close failures are observable and never reactivate a retired version. Shared owner
records prevent double close across deployment snapshots and legacy registrations.

### 10.4 Runtime close

Recommended order:

```text
stop new starts and publications
    -> choose drain or cancel according to Runtime policy
    -> allow draining executions to acquire resources guaranteed by their pins
    -> wait for/cancel active model operations
    -> release in-memory active pins
    -> persist or hand off durable manifest ownership
    -> close owned live sources after final leases
    -> return after owned cleanup completes
```

Setting Runtime to closing must not make already-draining pinned calls fail merely
because they acquire their first live lease after shutdown began.

## 11. Recovery and history

History must retain, or durably reference, the complete admission manifest. Suggested
logical records:

```text
execution_admissions(execution_id, admission_id, scope_id, status)
model_deployment_manifests(admission_id, group_name, snapshot_id,
                           digest, resource_bundle_digest, resource_bundle, capacity_owner,
                           concrete_group)
```

Recovery order:

```text
claim logical execution
    -> validate ExecutionPlan/code compatibility
    -> restore committed admission manifests
    -> replay forward()
    -> for completed model effect: validate request and return history
    -> for missing/replay-safe model work: acquire original exact resource bundle
    -> if unavailable: fail closed
```

Required durability needs a capability equivalent to exact model deployment
recovery. Bind may validate structural Runtime capability; start/admission validates
that the scope, manifest, and exact resource bundle remain recoverable for the same
lifetime as the retained execution record. `preferred` reports degradation
explicitly.

## 12. Remote placement

The first implementation should reject deferred requirements in any boundary that
may execute remotely unless both protocol and target advertise exact-pin resolution.

A later start request should carry authenticated pin values or a coordinator-owned
manifest reference. The Worker validates:

- plan and graph identity;
- admission and deployment scope;
- group, snapshot ID, snapshot/resource digest, and store namespace;
- complete resource-bundle digest and exact route revisions;
- capacity owner and coordinator domain;
- required capabilities and placement authority.

The Worker never substitutes its local current pointer.

## 13. Events and errors

Do not add deployment data to existing closed `model.*` payloads. Put correlation in
execution/span metadata or introduce an explicitly versioned control-plane event.

Admission failures occur before model events. Publication and recovery errors remain
deployment errors rather than Provider failures. Error rendering must use existing
redaction and bounded-payload rules.

## 14. Suggested implementation phases

### Phase 1: definition and plan contracts

- add deferred construction and invariants;
- reject deferred local invokers and direct execution;
- add Runtime-issued deployment scope identity;
- add typed plan requirements and admission-boundary partitioning;
- enforce graph closure;
- update plan/Worker/history fixtures for the schema change.

### Phase 2: process-local non-durable execution

- add preparation boundary and exact per-route resource bundles;
- add process-local registry, immutable profile publication, and atomic boundary pinning;
- resolve deferred calls from pins;
- move capacity/live lease acquisition inside effect operation;
- support replacement and independent Child admissions;
- implement lifecycle and shutdown tests.

### Phase 3: durable exact-version recovery

- define and implement the durable admission commit protocol;
- persist manifests by admission identity;
- persist durable scope mappings and exact resource bundles;
- add resolver revision retention and exact reacquisition tied to history retention;
- integrate idempotent duplicate start and recovery;
- add crash-point, retirement, and history-retention tests.

### Phase 4: application facade

- expose requirement-aware configuration builders;
- support raw-key consumption without portable secret leakage;
- integrate optional catalog preflight outside Runtime.

### Phase 5: distributed capability

- define authenticated pin serialization;
- negotiate exact-resolution capability;
- add coordinator or external resolver support;
- validate pins on Worker admission.

## 15. Required verification matrix

At minimum, tests should cover:

- fixed direct and fixed managed regression examples;
- invalid concrete/deferred states;
- same group name across Binding scopes;
- atomic multi-group admission failure;
- multi-group admission may contain independently published snapshots;
- undeclared dynamic call rejection;
- an execution continuing on its old snapshot after profile reconfiguration;
- repeated independent Child admissions selecting old then new snapshots;
- replay taking no model permit and no live lease;
- shared-resource and multi-Provider route bundles;
- mutable alias rejection and exact resource-bundle recovery;
- durable scope identity across Runtime restart;
- concurrent publication/default/retirement serialization;
- idempotent duplicate start after publication selecting original pins;
- every durable admission crash point;
- resident-only and reconstructable retirement;
- shared ownership and exactly-once close;
- cancellation and shutdown while waiting/acquiring/executing;
- unsupported remote placement rejection;
- event-schema and secret-redaction regression;
- decoding or migrating every still-supported fixed-model plan/history schema.

## 16. Open technical choices

The following may be decided during implementation without changing the governing
specification:

- whether the public Python type is one tagged `ModelGroupConfig` or separate
  concrete/deferred frozen subclasses;
- exact names and module locations of deployment APIs;
- process-local registry lock and data structures;
- durable table layout and whether registry/history share one store;
- the durable intent/commit mechanism when they do not share a transaction;
- the concrete format of deployment scope, snapshot, admission, and resource revision
  IDs;
- idle eviction policy for reconstructable live instances;
- control-plane observability API;
- future distributed registry and resolver protocols.

These choices must still satisfy every frozen second principle and acceptance test.
