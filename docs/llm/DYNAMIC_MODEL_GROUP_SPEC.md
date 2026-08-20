# Deferred and Dynamic Model Group Specification

- Status: Proposed
- Target: Pygent post-0.2 design review
- Audience: framework maintainers, Runtime implementers, Agent application developers
- Scope: `pygent.llm`, `pygent.runtime`, managed Agent execution
- Normative language: MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are requirements keywords
- Implementation design: [Dynamic Model Group Implementation Design](DYNAMIC_MODEL_GROUP_IMPLEMENTATION.md)

## 1. Purpose

Pygent SHALL allow an Agent definition to declare a named model requirement before
any concrete model, endpoint, credential, client, or `ModelInvoker` exists. An
application may later prepare and publish a concrete deployment for that requirement
inside one Binding governance scope.

Publishing a replacement creates a new immutable model-deployment snapshot. A new
managed admission may select the new snapshot. An already admitted execution or
Child admission boundary remains pinned to the exact snapshot it selected.

This document defines stable behavior and public contracts. Registry layout,
locking, transaction implementation, storage tables, concrete internal class names,
and cleanup algorithms belong to the implementation design document.

### 1.1 Snapshot identity is not an application version

Every successful profile configuration creates an opaque immutable snapshot ID and
digest owned by the framework. Applications do not assign, compare, persist, or
increment deployment versions; they persist only stable profile names when needed.
Snapshot identity is distinct from the Pygent package version, SDK version, Provider
model name, and ExecutionPlan schema version.

For example:

```text
assistant profile before update: qwen3-32b -> deepseek-v3
assistant profile after update: gpt-5 -> qwen3-32b
```

An execution admitted before the update continues to use its exact snapshot. A later
execution may use the new snapshot.

## 2. Governing first principles

This specification is subordinate to:

1. the repository-wide [Pygent 0.2 first principles](../FEATURES.md);
2. the [LLM first principles](FEATURES.md);
3. the [Runtime first principles](../runtime/FEATURES.md);
4. the existing LLM and Runtime SDK contracts.

If this specification can be interpreted in a way that conflicts with a governing
first principle, that interpretation is invalid. The first principles remain the
authority; this specification may clarify their application but may not weaken or
replace them.

The feature therefore preserves all of the following:

- `ModelCallLayer` remains an ordinary stateless `Module`.
- `Message`, `Context`, `forward()`, direct execution, Binding, managed execution,
  streaming, and final result shapes remain unchanged.
- Requirements, concrete deployments, retry, fallback, and generation values are
  immutable values; publication never mutates an Agent definition.
- `ModelInvoker` retains route, retry, and fallback execution behavior.
- Provider request construction and response interpretation remain in the LLM
  adapter and invoker, never in Runtime.
- Binding remains the deployment and resource-governance boundary.
- Dynamic publication cannot add Modules, alter graph control flow, or make an
  undeclared call site executable.
- Managed effect replay remains the durable model-call replay boundary.
- Existing closed `model.*` event schemas remain authoritative.
- Existing fixed-model direct and managed SDK paths remain complete paths.

## 3. Frozen second principles

The following are the second principles of this specification. They are subordinate
to the governing first principles, but are otherwise frozen for this feature.
Implementation choices, later sections, and later revisions MUST NOT weaken or
contradict them. Changing one requires an explicit design review that first proves
continued compliance with every governing first principle and existing fixed-model
SDK contract.

1. **Requirement and deployment are distinct immutable values.** A deferred model
   group in an Agent is a portable requirement. A published model deployment is a
   complete immutable snapshot. Publication changes only a Binding-scoped current
   pointer; it never mutates an Agent, Module, requirement, deployment snapshot, or
   ExecutionPlan.
2. **Admission chooses; execution does not chase latest.** Every effective Binding
   admission boundary resolves all of its deferred requirements before model work
   starts. Each boundary pins the exact selected snapshots. Model calls and recovery
   use those pins and never consult the current pointer.
3. **Binding isolation uses Runtime-issued stable identity.** Publication and
   resolution authority is the complete `(deployment_scope_id, model_group_name)`
   key. Display names, raw tenant values, Python object addresses, and Runtime-global
   group-name lookup are not dynamic-deployment authority. A durable scope identity
   remains stable for the lifetime of its retained execution history.
4. **Pins are namespaced by admission boundary.** An inherited local subgraph shares
   its Parent admission and pins. A Child with an independent Binding receives a new
   admission boundary when that Child execution starts. Multiple Child admissions
   in one Root may therefore pin different deployment snapshots without overwriting
   one another.
5. **Exact deployment includes exact route resources.** A pin identifies the
   concrete routes, fallback, capacity declaration, semantic metadata, and one
   immutable resource revision for every route. A shared revision may cover several
   routes. A mutable secret name or resolver alias by itself is not an exact resource
   reference. Secrets remain outside portable values.
6. **Portable pins, exact-version availability, and live leases have separate
   lifetimes.** A pin preserves portable identity. Exact-version availability keeps
   the original deployment resolvable while an active pin or retained recoverable
   manifest depends on it. A
   live lease protects an invoker/client only while Provider work is occurring.
   Retirement may close an idle instance only when later pinned work can still
   reacquire the exact resource; otherwise the instance remains available until the
   relevant pins are released.
7. **Replay precedes scarce live resources.** Admission and effect replay use only
   portable values. A live deployment lease and model capacity are acquired only
   after replay determines that new Provider work is required.
8. **Durable identity is committed consistently.** Durable execution identity,
   admission pins, idempotency resolution, and the retained manifest have one
   defined commit protocol. A crash or duplicate start cannot silently replace an
   original pin with current or leave an execution recoverable without its manifest.
9. **Runtime remains Provider-neutral.** Runtime may validate portable identity,
   schema, digest, scope, publication concurrency, ownership, and lease state. It never validates
   Provider capabilities, constructs Provider clients, interprets Provider
   configuration, or claims arbitrary text is secret-free.
10. **Capacity retains one stable enforceable owner per snapshot.** A deployment
    publication cannot change the requirement's logical `capacity_key` or
    `max_concurrency`. Each snapshot maps that declaration to one resolver-issued
    capacity owner. Snapshots that share a capacity-limited resource must map to the
    same owner and cannot obtain independent gates that multiply the declared limit.
    A multi-resource route bundle uses one explicitly enforced aggregate owner.
11. **The declared graph is the authority.** Only typed deferred requirements found
    in the admitted ExecutionPlan boundary may resolve. Runtime rejects an unplanned
    deferred call even if a registry entry with the same name exists.
12. **Fixed-model SDK behavior remains a separate complete path.** A dynamic
    publication cannot override a concrete Layer invoker or fixed managed
    registration merely because a group name matches.

## 4. Scope

### 4.1 Goals

The feature MUST:

1. allow Agent construction before concrete model configuration;
2. publish complete immutable deployment snapshots atomically;
3. isolate publication and lookup by stable Binding governance scope;
4. fail managed admission when a required deferred group is unresolved;
5. pin one exact deployment per requirement per admission boundary;
6. support replacement, route addition, removal, and reorder for new admissions;
7. keep secrets and live objects out of Agent definitions, Context, ExecutionPlan,
   history payloads, and public events;
8. preserve deterministic managed-effect identity and replay;
9. define unambiguous ownership and shutdown behavior;
10. preserve fixed-model direct and managed usage;
11. fail closed when remote placement cannot resolve an authenticated exact pin.

### 4.2 Non-goals

The first implementation does not need to:

- choose a different deployment before every inference call;
- mutate a route list or deployment in place;
- discover Provider capabilities automatically;
- provide a cross-process registry;
- make an optional Python branch imply an optional deployment requirement;
- make undeclared ModelGroups available by name;
- change capacity identity through publication;
- support deferred direct execution;
- support remote deferred execution without exact-pin capability.
- atomically publish a coordinated release across several model-group keys.

## 5. Public concepts

### 5.1 Deferred ModelGroup requirement

A deferred group is a named, immutable Agent requirement with no concrete routes or
fallback order. It is portable but not independently executable.

### 5.2 Concrete ModelGroup

A concrete group is the existing immutable `ModelGroupConfig` behavior with one or
more validated routes and a valid fallback policy.

### 5.3 Deployment scope and key

`Binding` remains the existing immutable policy value. A Runtime-issued
`RuntimeBinding` (exact name may vary) is the authority for one effective Binding
governance scope and carries an opaque, portable `deployment_scope_id`. The model
deployment key is:

```text
(deployment_scope_id, model_group_name)
```

A `RuntimeBinding` may bind Modules and authorize publication, but its scope ID alone
is not a bearer credential. A raw `Binding` policy value never authorizes
publication.

A structured Child inheriting its Parent Binding inherits the same scope. A
pre-bound or remotely placed Child with an independent Binding has an independent
scope. Explicit shared scope requires an owner capable of enforcing that sharing.
For required durability, Runtime MUST persist or externally resolve the same scope
identity for as long as the corresponding execution history is retained.

### 5.4 Prepared and published deployment

The LLM/application deployment layer prepares:

- one concrete `ModelGroupConfig`;
- one typed exact resource bundle mapping every route to a resolver identity and
  immutable resource revision; several routes may share one bundle entry;
- a live-resource source with explicit resident or reconstructable availability and
  owned or borrowed ownership for every resource entry;
- a resolver-issued capacity owner identity and enforcement domain;
- optional strict, non-secret semantic metadata.

Runtime publishes an immutable record containing the complete deployment key, opaque
snapshot ID, canonical digest, prepared portable content, and exact resource source.

The canonical route projection always contains `route_id`, `provider`, `model`, and
`provider_options`, using an empty object when unset. The decoder no longer supplies
the historical missing-field default, and options are part of snapshot identity.

### 5.5 Pinned deployment reference

A portable pin contains at least:

```text
admission_id
deployment_scope_id
model_group_name
profile
snapshot_id
deployment_digest
resource_bundle_digest
capacity_owner_id
capacity_coordinator_domain
```

The referenced resource bundle contains the exact per-route resource identities and
revisions. The compact pin may contain that bundle inline or address an immutable
manifest by digest.

It contains no credential, endpoint secret, client, invoker, callback, lock, task,
lease, or arbitrary Python object.

### 5.6 Live deployment lease

A live lease is non-portable and bounded. It exposes the exact concrete group and
`ModelInvoker` required for one unit of actual model work. It is released on success,
failure, cancellation, or deadline.

## 6. Public SDK

Exact names may be refined before implementation, but the following call shapes and
boundaries are normative.

The ordinary application model is intentionally small: declare one deferred group,
obtain its Binding-scoped handle, configure named profiles, and invoke normally. A
call may select one permitted profile and permitted generation values. Scope IDs,
snapshot IDs/digests, pins, resource bundles, leases, and retirement are framework
concerns and are not required inputs in the common SDK path.

### 6.1 Declare the requirement and call policy

```python
assistant_group = ModelGroupConfig.deferred(
    name="assistant",
    max_concurrency=8,
    capacity_key="assistant-model",
)

model = ModelCallLayer(
    model_group=assistant_group,
    policy=ModelCallPolicy(
        allow_profile_override=True,
        overridable_generation=frozenset({"temperature", "max_output_tokens"}),
    ),
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=2048),
    tools=tools.definitions,
)

agent = MyAgent(model=model, tools=tools)
```

The ordinary `ModelGroupConfig` constructor remains concrete by default. An empty
concrete group and a non-empty deferred group are invalid. A deferred
`ModelCallLayer` MUST reject a local `invoker=` argument.

`ModelCallPolicy` is an immutable portable declaration. It controls whether a Root
execution may select a profile and which generation fields may vary. Session meaning
belongs to the application: a sticky selection is implemented by repeating the saved
profile in each Root `ExecutionOptions`. The policy
MUST NOT authorize route, credential, client, retry, or fallback overrides. The
`RetryPolicy` remains part of the Layer definition.

### 6.2 Configure named profiles through a handle

```python
runtime_binding = runtime.create_binding(...)
group = runtime_binding.model_groups.get(assistant_group)

await group.ensure_profile(
    profile="balanced",
    routes=(
        ModelRoute("primary", provider="openai", model=user_model),
    ),
    fallback=FallbackPolicy(("primary",)),
    invoker=balanced_invoker,
    resource_ref=resource_resolver.ref(
        "tenant-42/openai-primary",
        revision=credential_revision,
    ),
    make_default=True,
    deadline=configuration_deadline,
)

await group.ensure_profile(
    profile="quality",
    routes=(
        ModelRoute("primary", provider="openai", model="gpt-5"),
        ModelRoute("fallback", provider="openai", model=user_model),
    ),
    fallback=FallbackPolicy(("primary", "fallback")),
    invoker=quality_invoker,
    resource_ref=resource_resolver.ref(
        "tenant-42/openai-quality",
        revision=quality_credential_revision,
    ),
    deadline=configuration_deadline,
)
```

`ModelGroupHandle` is a mutable control-plane handle, not a Module dependency or
portable value. `get()` MUST accept the typed requirement, not grant authority from a
user-supplied group name alone. The handle derives the immutable capacity declaration
from that requirement, so ordinary callers do not repeat it.

`ensure_profile()` performs Provider-specific validation outside Runtime, followed by
Runtime validation of declaration compatibility, exact resource revisions, capacity
ownership, and publication safety. It normalizes the complete configuration and uses
its stable digest as the idempotency identity. Concurrent calls for the same scope,
profile, and digest single-flight and return the same immutable snapshot; different
digests serialize publication. `make_default=True` publishes the profile current
pointer and default pointer in the same transaction. Failure publishes nothing. Store
opening, validation, resource preparation, and publication all obey the explicit
control-plane deadline and caller cancellation. The user does not supply or retain a
framework snapshot version.

### 6.3 Invoke with the default or a temporary selection

```python
bound_agent = runtime_binding.bind(agent)

message, context = await bound_agent.invoke(message, context, execution=run)

message, context = await bound_agent.invoke(
    message,
    context,
    execution=ExecutionOptions(
        deadline=deadline,
        model_calls={
            "assistant": ModelCallOptions(
                profile="quality",
                temperature=0.1,
                max_output_tokens=4096,
            ),
        },
    ),
)
```

The first call selects the group default. The second call selects `quality` only for
that Root and overrides only fields allowed by `ModelCallPolicy`. `model_calls` MUST
name a deferred group declared in the admitted ExecutionPlan. It cannot add a group
or change routes, credentials, clients, retry, or fallback.

At admission, Runtime resolves the selected profile to one exact immutable snapshot
and validates all generation overrides before `forward()`. Retry and fallback remain
inside that snapshot and never switch to another profile. The resolved snapshot and
effective generation values participate in model effect identity.

### 6.4 Application-managed session affinity

Pygent does not require a mutable model choice inside an Agent instance. If an
application wants one conversation to keep a profile, it stores the profile name in
its own session state and passes it on each Root invocation:

```python
session_profile = conversation.model_profile  # for example, "quality"

message, context = await bound_agent.invoke(
    message,
    context,
    execution=ExecutionOptions(
        deadline=deadline,
        model_calls={
            "assistant": ModelCallOptions(profile=session_profile),
        },
    ),
)
```

The application value is a selection request, not a framework snapshot version.
Every admission pins the exact snapshot selected at that time. An SDK convenience
session facade MAY carry this option between calls, but MUST preserve the same
semantics and MUST NOT mutate the Agent or group default.

### 6.5 Multiple Agent instances

Multiple Agent instances are valid and may share one deferred requirement and one
Binding-scoped profile set:

```python
agent_1 = runtime_binding.bind(MyAgent(model=model, tools=tools))
agent_2 = runtime_binding.bind(MyAgent(model=model, tools=tools))

await agent_1.invoke(message_1, context_1, execution=options_1)
await agent_2.invoke(message_2, context_2, execution=options_2)
```

They do not need one ModelGroup per conversation. A separate Binding is appropriate
only when authorization, capacity ownership, deployment lifecycle, or another
governance boundary must be isolated. Assigning `agent.group = group` is invalid:
the handle is live control-plane state and the Agent definition is immutable.

### 6.6 Update and inspection

Changing the default affects only later admissions that do not select a profile:

```python
await group.set_default("quality", deadline=configuration_deadline)

current = await group.current("quality")
profiles = await group.list_profiles()
available = await group.available_models(resource_ref=quality_resource_ref)
```

Existing admissions retain their pins. `current()` returns an opaque snapshot
description for diagnostics; application logic does not persist its internal version.
`available_models()` is discovery only. It MUST NOT create or update a profile, change
the default, or bypass Provider and Runtime validation.

### 6.7 Application facade

An application facade may wrap the handle without exposing resource details:

```python
await app.models.ensure_profile(
    group=group,
    profile="balanced",
    provider="openai-compatible",
    base_url=user_base_url,
    credential=CredentialRef(
        "tenant-42/openai-primary",
        revision=credential_revision,
    ),
    models=[
        ModelSelection(id="qwen3-32b", role="primary"),
        ModelSelection(id="deepseek-v3", role="fallback"),
    ],
    deadline=configuration_deadline,
)
```

If a facade accepts a raw API key, it consumes it only while creating an exact
resource revision. It MUST NOT copy the key into portable metadata, digest input,
exceptions, events, Context, ExecutionPlan, or history.

The common case remains one endpoint and credential shared by all routes in a
profile. A profile with routes backed by different resources uses an advanced
per-route resource map; the facade may hide that map from ordinary Agent code.

## 7. Validation and authority

### 7.1 LLM/application preparation

Before producing a prepared deployment, the LLM/application layer MUST validate:

1. the group is concrete and has at least one route;
2. route IDs are unique and fallback entries reference them;
3. the chosen invoker/adapters support every declared Provider route;
4. every route maps to a suitable client through the exact resource bundle;
5. Provider-specific configuration is structurally valid;
6. portable metadata contains no secret or live resource;
7. every resource reference identifies an immutable revision for the required
   availability lifetime;
8. the capacity owner identity is stable and is enforced in its declared
   coordinator domain.

Route validation completes before canonical digest generation. A resident invoker
validates through `ModelProviderRouteValidator`; a reconstructable resource uses
`ModelResourceResolver.validate()` under the same contract. Non-empty options with
no validating path fail publication before Provider I/O.

A model catalog lookup may be used as preflight but does not prove future
availability and does not mutate a deployment.

### 7.2 Runtime publication

Runtime validates only provider-neutral facts:

1. Runtime is open and accepts publications;
2. publication authority belongs to this Runtime and resolves a valid deployment
   scope;
3. the group is concrete;
4. the live source exposes the generic invoker lease contract;
5. the resource bundle, capacity owner identity, and metadata are strict portable
   values within schema and size limits;
6. the canonical digest matches the portable content;
7. the publication/default/retirement operation is serialized against current state;
8. ownership, availability mode, and close responsibility are unambiguous;
9. exact-version availability covers active pins and, when durability requires it,
   the lifetime of retained recoverable manifests.

At admission, every deployment MUST exactly match the requirement's name,
`capacity_key`, and `max_concurrency`.

Plan compilation MUST reject two deferred declarations in one effective Binding
boundary that reuse a group name with incompatible capacity declarations.

Runtime does not query Provider catalogs, inspect credentials, construct Provider
clients, interpret route configuration, or infer that arbitrary strings are
secret-free.

## 8. Admission and execution semantics

### 8.1 Effective admission boundary

One admission boundary contains the Root and every local structured Child inheriting
the same effective Binding. Admission enumerates all typed deferred requirements in
that boundary and atomically resolves a complete pin set.

The registry read is atomic, but versions of different model-group keys need not
belong to one coordinated release. An admission may therefore pin independently
published `planner` and `executor` snapshots that are current in one registry read.
Applications that
require coordinated multi-group rollout should publish behind a new Binding scope;
coordinated release sets are outside the first implementation.

A Child with an independent Binding is outside the Parent boundary. Each invocation
of that Child creates its own admission identity and pin set when the Child execution
starts.

If any required deployment is missing or incompatible, admission fails before:

- user `forward()` work in that boundary starts;
- model capacity or a live model lease is acquired;
- a Provider request is sent;
- `model.started` is emitted;
- a model effect record is created.

All declared deferred groups in a boundary are required. Static graph inspection
does not guess that a Python branch is unreachable. Optional configuration requires
a separate Root/plan or independently admitted Child.

### 8.2 Resolution during `forward()`

Infrastructure resolves a deferred group only from the active admission's pin set.
It rejects a group that is absent from the typed plan requirements or pin set. It
never falls back to registry current or a Runtime-global name.

The resolved portable deployment supplies the concrete group used to construct the
managed effect request. After effect replay selects new work, the operation acquires
the appropriate capacity permit and an exact-version live lease.

### 8.3 Direct execution

Deferred groups are managed-only initially. Direct execution fails before Provider
work. Concrete Layers with local invokers continue to support direct execution
unchanged.

## 9. ExecutionPlan, effects, and durability

ExecutionPlan records typed deferred requirements, including resolution state,
group name, capacity key, and maximum concurrency. It does not contain the current
deployment, resource revision, endpoint, credential, client, or invoker.

Resolved manifests live beside the immutable plan in execution metadata. They are
namespaced by admission identity and contain the complete portable pin plus concrete
routes, fallback, exact resource bundle, capacity owner and declaration, and
digest-bearing semantic metadata.

The model effect request is built from the resolved manifest, retry policy,
generation policy, tools, message, and relevant Context projection. It includes the
pin identity so a replay reaching the same module occurrence with a different
deployment deterministically conflicts.

Recovery restores the committed original admission pins. It never resolves current.
A committed effect replays without a live resource. New model work requires the
exact pinned resource bundle. If any required revision is unavailable, recovery
fails with `ModelDeploymentUnavailableError`.

Required durability is admitted only when Runtime can retain every manifest and
reacquire every exact route resource for as long as its durable execution record
is retained. Purging that execution record releases the recoverable manifest and its
resource-retention obligation. Preferred durability may explicitly report a narrower
level. A process-local current pointer or Runtime-lifetime scope ID alone is not
durable deployment storage.

The ExecutionPlan schema must be versioned when typed requirements are added.
Runtime MUST either decode every still-supported fixed-model plan/history schema or
provide an explicit migration before dropping it. A schema bump must not silently
make a retained fixed-model execution unrecoverable. Fixed-model plan semantics and
SDK source shapes remain unchanged.

## 10. Publication, retirement, and shutdown

Publication atomically changes one profile's current snapshot. Readers observe
either the complete old snapshot or the complete new snapshot. The framework creates
the opaque identity and performs concurrency control internally; SDK callers do not
provide a version or CAS token.

Opening a shared deployment store is single-flight per Runtime. Publication is also
single-flight per `(deployment_scope_id, requirement_id, profile, config_digest)`.
Cancellation of one waiter does not cancel work still required by other waiters, while
every waiter remains bounded by its own control-plane deadline. Failed initialization
or publication removes the in-flight entry so a later call can retry.

A retired version accepts no new ordinary admissions. Existing active pins remain
valid until their admission ends; recoverable pins remain valid while their durable
execution records are retained. Retirement may close an
idle live instance only when the exact-version source remains able to serve every
such pin. Otherwise the required resource remains available until those pins are
released.

Every prepared resource declares one ownership mode:

- owned: the deployment owner is the sole close owner;
- borrowed: an external owner closes it and Runtime never does.

Shared owned resources use explicit owner records or reference-counted leases and
close exactly once. Object identity may optimize an in-process implementation but is
not the portable contract.

The resolver-issued capacity owner identity is independent of Binding scope and
deployment snapshot. Routes in one group use one owner; when they span resources,
that owner represents an explicitly enforced aggregate pool. Conflicting limits for
one owner in one coordinator domain fail publication. Cross-Runtime ownership may be
advertised only when an external coordinator enforces it.

Runtime close stops new execution admissions and publications before draining or
cancelling active work. It does not revoke resources still needed by work it has
chosen to drain. After active work no longer uses model resources, Runtime releases
in-memory pins, preserves externally retained durable manifests, and closes owned
resources exactly once.

## 11. Placement, security, and observability

The initial registry may be process-local but must report that limitation honestly.
A deferred call that may execute on another Runtime is rejected unless the remote
protocol can authenticate, validate, and resolve the exact supplied pin.

A capable Worker validates plan identity, admission scope, snapshot/resource digest,
exact resource revisions, capacity owner/domain, deployment-store namespace, and the
`model.deferred.exact-pin.v1` capability. It never discards a pin to query local
current. Failover is limited to Workers advertising the same store namespace.

When any pinned route has non-empty provider options, remote deployment also requires
`model.route.provider-options.v1`. The Worker codec preserves the canonical route
projection losslessly, and a Worker lacking that capability rejects placement or
admission before resource acquisition and Provider I/O.

Secrets, raw credential configuration, clients, invokers, transport exception
bodies, callbacks, locks, tasks, and leases never enter Agent state or portable
records. Public correlation fields remain subject to policy, redaction, schema, and
size limits.

Publication and retirement are control-plane facts, not model inference events. The
closed `model.*` vocabulary remains unchanged unless separately versioned. An
unresolved group produces an admission error, not `model.started` followed by
`model.failed`.

## 12. Errors

The public API provides `ModelGroupError` with configuration, selection,
unavailable, and conflict specializations:

```python
ModelGroupConfigurationError
ModelProfileSelectionError
ModelDeploymentConflictError
ModelDeploymentUnavailableError
```

Configuration, scope, publication, placement, and recovery errors are not
`ModelProviderError` or `ModelCallError`. Provider errors begin only after a valid
deployment, live lease, and Provider attempt exist.

## 13. Fixed-model SDK behavior

The existing concrete usage remains valid and unchanged:

```python
model = ModelCallLayer(
    model_group=ModelGroupConfig(
        name="assistant",
        routes=(ModelRoute("primary", "openai", "gpt-5"),),
        fallback=FallbackPolicy(("primary",)),
    ),
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(),
    invoker=local_invoker,
)
```

Existing fixed managed registration also remains valid:

```python
runtime.register_model_invoker("assistant", invoker)
```

Resolution precedence is explicit:

- concrete group plus Layer invoker: fixed local/direct resource path;
- concrete group without Layer invoker: fixed managed registration path;
- deferred group: Binding-scoped versioned deployment path.

Empty concrete groups remain invalid. Dynamic publication never overrides either
fixed path by name.

## 14. Acceptance criteria

### Definition and SDK

- Deferred Agent construction requires no live model resource.
- Empty concrete, non-empty deferred, and deferred-plus-local-invoker states fail.
- Existing fixed direct and managed examples execute unchanged.
- Dynamic examples use matching requirement and deployment capacity declarations.
- No secret or live object appears in portable values.

### Scope and admission

- Missing or incompatible required groups fail before boundary execution and model
  effects.
- Binding A publication cannot resolve Binding B.
- Inherited Children share their Parent admission pins.
- Each independent-Binding Child invocation has its own admission identity.
- A second Child admission may select a newer deployment without changing an earlier
  Child admission.
- An unplanned deferred call fails even when the registry contains the same name.
- Incompatible duplicate declarations for one group name fail plan compilation.
- A durable Binding scope retains the same identity across Runtime restart.

### Execution and publication

- One admission uses one exact deployment per requirement.
- Reconfiguring a profile affects only later admissions.
- Concurrent publication/default/retirement operations are serialized and never
  expose a partial snapshot.
- Replayed effects acquire neither model capacity nor live resources.
- New work uses routes and resources from the exact pin, never current.
- Physical-resource capacity is not multiplied by publication or Binding aliases.
- Multi-group admission reads one registry snapshot; coordinated cross-group release
  is not implied.

### Durability and lifecycle

- Durable identity and pin manifests survive every specified admission crash point.
- An idempotent duplicate start restores the original pins rather than selecting
  current.
- Committed effects replay without Provider resources.
- Uncommitted work requests the original exact resource bundle only.
- A retired version remains usable by active pins and retained recoverable manifests.
- Owned resources close exactly once; borrowed resources are never closed by Runtime.
- Shutdown does not revoke resources from executions it is draining.
- Every still-supported fixed-model plan/history schema remains decodable or has an
  explicit migration.

### Placement and observability

- Unsupported remote deferred placement fails closed.
- A capable Worker validates and resolves the supplied exact pin.
- Existing closed event payloads remain unchanged.
- Unresolved deployment failures are not counted as Provider failures.

## 15. Implementation boundary

The normative feature is:

```text
declare immutable deferred requirement
    -> prepare exact per-route Provider resource bundle outside Runtime
    -> publish immutable deployment under Runtime-issued Binding scope
    -> atomically admit one effective Binding boundary and persist its pins
    -> construct model effects from the pinned portable manifest
    -> replay without scarce live resources
       or lazily lease exact-version resources for new work
    -> publish replacements only for later admissions
    -> retire resources without invalidating active or recoverable pins
```

The separate implementation design defines the recommended internal types,
algorithms, commit protocol, lifecycle state machine, schema work, phased delivery,
and open technical choices.
