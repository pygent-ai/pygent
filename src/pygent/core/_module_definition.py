"""Module definitions, placement declarations, and public execution entrypoints."""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    cast,
)
from uuid import uuid4

from .definition import module_definition_config
from .execution import (
    ExecutionOptions,
)
from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    freeze_json_object,
)
from .values import Context

if TYPE_CHECKING:
    from pygent.runtime import Binding, BoundModule, Runtime

from ._direct_execution import (
    DirectExecutionStream,
    _direct_module_paths,
    _DirectExecutionHandle,
    _DirectExecutionRecord,
)
from ._module_contracts import (
    DirectExecutionError,
    ExecutionRequirements,
    InputMessageT,
    ModuleDependency,
    OutputMessageT,
    _execution_scope,
)


class PlacementMode(str, Enum):
    """Stable physical-placement strategy for a declared Child target."""

    INHERIT = "inherit"
    PINNED = "pinned"
    ADAPTIVE = "adaptive"


@dataclass(frozen=True, slots=True)
class PlacementPolicy:
    mode: PlacementMode
    target_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PlacementMode):
            object.__setattr__(self, "mode", PlacementMode(self.mode))
        if self.mode is PlacementMode.PINNED:
            if not isinstance(self.target_id, str) or not self.target_id:
                raise ValueError("pinned placement requires a stable target_id")
        elif self.target_id is not None:
            raise ValueError("only pinned placement accepts target_id")

    @classmethod
    def inherit(cls) -> PlacementPolicy:
        return cls(PlacementMode.INHERIT)

    @classmethod
    def pinned(cls, target_id: str) -> PlacementPolicy:
        return cls(PlacementMode.PINNED, target_id)

    @classmethod
    def adaptive(cls) -> PlacementPolicy:
        return cls(PlacementMode.ADAPTIVE)


@dataclass(frozen=True, slots=True, kw_only=True)
class RemoteModule(Generic[InputMessageT, OutputMessageT]):
    """Stable logical deployment reference resolved only by a managed executiontime."""

    binding_ref: str
    plan_id: str | None = None
    graph_hash: str | None = None
    required_capabilities: tuple[str, ...] = ()
    placement: PlacementPolicy = field(default_factory=PlacementPolicy.adaptive)

    def __post_init__(self) -> None:
        if not isinstance(self.binding_ref, str) or not self.binding_ref:
            raise ValueError("binding_ref must be a non-empty string")
        if (self.plan_id is None) != (self.graph_hash is None):
            raise ValueError("plan_id and graph_hash must be declared together")
        if self.graph_hash is not None:
            if (
                len(self.graph_hash) != 64
                or any(character not in "0123456789abcdef" for character in self.graph_hash)
            ):
                raise ValueError("graph_hash must be a lowercase SHA-256 digest")
            if self.plan_id != f"sha256:{self.graph_hash}":
                raise ValueError("plan_id must identify the declared graph_hash")
        capabilities = tuple(self.required_capabilities)
        if any(not isinstance(item, str) or not item for item in capabilities):
            raise ValueError("required capabilities must be non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("required capabilities must be unique")
        object.__setattr__(self, "required_capabilities", capabilities)
        if not isinstance(self.placement, PlacementPolicy):
            raise TypeError("placement must be a PlacementPolicy")
        if self.placement.mode is PlacementMode.INHERIT:
            raise ValueError("RemoteModule placement cannot be inherit")

    async def __call__(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError(
                "RemoteModule calls require an active managed execution; "
                "use a bound Root entrypoint"
            )
        output, next_context = await scope.invoke_module(self, message, context)
        return cast(OutputMessageT, output), next_context


class Module(Generic[InputMessageT, OutputMessageT]):
    """Reusable definition whose subclasses implement one ``forward`` method."""

    _children: dict[str, Module[Any, Any]]
    _dependencies: dict[str, ModuleDependency[Any, Any]]
    _definition_frozen: bool
    _definition_config_snapshot: FrozenJsonObject | None
    execution_requirements = ExecutionRequirements()
    trusted_live_resource_attributes: tuple[str, ...] = ()

    def __init__(self) -> None:
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_dependencies", {})
        object.__setattr__(self, "_definition_frozen", False)
        object.__setattr__(self, "_definition_config_snapshot", None)

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get("_definition_frozen", False):
            raise RuntimeError(
                f"{type(self).__name__} definition is frozen; Module attributes "
                "cannot be rebound after invoke(), stream(), or bind()"
            )
        children = self.__dict__.get("_children")
        dependencies = self.__dict__.get("_dependencies")
        if children is not None:
            if isinstance(value, Module):
                children[name] = value
            else:
                children.pop(name, None)
        if dependencies is not None:
            is_bound_dependency = (
                callable(value)
                and isinstance(getattr(value, "module", None), Module)
                and getattr(value, "binding", None) is not None
                and getattr(value, "runtime", None) is not None
            )
            if isinstance(value, (Module, RemoteModule)) or is_bound_dependency:
                dependencies[name] = cast(ModuleDependency[Any, Any], value)
            else:
                dependencies.pop(name, None)
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self.__dict__.get("_definition_frozen", False):
            raise RuntimeError(
                f"{type(self).__name__} definition is frozen; Module attributes "
                "cannot be deleted after invoke(), stream(), or bind()"
            )
        children = self.__dict__.get("_children")
        dependencies = self.__dict__.get("_dependencies")
        if children is not None:
            children.pop(name, None)
        if dependencies is not None:
            dependencies.pop(name, None)
        object.__delattr__(self, name)

    @property
    def definition_frozen(self) -> bool:
        """Whether this definition has entered an executable lifecycle."""

        return self._definition_frozen

    def _freeze_definition(self) -> None:
        """Recursively close the raw Module graph against definition drift.

        Stored definition values are validated before any node is frozen.
        Mutable or unknown state is rejected unless its attribute is explicitly
        declared as a trusted deployment live resource.
        """

        discovered: list[Module[Any, Any]] = []
        visited: set[int] = set()
        pending: list[Module[Any, Any]] = [self]
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            discovered.append(current)
            pending.extend(
                dependency
                for dependency in current._dependencies.values()
                if isinstance(dependency, Module)
            )

        # Validate the complete graph before changing any node, avoiding a
        # partially frozen definition when a later child is invalid.
        snapshots: list[tuple[Module[Any, Any], FrozenJsonObject]] = []
        for current in discovered:
            snapshot = freeze_json_object(module_definition_config(current))
            previous = current._definition_config_snapshot
            if previous is not None and previous != snapshot:
                raise RuntimeError(
                    f"{type(current).__name__} definition changed after freeze; "
                    "create a new Module definition"
                )
            snapshots.append((current, snapshot))
        for current, snapshot in snapshots:
            object.__setattr__(current, "_definition_config_snapshot", snapshot)
            object.__setattr__(current, "_definition_frozen", True)

    def named_children(self) -> tuple[tuple[str, Module[Any, Any]], ...]:
        return tuple(self._children.items())

    def named_dependencies(
        self,
    ) -> tuple[tuple[str, ModuleDependency[Any, Any]], ...]:
        """Return declared raw, pre-bound, and remote Child dependencies."""

        return tuple(self._dependencies.items())

    async def __call__(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError(
                "Module calls require an active execution scope; "
                "use module.invoke() or module.stream() for a Root call"
            )
        output, next_context = await scope.invoke_module(self, message, context)
        return cast(OutputMessageT, output), next_context

    async def invoke(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> tuple[OutputMessageT, Context]:
        """Return the result projection of one direct execution."""

        handle = await self.start(message, context, execution=execution)
        return await handle.result()

    def _start_direct(
        self,
        message: InputMessageT,
        context: Context,
        execution: ExecutionOptions | None,
        *,
        start_now: bool = True,
    ) -> _DirectExecutionHandle[OutputMessageT]:
        """Create the direct owner synchronously for stream() and start()."""

        if _execution_scope.get() is not None:
            raise RuntimeError(
                "start() cannot create a Root inside an execution scope; "
                "call the Module directly to create a Child"
            )
        context_type = getattr(self, "context_type", None)
        if context_type is not None:
            if not isinstance(context_type, type) or not issubclass(
                context_type, Context
            ):
                raise TypeError("Agent.context_type must be a Context subclass")
            if type(context) is not context_type:
                raise TypeError(
                    f"{type(self).__name__} requires Context type "
                    f"{context_type.__name__}"
                )
        self._freeze_definition()
        options = execution or ExecutionOptions()
        if options.model_calls:
            declared_model_groups: set[str] = set()
            pending: list[Module[Any, Any]] = [self]
            visited: set[int] = set()
            while pending:
                current = pending.pop()
                if id(current) in visited:
                    continue
                visited.add(id(current))
                model_group = getattr(current, "model_group", None)
                group_name = getattr(model_group, "name", None)
                if isinstance(group_name, str) and group_name:
                    declared_model_groups.add(group_name)
                pending.extend(
                    dependency
                    for dependency in current._dependencies.values()
                    if isinstance(dependency, Module)
                )
            unknown = set(options.model_calls) - declared_model_groups
            if unknown:
                raise DirectExecutionError(
                    "model_calls references undeclared groups: "
                    + ", ".join(sorted(unknown))
                )
        execution_id = options.execution_id or f"direct-{uuid4()}"
        trace_id = options.trace_id or str(uuid4())
        record = _DirectExecutionRecord(
            execution_id=execution_id,
            trace_id=trace_id,
            root_span_id=str(uuid4()),
            module=self,
            message=message,
            context=context,
            options=options,
            module_paths=_direct_module_paths(self),
        )
        if start_now:
            record.ensure_started()
        return _DirectExecutionHandle(record)

    async def start(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> _DirectExecutionHandle[OutputMessageT]:
        """Start one unbound Module execution and return its control plane."""

        return self._start_direct(message, context, execution)

    def stream(
        self,
        message: InputMessageT,
        context: Context,
        *,
        execution: ExecutionOptions | None = None,
    ) -> DirectExecutionStream[OutputMessageT]:
        """Return an owned event projection over one direct execution."""

        return DirectExecutionStream(
            self._start_direct(message, context, execution, start_now=False)
        )

    async def forward(
        self, message: InputMessageT, context: Context
    ) -> tuple[OutputMessageT, Context]:
        raise NotImplementedError

    async def emit(self, *, kind: str, data: JsonObjectInput) -> None:
        """Publish custom content through the current execution channel."""

        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError("Module events require a bound Runtime execution")
        prepare = getattr(scope, "_prepare_module_event_data", None)
        payload = prepare(data) if callable(prepare) else freeze_json_object(data)
        await scope.emit_event(self, kind, payload)

    async def wait_external(
        self,
        *,
        kind: str,
        key: str,
        request: JsonObjectInput = (),
        timeout: float | None = None,
    ) -> FrozenJsonObject:
        """Wait for one bounded external signal in a managed execution."""

        scope = _execution_scope.get()
        if scope is None:
            raise RuntimeError("wait_external() requires an active execution scope")
        result = await scope.wait_external(
            kind=kind,
            key=key,
            request=freeze_json_object(request),
            timeout=timeout,
        )
        return freeze_json_object(result)

    async def gather(self, *awaitables: Awaitable[Any]) -> tuple[Any, ...]:
        """Await a structured parallel group in stable input order."""

        scope = _execution_scope.get()
        if scope is None:
            for awaitable in awaitables:
                close = getattr(awaitable, "close", None)
                if callable(close):
                    close()
            raise RuntimeError("gather() requires an active execution scope")
        operations = tuple(
            (lambda awaitable=awaitable: awaitable) for awaitable in awaitables
        )
        return await scope.gather(operations)

    def bind(
        self, runtime: Runtime, *, binding: Binding | None = None
    ) -> BoundModule[InputMessageT, OutputMessageT]:
        self._freeze_definition()
        return runtime.bind(self, binding=binding)


class Agent(
    Module[InputMessageT, OutputMessageT],
    Generic[InputMessageT, OutputMessageT],
):
    """Optional semantic name with exactly the same contract as Module."""

    context_type: type[Context] | None = None
