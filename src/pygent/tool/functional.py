"""Python-first local Tool declarations and deployment assembly helpers."""

from __future__ import annotations

import inspect
import operator
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MethodType, UnionType
from typing import (
    Annotated,
    Any,
    NotRequired,
    ParamSpec,
    Required,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from griffe import Docstring, DocstringSectionParameters, Parser
from pydantic import BaseModel, ConfigDict, TypeAdapter, create_model
from typing_extensions import TypedDict

from pygent.core import Context, FrozenJsonObject, JsonValue, Module, thaw_json

from .executors import ExecutorRegistry, LocalToolExecutor
from .layer import ToolCallLayer, TrustedAuthorizationAdapter
from .types import (
    IdempotencyPolicy,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)

P = ParamSpec("P")
R = TypeVar("R")
_DECLARATION_ATTRIBUTE = "__pygent_tool_declaration__"


@dataclass(frozen=True, slots=True)
class _ToolDeclaration:
    tool_id: str
    version: str
    side_effect: ToolSideEffect
    idempotency: IdempotencyPolicy
    name: str | None
    description: str | None
    timeout: float | None
    resource_key: str | None
    sandbox_profile: str | None
    required_permissions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompiledTool:
    handler: Callable[..., object]
    input_model: type[BaseModel]
    return_adapter: TypeAdapter[Any] | None
    spec: ToolSpec

    async def invoke(self, arguments: Mapping[str, JsonValue]) -> object:
        plain_arguments = (
            thaw_json(arguments)
            if isinstance(arguments, FrozenJsonObject)
            else dict(arguments)
        )
        validated = self.input_model.model_validate(plain_arguments, strict=True)
        kwargs = {
            name: getattr(validated, name) for name in self.input_model.model_fields
        }
        value = self.handler(**kwargs)
        if inspect.isawaitable(value):
            value = await cast(Awaitable[object], value)
        if self.return_adapter is None:
            return value
        # ToolCallLayer owns the authoritative output-schema validation so that
        # invalid outputs retain its established ToolResult classification.
        return self.return_adapter.dump_python(value, mode="json", warnings="none")


def tool(
    *,
    tool_id: str,
    version: str,
    side_effect: ToolSideEffect,
    name: str | None = None,
    description: str | None = None,
    idempotency: IdempotencyPolicy | None = None,
    timeout: float | None = None,
    resource_key: str | None = None,
    sandbox_profile: str | None = None,
    required_permissions: tuple[str, ...] = (),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Declare a Python function for explicit assembly by :class:`ToolKit`."""

    try:
        effect = ToolSideEffect(side_effect)
    except (TypeError, ValueError) as exc:
        raise TypeError("side_effect must be a ToolSideEffect") from exc
    if idempotency is None:
        if effect in (ToolSideEffect.PURE, ToolSideEffect.READ):
            policy = IdempotencyPolicy.INHERENT
        else:
            raise ValueError(
                "WRITE and EXTERNAL tools require an explicit idempotency policy"
            )
    else:
        try:
            policy = IdempotencyPolicy(idempotency)
        except (TypeError, ValueError) as exc:
            raise TypeError("idempotency must be an IdempotencyPolicy") from exc
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("tool_id must be non-empty")
    if not isinstance(version, str) or not version:
        raise ValueError("version must be non-empty")
    if name is not None and (not isinstance(name, str) or not name):
        raise ValueError("name must be non-empty when provided")
    if description is not None and (
        not isinstance(description, str) or not description.strip()
    ):
        raise ValueError("description must be non-empty when provided")

    declaration = _ToolDeclaration(
        tool_id=tool_id,
        version=version,
        side_effect=effect,
        idempotency=policy,
        name=name,
        description=description.strip() if description is not None else None,
        timeout=timeout,
        resource_key=resource_key,
        sandbox_profile=sandbox_profile,
        required_permissions=tuple(required_permissions),
    )

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        if not inspect.isfunction(function):
            raise TypeError("@tool can decorate only Python functions or methods")
        if inspect.isgeneratorfunction(function) or inspect.isasyncgenfunction(
            function
        ):
            raise TypeError("@tool does not support generator functions")
        if hasattr(function, _DECLARATION_ATTRIBUTE):
            raise ValueError("function is already decorated with @tool")
        setattr(function, _DECLARATION_ATTRIBUTE, declaration)
        return function

    return decorate


class ToolKit:
    """Deployment-local collection of decorated Python Tool callables.

    A ToolKit is an assembly helper, not a portable Tool value. Only its
    ``definitions`` and ``specs`` projections belong in Context or Module state.
    """

    __slots__ = ("_compiled", "_definitions", "_specs")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolKit cannot be subclassed")

    def __init__(self, *handlers: Callable[..., object]) -> None:
        compiled = tuple(_compile_tool(handler) for handler in handlers)
        names = tuple(item.spec.definition.name for item in compiled)
        identities = tuple((item.spec.tool_id, item.spec.version) for item in compiled)
        if len(names) != len(set(names)):
            raise ValueError("ToolKit contains duplicate model-visible names")
        if len(identities) != len(set(identities)):
            raise ValueError("ToolKit contains duplicate (tool_id, version) identities")
        self._compiled = compiled
        self._definitions = tuple(item.spec.definition for item in compiled)
        self._specs = tuple(item.spec for item in compiled)

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return the portable model-visible Tool definitions."""

        return self._definitions

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return the portable execution declarations."""

        return self._specs

    def register_into(
        self,
        registry: ExecutorRegistry,
        *,
        replace_existing: bool = False,
    ) -> ExecutorRegistry:
        """Install local executors into an explicit deployment registry."""

        if not isinstance(registry, ExecutorRegistry):
            raise TypeError("registry must be an ExecutorRegistry")
        for item in self._compiled:
            registry.register(
                item.spec.tool_id,
                item.spec.version,
                LocalToolExecutor(item.invoke),
                replace_existing=replace_existing,
            )
        return registry

    def build_registry(self) -> ExecutorRegistry:
        """Build a new local executor registry for this ToolKit."""

        return self.register_into(ExecutorRegistry())

    def local_layer(
        self,
        *,
        authorization: Module[
            ToolAuthorizationRequest, ToolAuthorizationDecision
        ]
        | None = None,
        authorization_adapter: TrustedAuthorizationAdapter | None = None,
        max_concurrency: int | None = None,
    ) -> ToolCallLayer:
        """Build a direct/local ToolCallLayer without changing authorization."""

        return ToolCallLayer(
            tools=self.specs,
            authorization=authorization,
            authorization_adapter=authorization_adapter,
            executor_registry=self.build_registry(),
            max_concurrency=max_concurrency,
        )

    def make_visible_in(self, context: Context) -> Context:
        """Return a Context where this ToolKit's definitions are model-visible."""

        if type(context) is not Context:
            raise TypeError("context must be a Context")
        definitions_by_name = {item.name: item for item in context.tools}
        additions: list[ToolDefinition] = []
        for definition in self.definitions:
            existing = definitions_by_name.get(definition.name)
            if existing is None:
                definitions_by_name[definition.name] = definition
                additions.append(definition)
            elif existing != definition:
                raise ValueError(
                    f"Context already contains a different ToolDefinition "
                    f"named {definition.name!r}"
                )
        if not additions:
            return context
        return replace(context, tools=context.tools + tuple(additions))


def _compile_tool(handler: Callable[..., object]) -> _CompiledTool:
    if not callable(handler):
        raise TypeError("ToolKit values must be callable")
    declaration_owner = handler.__func__ if isinstance(handler, MethodType) else handler
    declaration = getattr(declaration_owner, _DECLARATION_ATTRIBUTE, None)
    if not isinstance(declaration, _ToolDeclaration):
        raise TypeError("ToolKit values must be decorated with @tool")
    if inspect.isgeneratorfunction(handler) or inspect.isasyncgenfunction(handler):
        raise TypeError("ToolKit does not support generator functions")

    try:
        signature = inspect.signature(handler)
        hints = get_type_hints(handler, include_extras=True)
    except (TypeError, NameError) as exc:
        raise TypeError(f"cannot inspect Tool signature for {handler!r}") from exc

    fields: dict[str, tuple[object, object]] = {}
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError("Tool parameters cannot be positional-only")
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError("Tool signatures cannot use *args or **kwargs")
        annotation = hints.get(parameter.name, parameter.annotation)
        if annotation is inspect.Signature.empty:
            raise TypeError(
                f"Tool parameter {parameter.name!r} must have a type annotation"
            )
        default = parameter.default
        fields[parameter.name] = (
            _pydantic_annotation(annotation),
            ... if default is inspect.Signature.empty else default,
        )

    visible_name = declaration.name or getattr(handler, "__name__", None)
    if not isinstance(visible_name, str) or not visible_name:
        raise ValueError("Tool callable must have a non-empty name")
    docstring = inspect.getdoc(handler) or ""
    description = declaration.description or _docstring_summary(docstring)
    if not description:
        raise ValueError(
            f"Tool {visible_name!r} requires a description or a non-empty docstring"
        )

    model_name = "".join(part.capitalize() for part in visible_name.replace("-", "_").split("_"))
    input_model = create_model(
        f"{model_name or 'Tool'}Arguments",
        __config__=ConfigDict(extra="forbid"),
        **cast(Any, fields),
    )
    parameters = input_model.model_json_schema(mode="validation")
    parameter_descriptions = _docstring_parameter_descriptions(docstring)
    properties = parameters.get("properties")
    if isinstance(properties, dict):
        for parameter_name, parameter_description in parameter_descriptions.items():
            property_schema = properties.get(parameter_name)
            if isinstance(property_schema, dict) and "description" not in property_schema:
                property_schema["description"] = parameter_description

    return_annotation = hints.get("return", signature.return_annotation)
    return_adapter: TypeAdapter[Any] | None = None
    output_schema: dict[str, Any] | None = None
    if return_annotation is not inspect.Signature.empty:
        return_adapter = TypeAdapter(_pydantic_annotation(return_annotation))
        output_schema = return_adapter.json_schema(mode="serialization")

    definition = ToolDefinition(
        name=visible_name,
        description=description,
        parameters=parameters,
        output_schema=output_schema,
    )
    spec = ToolSpec(
        tool_id=declaration.tool_id,
        version=declaration.version,
        definition=definition,
        side_effect=declaration.side_effect,
        idempotency=declaration.idempotency,
        timeout=declaration.timeout,
        resource_key=declaration.resource_key,
        sandbox_profile=declaration.sandbox_profile,
        required_permissions=declaration.required_permissions,
    )
    return _CompiledTool(handler, input_model, return_adapter, spec)


def _docstring_summary(docstring: str) -> str:
    if not docstring:
        return ""
    paragraphs = docstring.strip().split("\n\n", 1)
    return " ".join(line.strip() for line in paragraphs[0].splitlines()).strip()


def _docstring_parameter_descriptions(docstring: str) -> dict[str, str]:
    if not docstring:
        return {}
    descriptions: dict[str, str] = {}
    for parser in (Parser.auto, Parser.google, Parser.numpy, Parser.sphinx):
        try:
            sections = Docstring(docstring).parse(parser=parser, warnings=False)
        except (AttributeError, TypeError, ValueError):
            continue
        for section in sections:
            if not isinstance(section, DocstringSectionParameters):
                continue
            for parameter in section.value:
                description = " ".join(str(parameter.description).split())
                if description:
                    descriptions.setdefault(parameter.name, description)
    return descriptions


def _pydantic_annotation(
    annotation: object,
    typed_dicts: dict[type[object], type[object]] | None = None,
) -> object:
    """Adapt Python 3.11 stdlib TypedDicts to Pydantic's supported backport."""

    memo = {} if typed_dicts is None else typed_dicts
    if (
        sys.version_info < (3, 12)
        and isinstance(annotation, type)
        and is_typeddict(annotation)
        and type(annotation).__module__ == "typing"
    ):
        existing = memo.get(annotation)
        if existing is not None:
            return existing
        hints = get_type_hints(annotation, include_extras=True)
        required = getattr(annotation, "__required_keys__", frozenset(hints))
        fields: dict[str, object] = {}
        for name, value in hints.items():
            if get_origin(value) in (Required, NotRequired):
                value = get_args(value)[0]
            adapted_value = _pydantic_annotation(value, memo)
            fields[name] = (
                Required[adapted_value]
                if name in required
                else NotRequired[adapted_value]
            )
        adapted = TypedDict(  # type: ignore[misc]
            annotation.__name__, fields, total=False
        )
        memo[annotation] = adapted
        return adapted

    origin = get_origin(annotation)
    if origin is None:
        return annotation
    arguments = get_args(annotation)
    adapted_arguments = tuple(_pydantic_annotation(item, memo) for item in arguments)
    if adapted_arguments == arguments:
        return annotation
    if origin is Annotated:
        return Annotated[adapted_arguments[0], *adapted_arguments[1:]]
    if origin is UnionType:
        return reduce(operator.or_, adapted_arguments)
    try:
        return origin[adapted_arguments]
    except TypeError:
        return annotation


__all__ = ["ToolKit", "tool"]
