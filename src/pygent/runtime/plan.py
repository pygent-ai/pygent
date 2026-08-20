"""Versioned, serializable contracts produced by Runtime binding."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

PLAN_SCHEMA_VERSION = 3

_ARTIFACT_FIELDS = {"package", "version", "digest", "entrypoint"}
_MODULE_FIELDS = {
    "path",
    "type_name",
    "children",
    "resource_keys",
    "definition_id",
    "config_ref",
    "input_schema",
    "output_schema",
    "serializer",
    "placement_constraints",
    "required_capabilities",
    "retry_policy_ref",
    "checkpoint_policy_ref",
    "metadata",
    "model_requirements",
}
_PLAN_FIELDS = {
    "schema_version",
    "runtime_api_version",
    "root",
    "artifact",
    "modules",
    "metadata",
    "graph_hash",
    "context_codecs",
}


def _context_codec_tuple(value: object) -> tuple[tuple[str, int, str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanValidationError("context_codecs must be a sequence")
    result: list[tuple[str, int, str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            raise PlanValidationError("context codec identity must have four values")
        schema, version, codec, digest = item
        if (
            not isinstance(schema, str)
            or not schema
            or not isinstance(version, int)
            or isinstance(version, bool)
            or version <= 0
            or not isinstance(codec, str)
            or not codec
            or not isinstance(digest, str)
            or not digest.startswith("sha256:")
        ):
            raise PlanValidationError("invalid context codec identity")
        result.append((schema, version, codec, digest))
    if len(result) != len(set(result)):
        raise PlanValidationError("context_codecs contains duplicates")
    return tuple(sorted(result))


class PlanValidationError(ValueError):
    """Raised when an execution plan is structurally invalid."""


class PlanVersionError(PlanValidationError):
    """Raised when a runtime cannot read a plan schema version."""


class PlanIntegrityError(PlanValidationError):
    """Raised when serialized plan content does not match its graph hash."""


@dataclass(frozen=True, slots=True)
class ModelRequirement:
    group_name: str
    capacity_key: str
    max_concurrency: int | None
    allow_profile_override: bool = False
    overridable_generation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.group_name, "model_requirement.group_name")
        _require_text(self.capacity_key, "model_requirement.capacity_key")
        if self.max_concurrency is not None and (
            not isinstance(self.max_concurrency, int)
            or isinstance(self.max_concurrency, bool)
            or self.max_concurrency <= 0
        ):
            raise PlanValidationError("model requirement max_concurrency must be positive")
        if not isinstance(self.allow_profile_override, bool):
            raise PlanValidationError("allow_profile_override must be a bool")
        values = _text_tuple(
            self.overridable_generation,
            "model_requirement.overridable_generation",
        )
        object.__setattr__(self, "overridable_generation", tuple(sorted(values)))

    def to_dict(self) -> dict[str, object]:
        return {
            "group_name": self.group_name,
            "capacity_key": self.capacity_key,
            "max_concurrency": self.max_concurrency,
            "allow_profile_override": self.allow_profile_override,
            "overridable_generation": list(self.overridable_generation),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModelRequirement:
        allowed = {
            "group_name",
            "capacity_key",
            "max_concurrency",
            "allow_profile_override",
            "overridable_generation",
        }
        _reject_unknown_fields(value, allowed, "model requirement")
        return cls(
            group_name=_require_text(value.get("group_name"), "model_requirement.group_name"),
            capacity_key=_require_text(value.get("capacity_key"), "model_requirement.capacity_key"),
            max_concurrency=value.get("max_concurrency"),  # type: ignore[arg-type]
            allow_profile_override=value.get("allow_profile_override", False),  # type: ignore[arg-type]
            overridable_generation=_text_tuple(
                value.get("overridable_generation", ()),
                "model_requirement.overridable_generation",
            ),
        )


def _reject_unknown_fields(
    value: Mapping[str, object], allowed: set[str], object_name: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PlanValidationError(
            f"{object_name} contains unknown fields: {unknown!r}"
        )


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanValidationError(f"{field_name} must be a sequence of strings")
    result = tuple(_require_text(item, field_name) for item in value)
    if len(result) != len(set(result)):
        raise PlanValidationError(f"{field_name} contains duplicate values")
    return result


def _metadata_tuple(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise PlanValidationError("metadata must be a sequence of string pairs")
    result: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise PlanValidationError("metadata must contain string pairs")
        result.append(
            (
                _require_text(item[0], "metadata key"),
                _require_text(item[1], "metadata value"),
            )
        )
    if len(result) != len({key for key, _ in result}):
        raise PlanValidationError("metadata contains duplicate keys")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CodeArtifactSpec:
    """Immutable code artifact required to reconstruct a plan on a worker."""

    package: str
    version: str
    digest: str
    entrypoint: str

    def __post_init__(self) -> None:
        for name in ("package", "version", "digest", "entrypoint"):
            _require_text(getattr(self, name), f"artifact.{name}")

    def to_dict(self) -> dict[str, str]:
        return {
            "package": self.package,
            "version": self.version,
            "digest": self.digest,
            "entrypoint": self.entrypoint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CodeArtifactSpec:
        _reject_unknown_fields(value, _ARTIFACT_FIELDS, "artifact")
        return cls(
            package=_require_text(value.get("package"), "artifact.package"),
            version=_require_text(value.get("version"), "artifact.version"),
            digest=_require_text(value.get("digest"), "artifact.digest"),
            entrypoint=_require_text(
                value.get("entrypoint"), "artifact.entrypoint"
            ),
        )


@dataclass(frozen=True, slots=True)
class ModuleSpec:
    """Portable description of one definition node in a bound Module graph."""

    path: str
    type_name: str
    children: tuple[str, ...] = ()
    resource_keys: tuple[str, ...] = ()
    definition_id: str | None = None
    config_ref: str | None = None
    input_schema: str | None = None
    output_schema: str | None = None
    serializer: str | None = None
    placement_constraints: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    retry_policy_ref: str | None = None
    checkpoint_policy_ref: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    model_requirements: tuple[ModelRequirement, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.path, "module.path")
        _require_text(self.type_name, "module.type_name")
        for name in (
            "definition_id",
            "config_ref",
            "input_schema",
            "output_schema",
            "serializer",
            "retry_policy_ref",
            "checkpoint_policy_ref",
        ):
            _optional_text(getattr(self, name), f"module.{name}")
        for name in (
            "children",
            "resource_keys",
            "placement_constraints",
            "required_capabilities",
        ):
            values = _text_tuple(getattr(self, name), f"module.{name}")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        requirements = tuple(self.model_requirements)
        if any(not isinstance(item, ModelRequirement) for item in requirements):
            raise PlanValidationError(
                "module.model_requirements must contain ModelRequirement values"
            )
        names = tuple(item.group_name for item in requirements)
        if len(names) != len(set(names)):
            raise PlanValidationError("module.model_requirements contains duplicates")
        object.__setattr__(self, "model_requirements", requirements)

    @property
    def is_portable(self) -> bool:
        """Whether a remote worker can validate this node's wire contract."""

        return all(
            value is not None
            for value in (
                self.definition_id,
                self.input_schema,
                self.output_schema,
                self.serializer,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "type_name": self.type_name,
            "children": list(self.children),
            "resource_keys": list(self.resource_keys),
            "definition_id": self.definition_id,
            "config_ref": self.config_ref,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "serializer": self.serializer,
            "placement_constraints": list(self.placement_constraints),
            "required_capabilities": list(self.required_capabilities),
            "retry_policy_ref": self.retry_policy_ref,
            "checkpoint_policy_ref": self.checkpoint_policy_ref,
            "metadata": [list(item) for item in self.metadata],
            "model_requirements": [
                item.to_dict() for item in self.model_requirements
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ModuleSpec:
        _reject_unknown_fields(value, _MODULE_FIELDS, "module")
        raw_model_requirements = value.get("model_requirements", ())
        if not isinstance(raw_model_requirements, (list, tuple)) or any(
            not isinstance(item, Mapping) for item in raw_model_requirements
        ):
            raise PlanValidationError(
                "module.model_requirements must be a sequence of objects"
            )
        return cls(
            path=_require_text(value.get("path"), "module.path"),
            type_name=_require_text(value.get("type_name"), "module.type_name"),
            children=_text_tuple(value.get("children", ()), "module.children"),
            resource_keys=_text_tuple(
                value.get("resource_keys", ()), "module.resource_keys"
            ),
            definition_id=_optional_text(
                value.get("definition_id"), "module.definition_id"
            ),
            config_ref=_optional_text(value.get("config_ref"), "module.config_ref"),
            input_schema=_optional_text(
                value.get("input_schema"), "module.input_schema"
            ),
            output_schema=_optional_text(
                value.get("output_schema"), "module.output_schema"
            ),
            serializer=_optional_text(value.get("serializer"), "module.serializer"),
            placement_constraints=_text_tuple(
                value.get("placement_constraints", ()),
                "module.placement_constraints",
            ),
            required_capabilities=_text_tuple(
                value.get("required_capabilities", ()),
                "module.required_capabilities",
            ),
            retry_policy_ref=_optional_text(
                value.get("retry_policy_ref"), "module.retry_policy_ref"
            ),
            checkpoint_policy_ref=_optional_text(
                value.get("checkpoint_policy_ref"),
                "module.checkpoint_policy_ref",
            ),
            metadata=_metadata_tuple(value.get("metadata", ())),
            model_requirements=tuple(
                ModelRequirement.from_dict(item)
                for item in raw_model_requirements
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Validated graph snapshot exchanged between a binding and its runtime."""

    root: str
    modules: tuple[ModuleSpec, ...]
    schema_version: int = PLAN_SCHEMA_VERSION
    runtime_api_version: str = "0.2"
    artifact: CodeArtifactSpec | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    context_codecs: tuple[tuple[str, int, str, str], ...] = ()
    graph_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _require_text(self.root, "root")
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != PLAN_SCHEMA_VERSION
        ):
            raise PlanVersionError(
                f"unsupported plan schema version {self.schema_version}; "
                f"expected {PLAN_SCHEMA_VERSION}"
            )
        _require_text(self.runtime_api_version, "runtime_api_version")
        if self.runtime_api_version != "0.2":
            raise PlanVersionError(
                f"unsupported Runtime API version {self.runtime_api_version!r}; "
                "expected '0.2'"
            )
        if not isinstance(self.modules, (list, tuple)) or not all(
            isinstance(module, ModuleSpec) for module in self.modules
        ):
            raise PlanValidationError("modules must contain ModuleSpec values")
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "metadata", _metadata_tuple(self.metadata))
        object.__setattr__(self, "context_codecs", _context_codec_tuple(self.context_codecs))
        self._validate_graph()
        object.__setattr__(self, "graph_hash", self._calculate_graph_hash())

    @property
    def plan_id(self) -> str:
        return f"sha256:{self.graph_hash}"

    @property
    def is_portable(self) -> bool:
        return self.artifact is not None and all(
            module.is_portable for module in self.modules
        )

    def _validate_graph(self) -> None:
        if not self.modules:
            raise PlanValidationError("modules must not be empty")
        by_path = {module.path: module for module in self.modules}
        if len(by_path) != len(self.modules):
            raise PlanValidationError("modules contains a duplicate path")
        if self.root not in by_path:
            raise PlanValidationError(f"root {self.root!r} is not a module path")
        for module in self.modules:
            for child in module.children:
                if child not in by_path:
                    raise PlanValidationError(
                        f"module {module.path!r} references unknown child {child!r}"
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(path: str) -> None:
            if path in visiting:
                raise PlanValidationError(f"module graph contains a cycle at {path!r}")
            if path in visited:
                return
            visiting.add(path)
            for child in by_path[path].children:
                visit(child)
            visiting.remove(path)
            visited.add(path)

        visit(self.root)
        unreachable = sorted(set(by_path) - visited)
        if unreachable:
            raise PlanValidationError(
                f"module graph contains unreachable modules: {unreachable!r}"
            )

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_api_version": self.runtime_api_version,
            "root": self.root,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "modules": [
                module.to_dict()
                for module in sorted(self.modules, key=lambda item: item.path)
            ],
            "context_codecs": [list(item) for item in self.context_codecs],
        }

    def _calculate_graph_hash(self) -> str:
        encoded = json.dumps(
            self._hash_payload(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "metadata": [list(item) for item in self.metadata],
            "graph_hash": self.graph_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExecutionPlan:
        _reject_unknown_fields(value, _PLAN_FIELDS, "execution plan")
        schema_version = value.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise PlanVersionError("schema_version must be an integer")
        if schema_version != PLAN_SCHEMA_VERSION:
            raise PlanVersionError(
                f"unsupported plan schema version {schema_version}; "
                f"expected {PLAN_SCHEMA_VERSION}"
            )

        raw_modules = value.get("modules")
        if not isinstance(raw_modules, (list, tuple)):
            raise PlanValidationError("modules must be a sequence")
        modules: list[ModuleSpec] = []
        for raw_module in raw_modules:
            if not isinstance(raw_module, Mapping):
                raise PlanValidationError("modules must contain objects")
            modules.append(ModuleSpec.from_dict(raw_module))

        raw_artifact = value.get("artifact")
        if raw_artifact is not None and not isinstance(raw_artifact, Mapping):
            raise PlanValidationError("artifact must be an object or null")
        artifact = (
            CodeArtifactSpec.from_dict(raw_artifact)
            if isinstance(raw_artifact, Mapping)
            else None
        )
        plan = cls(
            root=_require_text(value.get("root"), "root"),
            modules=tuple(modules),
            schema_version=schema_version,
            runtime_api_version=_require_text(
                value.get("runtime_api_version"), "runtime_api_version"
            ),
            artifact=artifact,
            context_codecs=_context_codec_tuple(value.get("context_codecs", ())),
            metadata=_metadata_tuple(value.get("metadata", ())),
        )
        expected_hash = _require_text(value.get("graph_hash"), "graph_hash")
        if expected_hash != plan.graph_hash:
            raise PlanIntegrityError(
                "serialized execution plan does not match its graph_hash"
            )
        return plan


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "CodeArtifactSpec",
    "ExecutionPlan",
    "ModelRequirement",
    "ModuleSpec",
    "PlanIntegrityError",
    "PlanValidationError",
    "PlanVersionError",
]
