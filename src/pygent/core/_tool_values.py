"""Portable tool values shared by Core messages and the Tool domain."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from .json_values import (
    FrozenJsonObject,
    JsonObjectInput,
    JsonValue,
    freeze_json,
    freeze_json_object,
    thaw_json,
)


class ToolSideEffect(str, Enum):
    PURE = "pure"
    READ = "read"
    WRITE = "write"
    EXTERNAL = "external"


class IdempotencyPolicy(str, Enum):
    INHERENT = "inherent"
    REQUIRES_KEY = "requires_key"
    NOT_IDEMPOTENT = "not_idempotent"


class ToolTaskState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


ToolResultStatus = Literal[
    "succeeded", "rejected", "failed", "cancelled", "unknown", "detached"
]
MediaSourceKind = Literal["resource", "url", "inline"]
ToolResultMediaType = Literal["image", "video"]
ToolResultMediaDetail = Literal["auto", "low", "high"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIME_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")


def _non_empty(value: object, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value:
        raise ValueError(f"{name} must be non-empty")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """The model-visible name and JSON Schema of a tool."""

    name: str
    description: str
    parameters: JsonObjectInput
    output_schema: JsonObjectInput | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolDefinition cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.name, "tool definition name")
        if not isinstance(self.description, str):
            raise TypeError("tool definition description must be a string")
        object.__setattr__(self, "parameters", freeze_json_object(self.parameters))
        Draft202012Validator.check_schema(
            cast(dict[str, Any], thaw_json(cast(FrozenJsonObject, self.parameters)))
        )
        if self.output_schema is not None:
            object.__setattr__(
                self, "output_schema", freeze_json_object(self.output_schema)
            )
            Draft202012Validator.check_schema(
                cast(
                    dict[str, Any],
                    thaw_json(cast(FrozenJsonObject, self.output_schema)),
                )
            )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-requested invocation; this is not an admitted task."""

    call_id: str
    name: str
    arguments: JsonObjectInput
    tool_id: str | None = None
    tool_version: str | None = None
    idempotency_key: str | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolCall cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.call_id, "tool call_id")
        _non_empty(self.name, "tool name")
        for field_name in ("tool_id", "tool_version", "idempotency_key"):
            value = getattr(self, field_name)
            if value is not None:
                _non_empty(value, field_name)
        object.__setattr__(self, "arguments", freeze_json_object(self.arguments))


@dataclass(frozen=True, slots=True)
class ToolTask:
    """Immutable public snapshot of one admitted tool execution."""

    task_id: str
    call_id: str
    tool_id: str
    version: str
    state: ToolTaskState
    job_id: str | None = None
    metadata: JsonObjectInput = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolTask cannot be subclassed")

    def __post_init__(self) -> None:
        for name in ("task_id", "call_id", "tool_id", "version"):
            _non_empty(getattr(self, name), name)
        if not isinstance(self.state, ToolTaskState):
            object.__setattr__(self, "state", ToolTaskState(self.state))
        if self.job_id is not None:
            _non_empty(self.job_id, "job_id")
        object.__setattr__(self, "metadata", freeze_json_object(self.metadata))


@dataclass(frozen=True, slots=True)
class MediaSource:
    """Portable media source descriptor without an open resource."""

    kind: MediaSourceKind
    uri: str | None = None
    url: str | None = None
    base64_data: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MediaSource cannot be subclassed")

    def __post_init__(self) -> None:
        if self.kind not in ("resource", "url", "inline"):
            raise ValueError(f"unsupported media source kind: {self.kind!r}")
        selected = {
            "resource": self.uri,
            "url": self.url,
            "inline": self.base64_data,
        }
        expected = selected[self.kind]
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"{self.kind} media source requires a non-empty value")
        if any(value is not None for key, value in selected.items() if key != self.kind):
            raise ValueError("media source must contain exactly one source value")
        if self.kind == "url":
            parsed = urlsplit(cast(str, self.url))
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("media URL must be an absolute HTTP(S) URL")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("media URL must not contain credentials")
        if self.sha256 is not None and (
            not isinstance(self.sha256, str)
            or _SHA256.fullmatch(self.sha256) is None
        ):
            raise ValueError("media sha256 must be 64 lowercase hexadecimal characters")
        if self.size_bytes is not None and (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes <= 0
        ):
            raise ValueError("media size_bytes must be a positive integer")
        if self.kind == "inline":
            try:
                decoded = base64.b64decode(cast(str, self.base64_data), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("inline media must contain valid Base64") from exc
            if not decoded:
                raise ValueError("inline media must be non-empty")
            digest = hashlib.sha256(decoded).hexdigest()
            if self.sha256 is not None and self.sha256 != digest:
                raise ValueError("inline media sha256 does not match its content")
            if self.size_bytes is not None and self.size_bytes != len(decoded):
                raise ValueError("inline media size_bytes does not match its content")
            object.__setattr__(self, "sha256", digest)
            object.__setattr__(self, "size_bytes", len(decoded))

    @classmethod
    def resource(
        cls,
        uri: str,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> MediaSource:
        return cls(kind="resource", uri=uri, sha256=sha256, size_bytes=size_bytes)

    @classmethod
    def remote_url(
        cls,
        url: str,
        *,
        sha256: str | None = None,
        size_bytes: int | None = None,
    ) -> MediaSource:
        return cls(kind="url", url=url, sha256=sha256, size_bytes=size_bytes)

    @classmethod
    def inline(cls, data: bytes) -> MediaSource:
        if not isinstance(data, bytes):
            raise TypeError("inline media data must be bytes")
        if not data:
            raise ValueError("inline media data must be non-empty")
        return cls(kind="inline", base64_data=base64.b64encode(data).decode("ascii"))


@dataclass(frozen=True, slots=True)
class ToolResultText:
    text: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolResultText cannot be subclassed")

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("tool result text must be a string")


@dataclass(frozen=True, slots=True)
class ToolResultJson:
    value: JsonValue

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolResultJson cannot be subclassed")

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", freeze_json(self.value))


@dataclass(frozen=True, slots=True)
class ToolResultMedia:
    media_type: ToolResultMediaType
    mime_type: str
    source: MediaSource
    detail: ToolResultMediaDetail | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolResultMedia cannot be subclassed")

    def __post_init__(self) -> None:
        if self.media_type not in ("image", "video"):
            raise ValueError(f"unsupported tool result media type: {self.media_type!r}")
        if not isinstance(self.mime_type, str):
            raise TypeError("media mime_type must be a string")
        mime_type = self.mime_type.lower()
        if _MIME_TYPE.fullmatch(mime_type) is None:
            raise ValueError("media mime_type is invalid")
        if not mime_type.startswith(f"{self.media_type}/"):
            raise ValueError("media mime_type does not match media_type")
        if type(self.source) is not MediaSource:
            raise TypeError("media source must be a MediaSource")
        if self.detail not in (None, "auto", "low", "high"):
            raise ValueError(f"unsupported media detail: {self.detail!r}")
        object.__setattr__(self, "mime_type", mime_type)


ToolResultContent = ToolResultText | ToolResultJson | ToolResultMedia


def _tool_result_content_to_value(value: ToolResultContent) -> dict[str, object]:
    if type(value) is ToolResultText:
        return {"type": "text", "text": value.text}
    if type(value) is ToolResultJson:
        return {"type": "json", "value": thaw_json(value.value)}
    if type(value) is ToolResultMedia:
        source = value.source
        return {
            "type": "media",
            "media_type": value.media_type,
            "mime_type": value.mime_type,
            "detail": value.detail,
            "source": {
                "kind": source.kind,
                "uri": source.uri,
                "url": source.url,
                "base64_data": source.base64_data,
                "sha256": source.sha256,
                "size_bytes": source.size_bytes,
            },
        }
    raise TypeError("unsupported ToolResult content subtype")


def _tool_result_content_from_value(value: object) -> ToolResultContent:
    if not isinstance(value, Mapping):
        raise TypeError("ToolResult content block must be an object")
    block = dict(value)
    block_type = block.get("type")
    if block_type == "text" and set(block) == {"type", "text"}:
        return ToolResultText(text=cast(str, block["text"]))
    if block_type == "json" and set(block) == {"type", "value"}:
        return ToolResultJson(value=cast(JsonValue, block["value"]))
    if block_type == "media" and set(block) == {
        "type", "media_type", "mime_type", "detail", "source"
    }:
        raw_source = block["source"]
        if not isinstance(raw_source, Mapping) or set(raw_source) != {
            "kind", "uri", "url", "base64_data", "sha256", "size_bytes"
        }:
            raise TypeError("ToolResult media source has invalid fields")
        source = MediaSource(
            kind=cast(MediaSourceKind, raw_source["kind"]),
            uri=cast(str | None, raw_source["uri"]),
            url=cast(str | None, raw_source["url"]),
            base64_data=cast(str | None, raw_source["base64_data"]),
            sha256=cast(str | None, raw_source["sha256"]),
            size_bytes=cast(int | None, raw_source["size_bytes"]),
        )
        return ToolResultMedia(
            media_type=cast(ToolResultMediaType, block["media_type"]),
            mime_type=cast(str, block["mime_type"]),
            source=source,
            detail=cast(ToolResultMediaDetail | None, block["detail"]),
        )
    raise TypeError("unsupported ToolResult content block")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Terminal result, admission rejection, or detached-task acknowledgment."""

    call_id: str
    name: str
    status: ToolResultStatus
    task: ToolTask | None = None
    output: JsonValue = None
    content: tuple[ToolResultContent, ...] = ()
    error: str | None = None
    error_kind: str | None = None
    error_code: str | None = None
    retryable: bool = False
    side_effect_committed: bool | None = None
    tool_id: str | None = None
    tool_version: str | None = None
    missing_capabilities: tuple[str, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ToolResult cannot be subclassed")

    def __post_init__(self) -> None:
        _non_empty(self.call_id, "tool result call_id")
        _non_empty(self.name, "tool result name")
        if self.status not in (
            "succeeded",
            "rejected",
            "failed",
            "cancelled",
            "unknown",
            "detached",
        ):
            raise ValueError(f"unsupported tool result status: {self.status!r}")
        if self.task is not None and type(self.task) is not ToolTask:
            raise TypeError("task must be a ToolTask or None")
        if self.status == "detached" and self.task is None:
            raise ValueError("a detached result must include a ToolTask snapshot")
        if self.status == "rejected" and self.task is not None:
            raise ValueError("an authorization rejection cannot include a ToolTask")
        content = tuple(self.content)
        if any(
            type(value) not in (ToolResultText, ToolResultJson, ToolResultMedia)
            for value in content
        ):
            raise TypeError("ToolResult content contains an unsupported value")
        if content and self.status != "succeeded":
            raise ValueError("only succeeded ToolResult values can contain content blocks")
        for name in ("error", "error_kind", "error_code", "tool_id", "tool_version"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a bool")
        if self.side_effect_committed is not None and not isinstance(
            self.side_effect_committed, bool
        ):
            raise TypeError("side_effect_committed must be a bool or None")
        capabilities = tuple(self.missing_capabilities)
        if any(not isinstance(value, str) or not value for value in capabilities):
            raise ValueError("missing_capabilities must contain non-empty strings")
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("missing_capabilities must be unique")
        object.__setattr__(self, "missing_capabilities", capabilities)
        object.__setattr__(self, "output", freeze_json(self.output))
        object.__setattr__(self, "content", content)


__all__ = [
    "IdempotencyPolicy",
    "MediaSource",
    "MediaSourceKind",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "ToolResultContent",
    "ToolResultJson",
    "ToolResultMedia",
    "ToolResultMediaDetail",
    "ToolResultMediaType",
    "ToolResultStatus",
    "ToolResultText",
    "ToolSideEffect",
    "ToolTask",
    "ToolTaskState",
]
