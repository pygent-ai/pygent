from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias

from tests.live.az_conformance.schemas import Scenario


class ErrorKind(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    CAPABILITY_MISMATCH = "capability_mismatch"
    INVALID_RESPONSE = "invalid_response"
    TIMEOUT = "timeout"
    GATEWAY_UNAVAILABLE = "gateway_unavailable"
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"
    CONTENT_REJECTED = "content_rejected"
    DEPENDENCY_FAILED = "dependency_failed"
    UNKNOWN = "unknown"


ResultKey: TypeAlias = tuple[str, str, str, str, Scenario]
ResultStatus: TypeAlias = Literal["passed", "failed"]

_REPLACE_ATTEMPTS = 20
_REPLACE_RETRY_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class ProbeResult:
    snapshot_sha256: str
    source_revision: str
    route_id: str
    canonical_provider: str | None
    canonical_model_id: str | None
    protocol: str
    scenario: Scenario
    status: ResultStatus
    attempts: int
    error_kind: ErrorKind | None
    private_detail: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_sha256(self.snapshot_sha256)
        _non_empty(self.source_revision, "source_revision")
        _non_empty(self.route_id, "route_id")
        _non_empty(self.protocol, "protocol")
        if not isinstance(self.scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        if self.status not in {"passed", "failed"}:
            raise ValueError("status must be passed or failed")
        if (
            not isinstance(self.attempts, int)
            or isinstance(self.attempts, bool)
            or self.attempts < 1
        ):
            raise ValueError("attempts must be an integer greater than or equal to 1")
        if self.status == "passed" and self.error_kind is not None:
            raise ValueError("passed result error_kind must be null")
        if self.status == "failed" and not isinstance(self.error_kind, ErrorKind):
            raise ValueError("failed result requires an error_kind")
        identity = (self.canonical_provider, self.canonical_model_id)
        if (identity[0] is None) != (identity[1] is None):
            raise ValueError("canonical identity must be complete or absent")
        if identity[0] is not None:
            _non_empty(identity[0], "canonical_provider")
            _non_empty(identity[1], "canonical_model_id")

    @property
    def key(self) -> ResultKey:
        return (
            self.snapshot_sha256,
            self.source_revision,
            self.route_id,
            self.protocol,
            self.scenario,
        )

    def to_public_mapping(self) -> dict[str, object]:
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "source_revision": self.source_revision,
            "route_id": self.route_id,
            "canonical_provider": self.canonical_provider,
            "canonical_model_id": self.canonical_model_id,
            "protocol": self.protocol,
            "scenario": self.scenario.value,
            "status": self.status,
            "attempts": self.attempts,
            "error_kind": None if self.error_kind is None else self.error_kind.value,
        }


_RESULT_FIELDS = frozenset(
    {
        "snapshot_sha256",
        "source_revision",
        "route_id",
        "canonical_provider",
        "canonical_model_id",
        "protocol",
        "scenario",
        "status",
        "attempts",
        "error_kind",
    }
)


def _non_empty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_non_empty(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _non_empty(value, label)


def _validate_sha256(value: object) -> str:
    digest = _non_empty(value, "snapshot_sha256")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("snapshot_sha256 must be lowercase hexadecimal")
    return digest


def result_from_mapping(value: object) -> ProbeResult:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise TypeError("result must be an object with string keys")
    actual = frozenset(value)
    unknown = actual - _RESULT_FIELDS
    missing = _RESULT_FIELDS - actual
    if unknown:
        raise ValueError(f"result has unknown fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"result is missing fields: {sorted(missing)!r}")
    try:
        scenario = Scenario(value["scenario"])
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid scenario") from exc
    status = value["status"]
    if status not in {"passed", "failed"}:
        raise ValueError("invalid result status")
    error_value = value["error_kind"]
    try:
        error_kind = None if error_value is None else ErrorKind(error_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid error_kind") from exc
    attempts = value["attempts"]
    if not isinstance(attempts, int):
        raise TypeError("attempts must be an integer")
    return ProbeResult(
        snapshot_sha256=_validate_sha256(value["snapshot_sha256"]),
        source_revision=_non_empty(value["source_revision"], "source_revision"),
        route_id=_non_empty(value["route_id"], "route_id"),
        canonical_provider=_optional_non_empty(
            value["canonical_provider"], "canonical_provider"
        ),
        canonical_model_id=_optional_non_empty(
            value["canonical_model_id"], "canonical_model_id"
        ),
        protocol=_non_empty(value["protocol"], "protocol"),
        scenario=scenario,
        status=status,
        attempts=attempts,
        error_kind=error_kind,
    )


class ResultLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._results: dict[ResultKey, ProbeResult] = {}

    @classmethod
    def load(cls, path: Path) -> ResultLedger:
        ledger = cls(path)
        if not path.exists():
            return ledger
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                raise ValueError(f"checkpoint line {line_number} is empty")
            try:
                raw: Any = json.loads(line)
                result = result_from_mapping(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"checkpoint line {line_number} is invalid") from exc
            if result.key in ledger._results:
                raise ValueError("checkpoint contains a duplicate result key")
            ledger._results[result.key] = result
        return ledger

    @property
    def results(self) -> tuple[ProbeResult, ...]:
        return tuple(self._results[key] for key in sorted(self._results))

    def reusable_pass(self, key: ResultKey) -> ProbeResult | None:
        result = self._results.get(key)
        if result is None or result.status != "passed":
            return None
        return result

    def record(self, result: ProbeResult) -> None:
        self._results[result.key] = result
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as stream:
                for result in self.results:
                    stream.write(
                        json.dumps(
                            result.to_public_mapping(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(_REPLACE_ATTEMPTS):
                try:
                    os.replace(temporary, self.path)
                    break
                except PermissionError:
                    if attempt == _REPLACE_ATTEMPTS - 1:
                        raise
                    time.sleep(_REPLACE_RETRY_SECONDS)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = [
    "ErrorKind",
    "ProbeResult",
    "ResultKey",
    "ResultLedger",
    "ResultStatus",
    "result_from_mapping",
]
