"""Validated TOML configuration for repeatable benchmark campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

BackendName = Literal["synthetic", "live"]
LoadShape = Literal["closed", "open"]


@dataclass(frozen=True, slots=True)
class ModelSettings:
    latency_ms: float = 25.0
    jitter_ms: float = 0.0
    ttft_ms: float = 5.0
    chunks: int = 3
    tool_latency_ms: float = 2.0
    retry_max_attempts: int = 1
    retry_on: tuple[str, ...] = ()
    retry_backoff_seconds: float = 0.0
    attempt_idle_timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "latency_ms",
            "jitter_ms",
            "ttft_ms",
            "tool_latency_ms",
            "retry_backoff_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if isinstance(self.chunks, bool) or self.chunks <= 0:
            raise ValueError("chunks must be a positive integer")
        if (
            isinstance(self.retry_max_attempts, bool)
            or self.retry_max_attempts < 1
        ):
            raise ValueError("retry_max_attempts must be a positive integer")
        retry_on = tuple(self.retry_on)
        supported = {"timeout", "rate_limit", "unavailable", "incomplete_response"}
        if len(retry_on) != len(set(retry_on)) or any(
            kind not in supported for kind in retry_on
        ):
            raise ValueError(
                "retry_on must contain unique timeout, rate_limit, unavailable, or "
                "incomplete_response values"
            )
        object.__setattr__(self, "retry_on", retry_on)
        if self.retry_max_attempts > 1 and not retry_on:
            raise ValueError("retry_on is required when retry_max_attempts is greater than one")
        timeout = self.attempt_idle_timeout_seconds
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "attempt_idle_timeout_seconds must be finite and positive"
            )


@dataclass(frozen=True, slots=True)
class LoadProfile:
    name: str
    backend: BackendName
    scenarios: tuple[str, ...]
    concurrency: tuple[int, ...]
    closed_duration_seconds: float
    open_multipliers: tuple[float, ...]
    open_duration_seconds: float
    warmup_seconds: float = 0.0
    cooldown_seconds: float = 0.0
    repetitions: int = 1
    max_inflight: int = 128
    request_deadline_seconds: float = 60.0
    seed: int = 1729
    model: ModelSettings = ModelSettings()

    def __post_init__(self) -> None:
        if not self.name or self.backend not in ("synthetic", "live"):
            raise ValueError("profile name and supported backend are required")
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        if not self.concurrency or any(
            isinstance(value, bool) or value <= 0 for value in self.concurrency
        ):
            raise ValueError("concurrency must contain positive integers")
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("concurrency values must be unique")
        for name in (
            "closed_duration_seconds",
            "open_duration_seconds",
            "warmup_seconds",
            "cooldown_seconds",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.closed_duration_seconds == 0 and self.open_duration_seconds == 0:
            raise ValueError("at least one measured duration must be positive")
        if any(not math.isfinite(value) or value <= 0 for value in self.open_multipliers):
            raise ValueError("open multipliers must be finite and positive")
        for name in ("repetitions", "max_inflight"):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.request_deadline_seconds <= 0 or not math.isfinite(
            self.request_deadline_seconds
        ):
            raise ValueError("request_deadline_seconds must be finite and positive")
        if self.backend == "live" and max(self.concurrency) > 32:
            raise ValueError("live model concurrency is hard-limited to 32")

    @property
    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def estimated_seconds(self) -> float:
        stages = len(self.scenarios) * self.repetitions
        measured = len(self.concurrency) * self.closed_duration_seconds
        measured += len(self.open_multipliers) * self.open_duration_seconds
        phases = (self.warmup_seconds + self.cooldown_seconds) * (
            len(self.concurrency) + len(self.open_multipliers)
        )
        return stages * (measured + phases)


def load_profile(path: str | Path) -> LoadProfile:
    source = Path(path)
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    profile = raw.get("profile")
    model = raw.get("model", {})
    if not isinstance(profile, dict) or not isinstance(model, dict):
        raise TypeError("profile TOML requires [profile] and optional [model]")
    return LoadProfile(
        name=profile["name"],
        backend=profile["backend"],
        scenarios=tuple(profile["scenarios"]),
        concurrency=tuple(profile["concurrency"]),
        closed_duration_seconds=profile["closed_duration_seconds"],
        open_multipliers=tuple(profile.get("open_multipliers", ())),
        open_duration_seconds=profile.get("open_duration_seconds", 0.0),
        warmup_seconds=profile.get("warmup_seconds", 0.0),
        cooldown_seconds=profile.get("cooldown_seconds", 0.0),
        repetitions=profile.get("repetitions", 1),
        max_inflight=profile.get("max_inflight", 128),
        request_deadline_seconds=profile.get("request_deadline_seconds", 60.0),
        seed=profile.get("seed", 1729),
        model=ModelSettings(**model),
    )


__all__ = ["BackendName", "LoadProfile", "LoadShape", "ModelSettings", "load_profile"]
