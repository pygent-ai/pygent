"""Resolve one foreground wait policy for native and model tool calls."""

import math


def resolve_wait_timeout(
    default: float | None, override: object = None, *, is_background: bool = False
) -> float | None:
    value = default if override is None else override
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("timeout must be a finite, non-negative number of seconds")
    return 0.0 if is_background else value
