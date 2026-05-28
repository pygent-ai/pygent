"""Compatibility wrapper for the renamed bash toolkit."""

from .bash import (
    BashToolkits,
    TerminalToolkits,
    _DEFAULT_TIMEOUT_MS,
    _MAX_OUTPUT_BYTES,
    _MAX_TIMEOUT_MS,
    _decode_output,
)


__all__ = [
    "BashToolkits",
    "TerminalToolkits",
    "_DEFAULT_TIMEOUT_MS",
    "_MAX_OUTPUT_BYTES",
    "_MAX_TIMEOUT_MS",
    "_decode_output",
]
