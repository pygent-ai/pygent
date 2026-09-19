"""Definition-time native shell identity for the standard tool adapters.

A resolved :class:`ShellIdentity` answers "which native shell does this assembly
use" once, at assembly time. It carries no working directory, environment
mutation, shell variable, history or process state, so it stays definition
configuration instead of invocation state.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_PROBE_TIMEOUT_SECONDS = 5.0
_VERSION_PATTERN = re.compile(r"\d+(?:\.\d+)+")
_VERSION_CACHE: dict[tuple[str, str, str], str | None] = {}


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def append_unique_path(paths: list[str], candidate: str | None) -> None:
    """Append one shell candidate path unless an equivalent path is present."""

    if not candidate:
        return
    normalized = os.path.normcase(
        os.path.abspath(os.path.expandvars(os.path.expanduser(candidate)))
    )
    existing = {os.path.normcase(os.path.abspath(item)) for item in paths}
    if normalized not in existing:
        paths.append(candidate)


def _platform_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return sys.platform


@dataclass(frozen=True, slots=True)
class ShellIdentity:
    """Immutable definition-time description of one resolved native shell."""

    platform: str
    name: str
    executable: str
    args: tuple[str, ...] = ()
    version: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.platform, "platform")
        _require_text(self.name, "name")
        _require_text(self.executable, "executable")
        args = tuple(self.args)
        if any(not isinstance(item, str) or not item.strip() for item in args):
            raise ValueError("args must contain non-empty strings")
        object.__setattr__(self, "args", args)
        if self.version is not None:
            _require_text(self.version, "version")

    @property
    def display_name(self) -> str:
        """Return ``name version`` when the version is known, otherwise the name."""

        return f"{self.name} {self.version}" if self.version else self.name


def _probe_argv(name: str, executable: str, command: str) -> list[str]:
    if name == "powershell":
        return [executable, "-NoLogo", "-NoProfile", "-Command", command]
    return [executable, "-lc", command]


def _run_probe(argv: list[str]) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _probe_text(data: bytes) -> str:
    # Windows PowerShell can emit UTF-16LE over a redirected pipe.
    return data.replace(b"\x00", b"").decode("utf-8", errors="replace").strip()


def run_probe(name: str, executable: str, command: str) -> tuple[int, str] | None:
    """Run one bounded probe command and return its exit status and stdout text."""

    process = _run_probe(_probe_argv(name, executable, command))
    if process is None:
        return None
    return process.returncode, _probe_text(process.stdout)


def probe_version(name: str, executable: str, command: str) -> str | None:
    """Return the version reported by ``command``, cached per executable."""

    key = (name, executable, command)
    if key not in _VERSION_CACHE:
        resolved: str | None = None
        probed = run_probe(name, executable, command)
        if probed is not None:
            returncode, output = probed
            if returncode == 0:
                match = _VERSION_PATTERN.search(output)
                if match is not None:
                    resolved = match.group(0)
        _VERSION_CACHE[key] = resolved
    return _VERSION_CACHE[key]


class ShellResolver:
    """Resolve one native shell executable at assembly time.

    Precedence follows the native shell design: an explicit executable wins, then
    the shell-specific environment override, then the detected candidates in
    order, then the declared fallback. ``candidates`` and ``probe`` are supplied
    by the shell adapter, so this class carries no shell-specific knowledge.
    """

    def __init__(
        self,
        name: str,
        *,
        candidates: Callable[[], Sequence[str]],
        probe: Callable[[str], bool],
        executable: str | None = None,
        env_override: str | None = None,
        args: tuple[str, ...] = (),
        version_command: str | None = None,
        probe_enabled: bool = True,
        fallback: str | None = None,
    ) -> None:
        self.name = _require_text(name, "name")
        if not callable(candidates):
            raise TypeError("candidates must be callable")
        if not callable(probe):
            raise TypeError("probe must be callable")
        if executable is not None:
            _require_text(executable, "executable")
        if env_override is not None:
            _require_text(env_override, "env_override")
        if version_command is not None:
            _require_text(version_command, "version_command")
        if fallback is not None:
            _require_text(fallback, "fallback")
        resolved_args = tuple(args)
        if any(not isinstance(item, str) or not item.strip() for item in resolved_args):
            raise ValueError("args must contain non-empty strings")
        self.candidates = candidates
        self.probe = probe
        self.executable = executable
        self.env_override = env_override
        self.args = resolved_args
        self.version_command = version_command
        self.probe_enabled = probe_enabled
        self.fallback = fallback

    def resolve(self) -> ShellIdentity:
        """Return the identity of the selected shell without probing explicit paths."""

        executable = self.executable or self.env_override
        if executable is None:
            candidates = [
                candidate for candidate in self.candidates() if candidate
            ]
            if self.probe_enabled:
                executable = next(
                    (item for item in candidates if self.probe(item)), None
                )
            if executable is None:
                executable = candidates[0] if candidates else self.fallback
        if executable is None:
            raise ValueError(
                f"no {self.name} executable is configured or available "
                f"on {_platform_name()}"
            )
        version = (
            probe_version(self.name, executable, self.version_command)
            if self.version_command is not None
            else None
        )
        return ShellIdentity(
            platform=_platform_name(),
            name=self.name,
            executable=executable,
            args=self.args,
            version=version,
        )


def describe_shell_environment(identity: ShellIdentity, *, cwd: str) -> str:
    """Describe the execution environment as facts for Runtime Context injection.

    Applications compose this text through their own ``Reminder`` or
    ``StandaloneUserMessage(kind=InjectionKind.RUNTIME_CONTEXT)``; it states the
    operating system, shell, executable and working directory, and never
    instructs the model or changes message roles.
    """

    if not isinstance(identity, ShellIdentity):
        raise TypeError("identity must be a ShellIdentity")
    _require_text(cwd, "cwd")
    return "\n".join(
        (
            f"Operating System: {identity.platform}",
            f"Shell: {identity.display_name}",
            f"Shell Executable: {identity.executable}",
            f"Working Directory: {cwd}",
        )
    )


__all__ = [
    "ShellIdentity",
    "ShellResolver",
    "append_unique_path",
    "describe_shell_environment",
    "probe_version",
    "run_probe",
]
