"""Definition-time native shell resolution for the standard Bash adapter."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import pytest

from pygent.tool import ShellIdentity, describe_shell_environment
from pygent.tool.standard import _bash as bash_module
from pygent.tool.standard import _shell as shell_module
from pygent.tool.standard._bash import BashTools, _find_bash_executable
from pygent.tool.standard._shell import ShellResolver, probe_version


def _resolver(name: str = "bash", **overrides: object) -> ShellResolver:
    arguments: dict[str, object] = {
        "candidates": lambda: ["first-bash", "second-bash"],
        "probe": lambda executable: executable == "second-bash",
    }
    arguments.update(overrides)
    return ShellResolver(name, **arguments)  # type: ignore[arg-type]


def test_shell_ut_identity_rejects_invalid_fields():
    with pytest.raises(TypeError):
        ShellIdentity(platform="windows", name="bash", executable=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ShellIdentity(platform="windows", name="", executable="bash")
    with pytest.raises(ValueError):
        ShellIdentity(platform="windows", name="bash", executable="bash", args=("",))
    with pytest.raises(ValueError):
        ShellIdentity(platform="windows", name="bash", executable="bash", version=" ")


def test_shell_ut_identity_carries_no_invocation_state():
    fields = {field.name for field in dataclasses.fields(ShellIdentity)}

    assert fields == {"platform", "name", "executable", "args", "version"}
    identity = ShellIdentity(
        platform="linux",
        name="bash",
        executable="/bin/bash",
        version="5.2.15",
    )
    assert identity.display_name == "bash 5.2.15"
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.executable = "/usr/bin/bash"  # type: ignore[misc]


def test_shell_ut_resolver_probes_candidates_in_order():
    assert _resolver().resolve().executable == "second-bash"


def test_shell_ut_resolver_prefers_explicit_executable_without_probing():
    probed: list[str] = []

    def probe(executable: str) -> bool:
        probed.append(executable)
        return False

    resolver = _resolver(executable="configured-bash", probe=probe)

    assert resolver.resolve().executable == "configured-bash"
    assert probed == []


def test_shell_ut_resolver_prefers_environment_override_over_candidates():
    resolver = _resolver(env_override="env-bash")

    assert resolver.resolve().executable == "env-bash"


def test_shell_ut_resolver_falls_back_to_first_candidate_when_no_probe_succeeds():
    resolver = _resolver(probe=lambda executable: False)

    assert resolver.resolve().executable == "first-bash"


def test_shell_ut_resolver_uses_declared_fallback_without_candidates():
    resolver = _resolver(candidates=list, fallback="bash")

    assert resolver.resolve().executable == "bash"


def test_shell_ut_resolver_reports_missing_shell():
    resolver = _resolver(candidates=list, fallback=None)

    with pytest.raises(ValueError, match="no bash executable"):
        resolver.resolve()


def test_shell_ut_resolver_rejects_invalid_arguments():
    with pytest.raises(TypeError):
        _resolver(candidates="not-callable")
    with pytest.raises(TypeError):
        _resolver(probe="not-callable")
    with pytest.raises(ValueError):
        _resolver(name="")
    with pytest.raises(ValueError):
        _resolver(executable=" ")


def test_shell_ut_resolver_reports_platform_and_version(monkeypatch):
    monkeypatch.setattr(shell_module.sys, "platform", "win32")
    monkeypatch.setattr(shell_module, "_VERSION_CACHE", {})
    monkeypatch.setattr(
        shell_module,
        "probe_version",
        lambda name, executable, command: "5.2.15",
    )
    resolver = _resolver(version_command='printf %s "$BASH_VERSION"')

    identity = resolver.resolve()

    assert (identity.platform, identity.name) == ("windows", "bash")
    assert identity.version == "5.2.15"


def test_shell_ut_probe_version_parses_and_caches(monkeypatch):
    calls: list[list[str]] = []

    class Process:
        returncode = 0
        stdout = b"5.2.15(1)-release\n"

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Process()

    monkeypatch.setattr(shell_module, "_VERSION_CACHE", {})
    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)

    assert probe_version("bash", "bash", "printf %s \"$BASH_VERSION\"") == "5.2.15"
    assert probe_version("bash", "bash", "printf %s \"$BASH_VERSION\"") == "5.2.15"
    assert len(calls) == 1


def test_shell_ut_probe_version_reports_unknown_executable(monkeypatch):
    monkeypatch.setattr(shell_module, "_VERSION_CACHE", {})

    assert probe_version("bash", "missing-bash", 'printf %s "$BASH_VERSION"') is None


def test_shell_ut_describe_environment_states_facts_only():
    identity = ShellIdentity(
        platform="windows",
        name="bash",
        executable=r"C:\Program Files\Git\bin\bash.exe",
        version="5.2.15",
    )

    described = describe_shell_environment(identity, cwd=r"C:\Projects\pygent")

    assert described.splitlines() == [
        "Operating System: windows",
        "Shell: bash 5.2.15",
        r"Shell Executable: C:\Program Files\Git\bin\bash.exe",
        r"Working Directory: C:\Projects\pygent",
    ]
    assert "Use " not in described
    assert "syntax" not in described


def test_shell_ut_describe_environment_rejects_invalid_input():
    identity = ShellIdentity(platform="linux", name="bash", executable="/bin/bash")

    with pytest.raises(TypeError):
        describe_shell_environment("bash", cwd="/tmp")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        describe_shell_environment(identity, cwd=" ")


def test_shell_ut_bash_tools_exposes_the_resolved_identity(tmp_path):
    tools = BashTools(workspace_root=tmp_path)

    assert tools.shell_identity.name == "bash"
    assert tools.shell_identity.executable == tools.bash_executable
    assert tools.shell_identity.executable == _find_bash_executable()
    assert tools.shell_identity.platform in {"windows", "linux", "darwin", sys.platform}


def test_shell_ut_bash_tools_keeps_explicit_executable_override(tmp_path):
    tools = BashTools(workspace_root=tmp_path, bash_executable="custom-bash")

    assert tools.bash_executable == "custom-bash"
    assert tools.shell_identity.executable == "custom-bash"


def test_shell_ut_bash_resolver_preserves_env_override(monkeypatch):
    monkeypatch.setenv("PYGENT_BASH_PATH", "env-bash")

    assert _find_bash_executable() == "env-bash"


def test_shell_ut_bash_resolver_skips_nonfunctional_candidates(monkeypatch):
    monkeypatch.delenv("PYGENT_BASH_PATH", raising=False)
    monkeypatch.setattr(
        bash_module, "_bash_candidates", lambda: ["wsl-bash", "git-bash"]
    )
    monkeypatch.setattr(
        bash_module, "_is_functional_bash", lambda executable: executable == "git-bash"
    )

    assert _find_bash_executable() == "git-bash"


@pytest.mark.parametrize(
    ("platform", "expected"),
    [
        ("win32", "windows"),
        ("darwin", "darwin"),
        ("linux", "linux"),
    ],
)
def test_shell_ut_platform_names(monkeypatch, platform, expected):
    monkeypatch.setattr(shell_module.sys, "platform", platform)
    resolver = _resolver(probe_enabled=False)

    assert resolver.resolve().platform == expected


def test_shell_ut_probe_timeout_is_bounded(monkeypatch):
    def fake_run(argv, **kwargs):
        assert kwargs["timeout"] == shell_module._PROBE_TIMEOUT_SECONDS
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs["timeout"])

    monkeypatch.setattr(shell_module, "_VERSION_CACHE", {})
    monkeypatch.setattr(shell_module.subprocess, "run", fake_run)

    assert probe_version("powershell", "pwsh", "$PSVersionTable") is None


def test_shell_ut_probe_decodes_utf16_output(monkeypatch):
    class Process:
        returncode = 0
        stdout = "7.5.2".encode("utf-16-le")

    monkeypatch.setattr(shell_module, "_VERSION_CACHE", {})
    monkeypatch.setattr(shell_module.subprocess, "run", lambda argv, **kwargs: Process())

    assert probe_version("powershell", "pwsh", "$PSVersionTable") == "7.5.2"


def test_shell_ut_standard_tools_export_the_identity():
    from pygent.tool.standard import ShellIdentity as StandardShellIdentity

    assert StandardShellIdentity is ShellIdentity
