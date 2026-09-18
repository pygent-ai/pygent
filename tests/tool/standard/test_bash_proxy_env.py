from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from pygent.tool.standard._bash import BashTools


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Start from a clean environment to avoid flakiness
    env = {"PATH": os.environ.get("PATH", "")}
    monkeypatch.setattr(os, "environ", env, raising=False)


def _mk_completed(stdout: str) -> Any:
    return SimpleNamespace(stdout=stdout)


def test_windows_netsh_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Simulate Windows platform, skip winreg (handled by broad exception), use netsh output
    monkeypatch.setattr(sys, "platform", "win32", raising=False)

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        cmd = " ".join(args)
        if cmd.startswith("netsh winhttp show proxy"):
            return _mk_completed(
                """
                Current WinHTTP proxy settings:

                    Proxy Server(s) : http=proxy.local:8080;https=secure.local:8443;socks=socks.local:1080
                    Bypass List     : localhost;*.internal,10.*
                """.strip()
            )
        return _mk_completed("")

    monkeypatch.setattr("subprocess.run", fake_run)

    tools = BashTools(workspace_root=tmp_path, bash_executable="bash")
    detected = tools._detect_system_proxy_env(os.environ.copy())

    assert detected["HTTP_PROXY"] == "http://proxy.local:8080"
    assert detected["HTTPS_PROXY"] == "http://secure.local:8443"
    assert detected["ALL_PROXY"] == "socks5://socks.local:1080"
    # Bypass list split by comma in our parser
    assert detected["NO_PROXY"] == "localhost,*.internal,10.*"


def test_macos_scutil_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)

    scutil_out = """
    <dictionary> {
      HTTPEnable : 1
      HTTPProxy : http.local
      HTTPPort : 8080
      HTTPSEnable : 1
      HTTPSProxy : https.local
      HTTPSPort : 8443
      SOCKSEnable : 1
      SOCKSProxy : socks.local
      SOCKSPort : 1080
      ExceptionsList : (
        "localhost",
        ".corp.local"
      )
    }
    """.strip()

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        cmd = " ".join(args)
        if cmd == "scutil --proxy":
            return _mk_completed(scutil_out)
        return _mk_completed("")

    monkeypatch.setattr("subprocess.run", fake_run)

    tools = BashTools(workspace_root=tmp_path, bash_executable="bash")
    detected = tools._detect_system_proxy_env(os.environ.copy())

    assert detected["HTTP_PROXY"] == "http://http.local:8080"
    assert detected["HTTPS_PROXY"] == "http://https.local:8443"
    assert detected["ALL_PROXY"] == "socks5://socks.local:1080"
    assert detected["NO_PROXY"] == "localhost,.corp.local"


def test_linux_gsettings_parsing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "linux", raising=False)

    def fake_which(cmd: str) -> str | None:
        return "/usr/bin/gsettings" if cmd == "gsettings" else None

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        cmd = " ".join(args)
        if cmd.endswith("org.gnome.system.proxy mode"):
            return _mk_completed("'manual'\n")
        if cmd.endswith("org.gnome.system.proxy.http host"):
            return _mk_completed("'http.local'\n")
        if cmd.endswith("org.gnome.system.proxy.http port"):
            return _mk_completed("8080\n")
        if cmd.endswith("org.gnome.system.proxy.https host"):
            return _mk_completed("'https.local'\n")
        if cmd.endswith("org.gnome.system.proxy.https port"):
            return _mk_completed("8443\n")
        if cmd.endswith("org.gnome.system.proxy.socks host"):
            return _mk_completed("'socks.local'\n")
        if cmd.endswith("org.gnome.system.proxy.socks port"):
            return _mk_completed("1080\n")
        if cmd.endswith("org.gnome.system.proxy ignore-hosts"):
            return _mk_completed("['localhost', '127.0.0.1']\n")
        return _mk_completed("")

    monkeypatch.setattr("shutil.which", fake_which)
    monkeypatch.setattr("subprocess.run", fake_run)

    tools = BashTools(workspace_root=tmp_path, bash_executable="bash")
    detected = tools._detect_system_proxy_env(os.environ.copy())

    assert detected["HTTP_PROXY"] == "http://http.local:8080"
    assert detected["HTTPS_PROXY"] == "http://https.local:8443"
    assert detected["ALL_PROXY"] == "socks5://socks.local:1080"
    assert detected["NO_PROXY"] == "localhost,127.0.0.1"


def test_injection_does_not_override_and_mirrors_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Prepare existing env with uppercase HTTP_PROXY; detection suggests a different value
    os.environ["HTTP_PROXY"] = "http://user-proxy:9000"

    tools = BashTools(workspace_root=tmp_path)

    calls: list[dict[str, str]] = []

    def fake_detect(_env: dict[str, str]) -> dict[str, str]:
        out = {"HTTP_PROXY": "http://system-proxy:8000"}
        calls.append(out)
        return out

    monkeypatch.setattr(tools, "_detect_system_proxy_env", fake_detect)  # type: ignore[attr-defined]

    kwargs = tools._process_kwargs(cwd=str(tmp_path), output=None)
    env = kwargs["env"]

    # Should keep user value and mirror to lowercase
    assert env["HTTP_PROXY"] == "http://user-proxy:9000"
    assert env["http_proxy"] == "http://user-proxy:9000"
    # Detection was called exactly once
    assert len(calls) == 1


def test_detection_cached_in_process_kwargs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tools = BashTools(workspace_root=tmp_path)
    counter = {"n": 0}

    def fake_detect(_env: dict[str, str]) -> dict[str, str]:
        counter["n"] += 1
        return {"HTTP_PROXY": "http://cache.test:8080"}

    monkeypatch.setattr(tools, "_detect_system_proxy_env", fake_detect)  # type: ignore[attr-defined]

    env1 = tools._process_kwargs(cwd=str(tmp_path), output=None)["env"]
    env2 = tools._process_kwargs(cwd=str(tmp_path), output=None)["env"]

    # First call populates, second call uses cache
    assert counter["n"] == 1
    assert env1["HTTP_PROXY"] == "http://cache.test:8080"
    assert env2["HTTP_PROXY"] == "http://cache.test:8080"
