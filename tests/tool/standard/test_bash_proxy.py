"""System proxy detection and environment injection for the bash adapter.

These tests pin the mapping from host proxy settings (Windows WinINET /
WinHTTP, macOS scutil, Linux GNOME gsettings) into the environment passed to
the bash subprocess, and verify that user-provided variables are never
overridden while case variants are mirrored.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from pygent.tool import BashTools

# ── Windows WinINET registry (winreg) ───────────────────────────────────────


def _fake_winreg(enable: int, server: str, override: str = "") -> types.ModuleType:
    """Build a fake ``winreg`` module with a single Internet Settings key."""

    values = {
        "ProxyEnable": enable,
        "ProxyServer": server,
        "ProxyOverride": override,
    }

    class _FakeKey:
        def __enter__(self):
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    def _query_value(key, name):
        if name in values:
            return values[name], 1
        raise FileNotFoundError(name)

    fake = types.ModuleType("winreg")
    fake.HKEY_CURRENT_USER = object()
    fake.QueryValueEx = _query_value
    fake.OpenKey = lambda _root, _path: _FakeKey()
    return fake


def _detect_proxy(
    monkeypatch: pytest.MonkeyPatch,
    *,
    platform: str,
    winreg: types.ModuleType | None = None,
    subprocess_stdout: str | None = None,
    gsettings_available: bool = True,
) -> dict[str, str]:
    """Call ``BashTools._detect_system_proxy_env`` with the platform mocked."""

    # Build the tools first (bash discovery may itself run subprocesses),
    # then force a fresh detection pass with the platform/subprocess mocked.
    import shutil

    bash = shutil.which("bash")
    if bash is None:
        bash = os.environ.get("PYGENT_BASH_PATH")
    tools = BashTools(
        workspace_root=Path("."),
        bash_executable=bash,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(sys, "platform", platform)

    # Default: disable WinINET registry proxy so tests are independent
    # of the actual machine's proxy settings.
    if platform == "win32" and winreg is None:
        winreg = _fake_winreg(0, "")
    if winreg is not None:
        monkeypatch.setitem(sys.modules, "winreg", winreg)

    def fake_run(args, **kwargs):
        class _Proc:
            stdout = subprocess_stdout or ""

        return _Proc()

    if subprocess_stdout is not None or platform == "linux":
        monkeypatch.setattr("subprocess.run", fake_run)
        monkeypatch.setattr(
            "shutil.which",
            lambda name: "/usr/bin/gsettings" if gsettings_available else None,
        )

    tools._system_proxy_env = None  # type: ignore[attr-defined]
    return tools._detect_system_proxy_env({"PATH": "/usr/bin"})


def test_proxy_winreg_basic_http_https(monkeypatch):
    """ProxyEnable=1 with a bare server sets HTTP_PROXY and HTTPS_PROXY."""

    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        winreg=_fake_winreg(1, "127.0.0.1:8080"),
    )

    assert result["HTTP_PROXY"] == "http://127.0.0.1:8080"
    assert result["HTTPS_PROXY"] == "http://127.0.0.1:8080"


def test_proxy_winreg_split_protocols(monkeypatch):
    """Per-protocol entries map to their own variables."""

    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        winreg=_fake_winreg(1, "http=proxy-a:8080;https=proxy-b:8443;socks=socks-c:1080"),
    )

    assert result["HTTP_PROXY"] == "http://proxy-a:8080"
    assert result["HTTPS_PROXY"] == "http://proxy-b:8443"
    assert result["ALL_PROXY"] == "socks5://socks-c:1080"


def test_proxy_winreg_override_becomes_no_proxy(monkeypatch):
    """ProxyOverride entries become NO_PROXY."""

    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        winreg=_fake_winreg(1, "127.0.0.1:8080", "<local>;*.example.com;10.0.0.0/8"),
    )

    assert result["NO_PROXY"] == "<local>,*.example.com,10.0.0.0/8"


def test_proxy_winreg_disabled_returns_empty(monkeypatch):
    """ProxyEnable=0 yields no proxy variables."""

    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        winreg=_fake_winreg(0, "127.0.0.1:8080"),
    )

    assert result == {}


# ── Windows WinHTTP (netsh fallback) ────────────────────────────────────────


def test_proxy_netsh_direct_access_is_empty(monkeypatch):
    """netsh reporting Direct access yields no proxy variables."""

    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        subprocess_stdout="\n    Direct access (no proxy server).\n",
    )

    assert result == {}


def test_proxy_netsh_proxy_server_and_bypass(monkeypatch):
    """netsh proxy server + bypass list are parsed."""

    output = (
        "    Proxy Server(s) : 127.0.0.1:8080\n"
        "    Bypass List     : <local>;*.example.com\n"
    )
    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        subprocess_stdout=output,
    )

    assert result["HTTP_PROXY"] == "http://127.0.0.1:8080"
    assert result["HTTPS_PROXY"] == "http://127.0.0.1:8080"
    assert result["NO_PROXY"] == "<local>,*.example.com"


def test_proxy_netsh_split_protocols(monkeypatch):
    """netsh per-protocol entries are honored."""

    output = "    Proxy Server(s) : http=proxy-a:8080;https=proxy-b:8443\n"
    result = _detect_proxy(
        monkeypatch,
        platform="win32",
        subprocess_stdout=output,
    )

    assert result["HTTP_PROXY"] == "http://proxy-a:8080"
    assert result["HTTPS_PROXY"] == "http://proxy-b:8443"


# ── macOS scutil ────────────────────────────────────────────────────────────


def test_proxy_scutil_http_https(monkeypatch):
    """scutil --proxy HTTP/HTTPS entries map to the proxy variables."""

    output = (
        "<dictionary> {\n"
        "    HTTPEnable : 1\n"
        "    HTTPProxy : proxy-a.local\n"
        "    HTTPPort : 8080\n"
        "    HTTPSEnable : 1\n"
        "    HTTPSProxy : proxy-b.local\n"
        "    HTTPSPort : 8443\n"
        "}\n"
    )
    result = _detect_proxy(
        monkeypatch,
        platform="darwin",
        subprocess_stdout=output,
    )

    assert result["HTTP_PROXY"] == "http://proxy-a.local:8080"
    assert result["HTTPS_PROXY"] == "http://proxy-b.local:8443"


def test_proxy_scutil_socks_and_exceptions(monkeypatch):
    """scutil SOCKS proxy and ExceptionsList become ALL_PROXY / NO_PROXY."""

    output = (
        "<dictionary> {\n"
        "    SOCKSEnable : 1\n"
        "    SOCKSProxy : socks.local\n"
        "    SOCKSPort : 1080\n"
        "    ExceptionsList : (\n"
        "        'localhost',\n"
        "        '*.example.com'\n"
        "    )\n"
        "}\n"
    )
    result = _detect_proxy(
        monkeypatch,
        platform="darwin",
        subprocess_stdout=output,
    )

    assert result["ALL_PROXY"] == "socks5://socks.local:1080"
    assert result["NO_PROXY"] == "localhost,*.example.com"


def test_proxy_scutil_disabled_returns_empty(monkeypatch):
    """scutil with all enable flags off yields nothing."""

    output = (
        "<dictionary> {\n"
        "    HTTPEnable : 0\n"
        "    HTTPSEnable : 0\n"
        "    SOCKSEnable : 0\n"
        "}\n"
    )
    result = _detect_proxy(
        monkeypatch,
        platform="darwin",
        subprocess_stdout=output,
    )

    assert result == {}


# ── Linux GNOME gsettings ───────────────────────────────────────────────────


def _fake_proc(stdout: str = "") -> object:
    """Return a fake CompletedProcess-like object with the given stdout."""

    class _Proc:
        returncode = 0

        def __init__(self, s: str = "") -> None:
            self.stdout = s

    return _Proc(stdout)


def test_proxy_gsettings_manual_mode(monkeypatch):
    """gsettings manual mode maps hosts/ports to proxy variables."""

    import shutil

    bash = shutil.which("bash")
    # Build tools BEFORE monkeypatching subprocess.run
    tools = BashTools(workspace_root=Path("."), bash_executable=bash)  # type: ignore[arg-type]

    def gsettings_run(args, **kwargs):
        if args[0] != "gsettings":
            return _fake_proc()
        schema, key = args[2], args[3]
        gvalues = {
            ("org.gnome.system.proxy", "mode"): "'manual'",
            ("org.gnome.system.proxy.http", "host"): "'proxy-a.local'",
            ("org.gnome.system.proxy.http", "port"): "8080",
            ("org.gnome.system.proxy.https", "host"): "'proxy-b.local'",
            ("org.gnome.system.proxy.https", "port"): "8443",
            ("org.gnome.system.proxy.socks", "host"): "'socks.local'",
            ("org.gnome.system.proxy.socks", "port"): "1080",
            ("org.gnome.system.proxy", "ignore-hosts"): "['localhost', '*.example.com']",
        }
        return _fake_proc(str(gvalues.get((schema, key), "''")))

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr("subprocess.run", gsettings_run)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/gsettings" if name == "gsettings" else None
    )

    tools._system_proxy_env = None  # type: ignore[attr-defined]
    result = tools._detect_system_proxy_env({"PATH": "/usr/bin"})

    assert result["HTTP_PROXY"] == "http://proxy-a.local:8080"
    assert result["HTTPS_PROXY"] == "http://proxy-b.local:8443"
    assert result["ALL_PROXY"] == "socks5://socks.local:1080"
    assert result["NO_PROXY"] == "localhost,*.example.com"


def test_proxy_gsettings_not_manual_is_empty(monkeypatch):
    """gsettings mode != manual yields no proxy variables."""

    import shutil

    bash = shutil.which("bash")
    # Build tools BEFORE monkeypatching subprocess.run
    tools = BashTools(workspace_root=Path("."), bash_executable=bash)  # type: ignore[arg-type]

    monkeypatch.setattr(sys, "platform", "linux")

    def fake_run(args, **kwargs):
        return _fake_proc("'none'")

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(
        "shutil.which", lambda name: "/usr/bin/gsettings" if name == "gsettings" else None
    )

    tools._system_proxy_env = None  # type: ignore[attr-defined]
    result = tools._detect_system_proxy_env({"PATH": "/usr/bin"})

    assert result == {}


# ── Environment injection ───────────────────────────────────────────────────


def _tools_with_proxy(monkeypatch: pytest.MonkeyPatch) -> BashTools:
    """Return BashTools whose cached proxy env is a known value."""

    tools = BashTools(workspace_root=Path("."))
    tools._system_proxy_env = {  # type: ignore[attr-defined]
        "HTTP_PROXY": "http://127.0.0.1:8080",
        "NO_PROXY": "<local>",
    }
    return tools


def test_proxy_env_does_not_override_user_variables(monkeypatch):
    """User-provided proxy variables win over detected ones."""

    tools = _tools_with_proxy(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://user-set:9999")

    kwargs = tools._process_kwargs(str(Path(".")), None)
    env = kwargs["env"]

    assert env["HTTP_PROXY"] == "http://user-set:9999"


def test_proxy_env_injects_detected_variables(monkeypatch):
    """Detected proxy variables are added to the subprocess environment."""

    tools = _tools_with_proxy(monkeypatch)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    kwargs = tools._process_kwargs(str(Path(".")), None)
    env = kwargs["env"]

    assert env["HTTP_PROXY"] == "http://127.0.0.1:8080"
    assert env["NO_PROXY"] == "<local>"


def test_proxy_env_mirrors_case_variants(monkeypatch):
    """Lowercase variants are mirrored when only uppercase exists."""

    tools = _tools_with_proxy(monkeypatch)
    monkeypatch.delenv("http_proxy", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    kwargs = tools._process_kwargs(str(Path(".")), None)
    env = kwargs["env"]

    assert env["HTTP_PROXY"] == "http://127.0.0.1:8080"
    assert env["http_proxy"] == "http://127.0.0.1:8080"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Git Bash / MSYS2 does not forward HTTP_PROXY into child processes "
    "the same way POSIX bash does; environment injection is validated on POSIX",
)
def test_proxy_env_injected_proxy_visible_to_bash(monkeypatch, tmp_path):
    """The injected proxy environment is actually visible inside bash."""

    import asyncio

    tools = _tools_with_proxy(monkeypatch)
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("http_proxy", raising=False)

    async def run():
        async with tools:
            output = await tools.bash("printf '%s' \"$HTTP_PROXY\"", timeout=5)
            return output

    output = asyncio.run(run())
    assert "http://127.0.0.1:8080" in str(output)
