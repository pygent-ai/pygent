"""Windows ConPTY-backed interactive process for the terminal session tools.

EXPERIMENTAL: on some Windows builds the pseudo console does not forward
child output to the output pipe even though every Win32 call succeeds.  Until
that capture path is fixed, prefer the ``pipe`` backend for reliable Windows
interactive sessions.

Provides ``ConPtyProcess`` with the same duck interface as ``ShellProcess``
so that ``_terminal.py`` can select the ``conpty`` backend on Windows without
changing its control loop.

The module uses ``kernel32.CreatePseudoConsole`` + ``CreateProcessW`` with
``PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE``.  Output is polled through
``PeekNamedPipe`` + ``ReadFile`` in the event loop (no background reader
thread), because ConPTY may not close its output pipe until
``ClosePseudoConsole`` is called.
"""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.wintypes
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from . import _process as _process_mod

_STDIN_WRITE_TIMEOUT = _process_mod.STDIN_WRITE_TIMEOUT_SECONDS

# ── Win32 API constants ─────
_PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x20016
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_NEW_CONSOLE = 0x00000010
_INVALID_HANDLE_VALUE = -1
_STILL_ACTIVE = 259
# Bounded wait (ms) after TerminateProcess before closing handles.
_PROCESS_TERMINATE_WAIT_MS = 1000

# ── Win32 type aliases ─────
wintypes = ctypes.wintypes

# ── Structure definitions ────


class _COORD(ctypes.Structure):
    _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _STARTUPINFOW),
        ("lpAttributeList", wintypes.LPVOID),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


# ── Load kernel32 ────────────

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_kernel32.CreatePipe.restype = wintypes.BOOL
_kernel32.CreatePipe.argtypes = [
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(wintypes.HANDLE),
    ctypes.POINTER(_SECURITY_ATTRIBUTES),
    wintypes.DWORD,
]

_kernel32.CreatePseudoConsole.restype = ctypes.HRESULT
_kernel32.CreatePseudoConsole.argtypes = [
    ctypes.POINTER(_COORD),
    wintypes.HANDLE,
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]

_kernel32.ClosePseudoConsole.restype = None
_kernel32.ClosePseudoConsole.argtypes = [wintypes.HANDLE]

_kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
_kernel32.InitializeProcThreadAttributeList.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]

_kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
_kernel32.UpdateProcThreadAttribute.argtypes = [
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.c_ulonglong,
    wintypes.LPVOID,
    ctypes.c_size_t,
    wintypes.LPVOID,
    ctypes.POINTER(wintypes.DWORD),
]

_kernel32.DeleteProcThreadAttributeList.restype = None
_kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]

_kernel32.CreateProcessW.restype = wintypes.BOOL
_kernel32.CreateProcessW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    wintypes.LPVOID,
    wintypes.LPVOID,
    wintypes.BOOL,
    wintypes.DWORD,
    wintypes.LPVOID,
    wintypes.LPCWSTR,
    ctypes.POINTER(_STARTUPINFOEXW),
    ctypes.POINTER(_PROCESS_INFORMATION),
]

_kernel32.WriteFile.restype = wintypes.BOOL
_kernel32.WriteFile.argtypes = [
    wintypes.HANDLE,
    wintypes.LPCVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]

_kernel32.ReadFile.restype = wintypes.BOOL
_kernel32.ReadFile.argtypes = [
    wintypes.HANDLE,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    wintypes.LPVOID,
]

_kernel32.PeekNamedPipe.restype = wintypes.BOOL
_kernel32.PeekNamedPipe.argtypes = [
    wintypes.HANDLE,
    wintypes.LPVOID,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(wintypes.DWORD),
]

_kernel32.GetExitCodeProcess.restype = wintypes.BOOL
_kernel32.GetExitCodeProcess.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.DWORD),
]

_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.WaitForSingleObject.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
]

_kernel32.TerminateProcess.restype = wintypes.BOOL
_kernel32.TerminateProcess.argtypes = [
    wintypes.HANDLE,
    wintypes.UINT,
]

_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


# ── Utility ─────────────────────────────────────────────────────────────────


def _safe_close(h: int | None) -> None:
    if h and h != _INVALID_HANDLE_VALUE:
        _kernel32.CloseHandle(wintypes.HANDLE(h))


# ── Transport ────────────────

@dataclass(slots=True)
class _ConPtyTransport:
    _pid: int
    _returncode: int | None = None

    def get_pid(self) -> int:
        return self._pid

    def get_returncode(self) -> int | None:
        return self._returncode


# ── ConPtyProcess ────────────

_POLL_INTERVAL = 0.05          # seconds between polls
_PROCESS_POLL_INTERVAL = 0.25  # seconds between process-exit checks


class ConPtyProcess:
    """Windows ConPTY-backed interactive process.

    The duck interface matches ``ShellProcess``:

    * ``start(*args, cwd=..., **kwargs)`` — create ConPTY + start child.
    * ``outcome_ready`` — future completed when the child exits.
    * ``write_stdin(data, *, timeout)`` — write to the ConPTY input pipe.
    * ``captured`` / ``truncated`` / ``last_output_at`` — output capture state.
    * ``transport`` — ``_ConPtyTransport`` with ``get_pid()`` / ``get_returncode()``.
    * ``aclose(is_windows)`` — close ConPTY and terminate the child.
    * ``close()`` — release local resources immediately.
    """

    def __init__(
        self,
        output_file: Any = None,
        *,
        max_capture_bytes: int = _process_mod.MAX_FULL_OUTPUT_BYTES,
        task_prefix: str = "pygent-conpty",
    ) -> None:
        self.output_file = output_file
        self.captured: int = 0
        self.last_output_at: float = time.monotonic()
        self.truncated: bool = False
        self.max_capture_bytes = max_capture_bytes
        self.task_prefix = task_prefix

        # Win32 handles (stored as ints via .value)
        self._hPC: int | None = None
        self._hProcess: int | None = None
        self._hThread: int | None = None
        self._hStdinWrite: int | None = None
        self._hOutputRead: int | None = None
        self._child_pid: int | None = None

        self._outcome_ready: asyncio.Future[None] | None = None
        self._transport: _ConPtyTransport | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._close_requested: bool = False
        self._process_exited: bool = False
        self._poll_task: asyncio.Task[None] | None = None

    # -- read-only properties -----------------------------------------------

    @property
    def outcome_ready(self) -> asyncio.Future[None]:
        if self._outcome_ready is None:
            raise RuntimeError("ConPtyProcess has not been started")
        return self._outcome_ready

    @property
    def transport(self) -> _ConPtyTransport | None:
        return self._transport

    # -- start --------------------------------------------------------------

    async def start(
        self, *args: str, cwd: str | None = None, **kwargs: Any
    ) -> None:
        """Create a pseudo console and start the child shell process.

        ``args[0]`` is the shell executable.  The only keyword argument used
        is ``cwd``; subprocess-specific kwargs are silently ignored.
        """
        executable = args[0] if args else ""
        if executable and not _resolve_windows_executable(executable):
            raise FileNotFoundError(f"{executable} was not found")

        self._loop = asyncio.get_running_loop()
        sa = _SECURITY_ATTRIBUTES()
        sa.nLength = ctypes.sizeof(_SECURITY_ATTRIBUTES)
        sa.bInheritHandle = True

        # Input pipe: we write, ConPTY reads
        hin_r, hin_w = self._make_pipe(sa)
        # Output pipe: ConPTY writes, we read
        hout_r, hout_w = self._make_pipe(sa)

        size = _COORD(120, 32)
        hpc = wintypes.HANDLE(0)
        hr = _kernel32.CreatePseudoConsole(
            ctypes.byref(size), hin_r, hout_w, 0, ctypes.byref(hpc),
        )
        if hr != 0:
            _safe_close(hin_r)
            _safe_close(hin_w)
            _safe_close(hout_r)
            _safe_close(hout_w)
            raise OSError(f"CreatePseudoConsole failed: HRESULT {hr:#010x}")

        # ConPTY now owns hin_r and hout_w — our ends are hin_w and hout_r.
        hpc_value = hpc.value
        assert hpc_value is not None
        self._hPC = hpc_value
        self._hStdinWrite = hin_w
        self._hOutputRead = hout_r

        # For CreateProcessW: STARTUPINFOEXW with PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE
        attr_list = self._make_attr_list(hpc_value)
        si = _STARTUPINFOEXW()
        si.StartupInfo.cb = ctypes.sizeof(_STARTUPINFOEXW)
        si.lpAttributeList = ctypes.cast(attr_list, wintypes.LPVOID)

        cmd_line = subprocess.list2cmdline(args)
        cmd_buffer = ctypes.create_unicode_buffer(cmd_line)
        pi = _PROCESS_INFORMATION()

        cwd_str: str | None = cwd or None
        success = _kernel32.CreateProcessW(
            None,
            cmd_buffer,
            None,
            None,
            False,  # bInheritHandles — attribute list handles the inheritance
            _EXTENDED_STARTUPINFO_PRESENT | _CREATE_NEW_CONSOLE,
            None,
            ctypes.c_wchar_p(cwd_str) if cwd_str else None,
            ctypes.byref(si),
            ctypes.byref(pi),
        )
        _kernel32.DeleteProcThreadAttributeList(attr_list)

        if not success:
            _kernel32.ClosePseudoConsole(wintypes.HANDLE(self._hPC))
            self._hPC = None
            raise OSError("CreateProcessW failed")

        # Store process/thread handles and PID
        self._hProcess = pi.hProcess  # type: ignore[union-attr]
        self._hThread = pi.hThread  # type: ignore[union-attr]
        self._child_pid = pi.dwProcessId  # type: ignore[union-attr]
        self._transport = _ConPtyTransport(self._child_pid)
        self._outcome_ready = self._loop.create_future()

        # Start the background polling task that drains output and completes
        # outcome_ready when the child exits.
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name=f"{self.task_prefix}-poll"
        )

    async def _poll_loop(self) -> None:
        """Drain output and detect child exit without a reader thread."""
        try:
            while not self._close_requested and not self._process_exited:
                await self.poll_output()
                if self._check_process_exit():
                    break
                await asyncio.sleep(_POLL_INTERVAL)
            # Drain any remaining buffered output after the child exited.
            while not self._close_requested:
                data = await self.poll_output()
                if data is None:
                    break
            self._mark_outcome_ready()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — polling must never crash the loop
            self._mark_outcome_ready()

    @staticmethod
    def _make_pipe(sa: _SECURITY_ATTRIBUTES) -> tuple[int, int]:
        r, w = wintypes.HANDLE(0), wintypes.HANDLE(0)
        if not _kernel32.CreatePipe(
            ctypes.byref(r), ctypes.byref(w), ctypes.byref(sa), 0,
        ):
            raise OSError("failed to create ConPTY pipe")
        r_value, w_value = r.value, w.value
        assert r_value is not None and w_value is not None
        return r_value, w_value

    @staticmethod
    def _make_attr_list(hpc: int) -> Any:
        sz = wintypes.DWORD(0)
        _kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(sz))
        buf = ctypes.create_string_buffer(sz.value)
        _kernel32.InitializeProcThreadAttributeList(buf, 1, 0, ctypes.byref(sz))
        hpc_handle = wintypes.HANDLE(hpc)
        if not _kernel32.UpdateProcThreadAttribute(
            buf, 0, _PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
            ctypes.byref(hpc_handle), ctypes.sizeof(wintypes.HANDLE),
            None, None,
        ):
            _kernel32.DeleteProcThreadAttributeList(buf)
            _kernel32.ClosePseudoConsole(wintypes.HANDLE(hpc))
            raise OSError("UpdateProcThreadAttribute failed")
        return buf

    # -- output polling (called from the event loop) -------------------------

    async def poll_output(self) -> bytes | None:
        """Read any available output without blocking.

        Returns ``bytes`` when data is available, ``None`` when the pipe
        is empty (or closed).
        """
        if self._hOutputRead is None:
            return None
        avail = wintypes.DWORD(0)
        total = wintypes.DWORD(0)
        peek_ok = _kernel32.PeekNamedPipe(
            wintypes.HANDLE(self._hOutputRead),
            None, 0, None, ctypes.byref(avail), ctypes.byref(total),
        )
        if not peek_ok or avail.value == 0:
            return None
        buf = ctypes.create_string_buffer(avail.value)
        nread = wintypes.DWORD(0)
        ok = _kernel32.ReadFile(
            wintypes.HANDLE(self._hOutputRead),
            buf, avail.value, ctypes.byref(nread), None,
        )
        if not ok or nread.value == 0:
            return None
        data = buf.raw[: nread.value]
        self._capture_data(data)
        return data

    def _capture_data(self, data: bytes) -> None:
        if self.output_file is None or self._close_requested:
            return
        remaining = self.max_capture_bytes - self.captured
        saved = data[:remaining]
        try:
            self.output_file.write(saved)
            self.captured += len(saved)
            self.last_output_at = time.monotonic()
            self.truncated |= len(data) > remaining
        except OSError:
            pass

    def _check_process_exit(self) -> bool:
        """Return ``True`` if the child has exited (and update returncode)."""
        if self._process_exited or self._hProcess is None:
            return self._process_exited
        result = _kernel32.WaitForSingleObject(
            wintypes.HANDLE(self._hProcess), 0,
        )
        if result == 0:  # WAIT_OBJECT_0
            self._process_exited = True
            exit_code = wintypes.DWORD(0)
            if (
                _kernel32.GetExitCodeProcess(
                    wintypes.HANDLE(self._hProcess),
                    ctypes.byref(exit_code),
                )
                and self._transport is not None
            ):
                self._transport._returncode = exit_code.value
            return True
        return False

    def _mark_outcome_ready(self) -> None:
        if self._outcome_ready is not None and not self._outcome_ready.done():
            self._outcome_ready.set_result(None)

    # -- stdin write ---------------------------------------------------------

    async def write_stdin(
        self,
        data: bytes,
        *,
        timeout: float = _STDIN_WRITE_TIMEOUT,
    ) -> bool:
        if self._hStdinWrite is None or self._close_requested:
            return False
        nwritten = wintypes.DWORD(0)
        # WriteFile can block when the ConPTY input buffer is full; run it in
        # a worker thread so the event loop is never stalled.
        try:
            async with asyncio.timeout(timeout):
                success = await asyncio.to_thread(
                    _kernel32.WriteFile,
                    wintypes.HANDLE(self._hStdinWrite),
                    data,
                    len(data),
                    ctypes.byref(nwritten),
                    None,
                )
        except (TimeoutError, OSError):
            return False
        return bool(success)

    # -- cleanup -------------------------------------------------------------

    async def aclose(self, is_windows: bool) -> bool:
        """Close the ConPTY and terminate the child.

        ``is_windows`` is ignored — ConPTY is Windows-only.
        """
        _ = is_windows
        self._close_requested = True

        # Cancel the background poll task first.
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001,S110
                pass
            self._poll_task = None

        # Close the output read pipe.
        if self._hOutputRead is not None:
            _safe_close(self._hOutputRead)
            self._hOutputRead = None

        # Close the pseudo console (terminates the child).
        if self._hPC is not None:
            _kernel32.ClosePseudoConsole(wintypes.HANDLE(self._hPC))
            self._hPC = None

        if self._hProcess is not None:
            try:
                async with asyncio.timeout(
                    _process_mod.PROCESS_CLEANUP_SECONDS
                ):
                    await asyncio.to_thread(
                        _kernel32.WaitForSingleObject,
                        wintypes.HANDLE(self._hProcess),
                        int(_process_mod.PROCESS_CLEANUP_SECONDS * 1000),
                    )
                    self._check_process_exit()
            except (TimeoutError, Exception):  # noqa: BLE001
                _kernel32.TerminateProcess(
                    wintypes.HANDLE(self._hProcess), 1,
                )
                await asyncio.to_thread(
                    _kernel32.WaitForSingleObject,
                    wintypes.HANDLE(self._hProcess), _PROCESS_TERMINATE_WAIT_MS,
                )
            finally:
                _safe_close(self._hProcess)
                self._hProcess = None
                _safe_close(self._hThread)
                self._hThread = None

        if self._hStdinWrite is not None:
            _safe_close(self._hStdinWrite)
            self._hStdinWrite = None

        self.close()
        return True

    def close(self) -> None:
        self._close_requested = True
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None
        self.output_file = None


# ── Helper ──────────────────────────────────────────────────────────────────


def _resolve_windows_executable(name: str) -> bool:
    """Check if *name* refers to an existing file or a PATH-resolvable command."""
    if os.path.isabs(name):
        return os.path.exists(name)
    import shutil

    resolved = shutil.which(name)
    return resolved is not None and os.path.exists(resolved)


__all__ = ["ConPtyProcess"]