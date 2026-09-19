"""PTY-backed interactive process for POSIX interactive terminal sessions.

Provides ``PtyProcess`` with the same duck interface as ``ShellProcess``
so that ``_terminal.py`` can select either backend without changing its
control loop.  The process uses ``pty.openpty()`` + ``os.fork()`` and
reads the PTY master through ``loop.add_reader`` so no background thread
or SIGCHLD handler is needed.
"""

from __future__ import annotations

import asyncio
import os
import signal
import time
from dataclasses import dataclass
from typing import Any

from . import _process as _process_mod

_STDIN_WRITE_TIMEOUT = 2.0
# Bounded read chunk for the PTY master; 64 KiB keeps the event-loop callback
# short while draining interactive output promptly.
_PTY_READ_CHUNK_BYTES = 64 * 1024
# Retry/poll intervals for non-blocking writes and process reaping.
_PTY_WRITE_RETRY_SECONDS = 0.02
_PTY_REAP_POLL_SECONDS = 0.05
# Bounded wait after SIGKILL before declaring cleanup incomplete.
_PTY_KILL_WAIT_SECONDS = 0.5
# Conventional exec failure exit code when the shell cannot be started.
_EXIT_CODE_EXEC_FAILED = 127


@dataclass(slots=True)
class _PtyTransport:
    """Minimal transport exposing ``get_pid()`` and ``get_returncode()``,
    matching the subset of ``asyncio.SubprocessTransport`` that the terminal
    control loop uses."""

    _pid: int
    _returncode: int | None = None

    def get_pid(self) -> int:
        return self._pid

    def get_returncode(self) -> int | None:
        return self._returncode


class PtyProcess:
    """POSIX PTY-backed interactive process.

    The duck interface matches ``ShellProcess``:

    * ``start(*args, cwd=..., **kwargs)`` — fork + exec with PTY.
    * ``outcome_ready`` — future completed when the child exits.
    * ``write_stdin(data, *, timeout)`` — write to the PTY master.
    * ``captured`` / ``truncated`` / ``last_output_at`` — output capture state.
    * ``transport`` — ``_PtyTransport`` with ``get_pid()`` / ``get_returncode()``.
    * ``aclose(is_windows)`` — stop the child and release resources.
    * ``close()`` — release local resources immediately.
    """

    def __init__(
        self,
        output_file: Any = None,
        *,
        max_capture_bytes: int = _process_mod.MAX_FULL_OUTPUT_BYTES,
        task_prefix: str = "pygent-pty",
        workspace_root: str | None = None,
    ) -> None:
        self.master_fd: int | None = None
        self.child_pid: int | None = None
        self.output_file = output_file
        self.captured: int = 0
        self.last_output_at: float = time.monotonic()
        self.truncated: bool = False
        self.max_capture_bytes = max_capture_bytes
        self.task_prefix = task_prefix
        self.workspace_root = workspace_root
        self._outcome_ready: asyncio.Future[None] | None = None
        self._transport: _PtyTransport | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._close_requested: bool = False
        self._reader_active: bool = False
        self._capture_error: Exception | None = None
        self._reap_done: bool = False
        # True only when the child confirmed (through the startup pipe) that it
        # applied Landlock confinement; the owner uses this to decide whether
        # the "workspace" profile is honestly claimed.
        self.confinement_applied: bool = False

    # -- read-only properties ------------------------------------------------

    @property
    def outcome_ready(self) -> asyncio.Future[None]:
        if self._outcome_ready is None:
            raise RuntimeError("PtyProcess has not been started")
        return self._outcome_ready

    @property
    def transport(self) -> _PtyTransport | None:
        return self._transport

    # -- start ---------------------------------------------------------------

    async def start(
        self, *args: str, cwd: str | None = None, **kwargs: Any
    ) -> None:
        """Fork a child with a PTY controlling terminal.

        ``args[0]`` is the shell executable.  The only keyword argument used
        is ``cwd``; subprocess-specific kwargs (``stdin``, ``stdout``,
        ``stderr``, ``creationflags``, ``start_new_session``) are accepted
        for compatibility with ``_process_kwargs`` and silently ignored.
        """
        import fcntl  # type: ignore[attr-defined]
        import pty  # type: ignore[attr-defined]
        import termios  # type: ignore[attr-defined]

        executable = self._resolve_executable(args[0] if args else "")
        if not os.path.exists(executable):
            raise FileNotFoundError(
                f"{executable} was not found"
            )

        master_fd, slave_fd = pty.openpty()  # type: ignore[attr-defined]
        # Non-blocking master so the reader callback never stalls the loop.
        flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)  # type: ignore[attr-defined]
        fcntl.fcntl(  # type: ignore[attr-defined]
            master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK  # type: ignore[attr-defined]
        )

        self._loop = asyncio.get_running_loop()
        workspace_root = self.workspace_root
        # The child reports whether Landlock confinement was really applied
        # before exec, so the owner only claims the "workspace" profile when
        # the running process tree is actually confined.
        confirmation_r, confirmation_w = os.pipe()
        try:
            pid = os.fork()  # type: ignore[attr-defined]
        except OSError:
            # Do not leak the PTY descriptors when fork fails.
            try:
                os.close(master_fd)
            finally:
                os.close(slave_fd)
            os.close(confirmation_r)
            os.close(confirmation_w)
            raise
        if pid == 0:  # child
            os.close(confirmation_r)
            try:
                os.close(master_fd)
                os.setsid()  # type: ignore[attr-defined]
                fcntl.ioctl(slave_fd, termios.TIOCSCTTY, 0)  # type: ignore[attr-defined]
                os.dup2(slave_fd, 0)
                os.dup2(slave_fd, 1)
                os.dup2(slave_fd, 2)
                if slave_fd > 2:
                    os.close(slave_fd)
                if cwd:
                    os.chdir(cwd)
                confined = True
                if workspace_root is not None:
                    from ._confinement import confine_workspace

                    confined = confine_workspace(workspace_root)
                os.write(confirmation_w, b"1" if confined else b"0")
                os.close(confirmation_w)
                os.execvp(args[0], args)
            except BaseException:  # noqa: BLE001 — never propagate
                os._exit(_EXIT_CODE_EXEC_FAILED)

        # parent
        self.child_pid = pid
        self.master_fd = master_fd
        os.close(slave_fd)
        os.close(confirmation_w)
        try:
            result = os.read(confirmation_r, 1)
        except OSError:
            result = b""
        finally:
            os.close(confirmation_r)
        # EOF (empty) means the child died before reporting; both count as
        # "not confirmed" so a claimed workspace profile is never a guess.
        self.confinement_applied = result == b"1"
        self._transport = _PtyTransport(pid)
        self._outcome_ready = self._loop.create_future()
        self._loop.add_reader(master_fd, self._reader_callback)
        self._reader_active = True

    @staticmethod
    def _resolve_executable(name: str) -> str:
        import shutil

        if os.path.isabs(name):
            return name
        resolved = shutil.which(name)
        return resolved if resolved else name

    # -- reader callback (called from the event loop) ------------------------

    def _reader_callback(self) -> None:
        if self._close_requested or self.master_fd is None:
            return
        try:
            data = os.read(self.master_fd, _PTY_READ_CHUNK_BYTES)
        except BlockingIOError:
            return
        except OSError:
            self._on_child_eof()
            return
        if not data:
            self._on_child_eof()
            return
        self._capture_data(data)

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
        except OSError as exc:
            self._on_capture_error(exc)

    def _on_child_eof(self) -> None:
        """PTY master returned EOF — the slave side is closed (child exited)."""
        self._deactivate_reader()
        self._reap_child()
        if self._outcome_ready is not None and not self._outcome_ready.done():
            self._outcome_ready.set_result(None)

    def _on_capture_error(self, exc: Exception) -> None:
        self._capture_error = exc
        self.output_file = None
        if self._outcome_ready is not None and not self._outcome_ready.done():
            self._outcome_ready.set_result(None)

    def _deactivate_reader(self) -> None:
        if self._reader_active and self.master_fd is not None and self._loop is not None:
            try:
                self._loop.remove_reader(self.master_fd)
            except (ValueError, RuntimeError):
                pass
            self._reader_active = False

    def _reap_child(self) -> None:
        if self.child_pid is None or self._reap_done:
            return
        try:
            wpid, status = os.waitpid(  # type: ignore[attr-defined]
                self.child_pid, os.WNOHANG  # type: ignore[attr-defined]
            )
            if wpid == self.child_pid:
                self._update_returncode(status)
                self._reap_done = True
        except ChildProcessError:
            self._set_returncode(-1)
            self._reap_done = True

    # -- write stdin ---------------------------------------------------------

    async def write_stdin(
        self,
        data: bytes,
        *,
        timeout: float = _STDIN_WRITE_TIMEOUT,
    ) -> bool:
        if self.master_fd is None or self._close_requested:
            return False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        view = memoryview(data)
        sent = 0
        while sent < len(view):
            try:
                written = os.write(self.master_fd, view[sent:])
                sent += written
            except BlockingIOError:
                if loop.time() >= deadline:
                    return False
                await asyncio.sleep(_PTY_WRITE_RETRY_SECONDS)
            except OSError:
                return False
        return True

    # -- cleanup -------------------------------------------------------------

    async def aclose(self, is_windows: bool) -> bool:
        """Close the PTY, terminate the child, and wait for it to exit.

        ``is_windows`` is ignored (PTY is POSIX-only).

        Close the master fd first.  On POSIX this sends SIGHUP to the session
        leader (the shell), which forwards it to its process group.  If the
        child does not exit within the cleanup budget, escalate to SIGTERM
        and then SIGKILL.
        """
        _ = is_windows  # PTY is POSIX-only.

        master_fd = self.master_fd
        if master_fd is not None:
            self._deactivate_reader()
            try:
                os.close(master_fd)
            except OSError:
                pass
            self.master_fd = None

        if self.child_pid is None:
            self.close()
            return True

        # Already reaped by the EOF / normal-exit path.
        if self._reap_done:
            self.close()
            return True

        # Bounded wait after SIGHUP (via master close).
        if await self._try_wait(_process_mod.PROCESS_CLEANUP_SECONDS):
            self.close()
            return True

        # Escalate: SIGTERM the process group.
        try:
            os.killpg(self.child_pid, signal.SIGTERM)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError):
            pass
        if await self._try_wait(_process_mod.PROCESS_KILL_GRACE_SECONDS):
            self.close()
            return True

        # Last resort: SIGKILL.
        try:
            os.killpg(self.child_pid, signal.SIGKILL)  # type: ignore[attr-defined]
        except (ProcessLookupError, PermissionError):
            pass
        await self._try_wait(_PTY_KILL_WAIT_SECONDS)
        self.close()
        return False

    async def _try_wait(self, timeout: float) -> bool:
        assert self.child_pid is not None
        try:
            async with asyncio.timeout(timeout):
                while True:
                    wpid, status = os.waitpid(  # type: ignore[attr-defined]  # noqa: ASYNC222 — WNOHANG non-blocking
                        self.child_pid, os.WNOHANG  # type: ignore[attr-defined]
                    )
                    if wpid == self.child_pid:
                        self._update_returncode(status)
                        self._reap_done = True
                        return True
                    await asyncio.sleep(_PTY_REAP_POLL_SECONDS)
        except (TimeoutError, ChildProcessError):
            return False

    def _update_returncode(self, status: int) -> None:
        if os.WIFEXITED(status):  # type: ignore[attr-defined]
            self._set_returncode(os.WEXITSTATUS(status))  # type: ignore[attr-defined]
        elif os.WIFSIGNALED(status):  # type: ignore[attr-defined]
            self._set_returncode(-os.WTERMSIG(status))  # type: ignore[attr-defined]
        else:
            self._set_returncode(-1)

    def _set_returncode(self, code: int) -> None:
        if self._transport is not None:
            self._transport._returncode = code

    def close(self) -> None:
        self._close_requested = True
        self._deactivate_reader()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        self.output_file = None


__all__ = ["PtyProcess"]