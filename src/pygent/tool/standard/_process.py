"""Shared native shell process engine for the standard tool adapters.

The engine owns what is identical for every native shell: bounded output
capture, cleanup budgets, process-tree termination and result formatting. Shell
adapters supply the executable, argv, environment, shell name and tool identity,
so nothing here translates Shell Language or interprets commands.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ._encoding import code_page_candidates, unique_encodings

MAX_OUTPUT_BYTES = 512 * 1024
MAX_FULL_OUTPUT_BYTES = 16 * 1024 * 1024
PROCESS_KILL_GRACE_SECONDS = 1.0
PROCESS_CLEANUP_SECONDS = 2.0
OUTPUT_COPY_CHUNK_BYTES = 1024 * 1024
STDIN_WRITE_TIMEOUT_SECONDS = 2.0


def _looks_like_utf16(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:256]
    return sample.count(b"\x00") >= max(2, len(sample) // 5)


def _guess_utf16_encoding(data: bytes) -> str | None:
    if data.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be"
    sample = data[:256]
    if len(sample) < 4:
        return None
    even_nulls = sample[0::2].count(0)
    odd_nulls = sample[1::2].count(0)
    threshold = max(2, len(sample) // 20)
    if odd_nulls >= threshold and odd_nulls > even_nulls * 2:
        return "utf-16-le"
    if even_nulls >= threshold and even_nulls > odd_nulls * 2:
        return "utf-16-be"
    return None


def _decode_mixed_utf16_prefix(data: bytes) -> str | None:
    encoding = _guess_utf16_encoding(data)
    if encoding not in {"utf-16-le", "utf-16-be"}:
        return None
    newline = b"\n\x00" if encoding == "utf-16-le" else b"\x00\n"
    search_from = 0
    while True:
        newline_at = data.find(newline, search_from)
        if newline_at < 0:
            return None
        prefix_end = newline_at + len(newline)
        suffix = data[prefix_end:]
        if not suffix:
            return None
        if _guess_utf16_encoding(suffix) or _looks_like_utf16(suffix):
            search_from = prefix_end
            continue
        prefix = data[:prefix_end]
        try:
            decoded_prefix = prefix.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            decoded_prefix = prefix.decode(encoding, errors="replace")
        return decoded_prefix + decode_output(suffix, max_bytes=len(suffix))


def decode_output(data: bytes, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Decode bounded shell output, tolerating UTF-16 and host code pages."""

    if not data:
        return ""
    data = data[:max_bytes]
    mixed_output = _decode_mixed_utf16_prefix(data)
    if mixed_output is not None:
        return mixed_output
    guessed_utf16 = _guess_utf16_encoding(data)
    utf16_candidates: list[str | None] = []
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        utf16_candidates = ["utf-16", guessed_utf16]
    elif guessed_utf16:
        utf16_candidates = [guessed_utf16]
    elif _looks_like_utf16(data):
        utf16_candidates = ["utf-16", "utf-16-le", "utf-16-be"]
    for encoding in unique_encodings(
        *utf16_candidates,
        *code_page_candidates(),
    ):
        candidate = data
        if (
            encoding.lower().replace("_", "-").startswith("utf-16")
            and len(candidate) % 2
        ):
            candidate = candidate[:-1]
        try:
            return candidate.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            normalized = encoding.lower().replace("_", "-")
            if (
                normalized in {"utf-8", "utf-8-sig"}
                and exc.reason == "unexpected end of data"
                and exc.start >= len(candidate) - 4
            ):
                return candidate.decode(encoding, errors="replace")
        except LookupError:
            continue
    return data.decode("utf-8", errors="replace")


def read_limited_output(
    output_file: Any, max_bytes: int = MAX_OUTPUT_BYTES
) -> tuple[bytes, bool]:
    output_file.seek(0)
    data = output_file.read(max_bytes + 1)
    return data[:max_bytes], len(data) > max_bytes


def save_full_output(
    output_file: Any,
    cwd: str,
    *,
    prefix: str,
    pid: int | None = None,
) -> tuple[str | None, str | None]:
    timestamp_ms = int(time.time() * 1000)
    pid_part = f"_{pid}" if pid is not None else ""
    for attempt in range(100):
        attempt_part = f"_{attempt}" if attempt else ""
        path = Path(cwd) / (f".{prefix}_{timestamp_ms}{pid_part}{attempt_part}.log")
        try:
            output_file.seek(0)
            with path.open("xb") as saved:
                shutil.copyfileobj(output_file, saved, length=OUTPUT_COPY_CHUNK_BYTES)
            return str(path.resolve()), None
        except FileExistsError:
            continue
        except OSError as exc:
            return None, str(exc)
    return None, "could not allocate a unique output file name"


def format_result(
    exit_code: int | str,
    output: str,
    *,
    max_output_bytes: int,
    max_capture_bytes: int,
    truncated: bool = False,
    full_output_path: str | None = None,
    full_output_error: str | None = None,
    capture_truncated: bool = False,
    cleanup_complete: bool = True,
) -> str:
    result = f"exit_code: {exit_code}\noutput:\n{output}"
    notices = []
    if not cleanup_complete:
        notices.append(
            "process cleanup incomplete; output capture closed; descendants may still be running"
        )
    if truncated:
        notices.append(f"output truncated to the first {max_output_bytes} bytes")
        if full_output_path:
            notices.append(f"full output saved to: {full_output_path}")
        elif full_output_error:
            notices.append(f"failed to save full output: {full_output_error}")
    if capture_truncated:
        notices.append(
            f"captured output capped at {max_capture_bytes} bytes while the process stream was drained"
        )
    if notices:
        if output and not output.endswith("\n"):
            result += "\n"
        result += "".join(f"[{notice}]\n" for notice in notices).rstrip("\n")
    return result


async def terminate_windows_process_tree(
    pid: int, terminator: ShellProcess
) -> bool:
    """Run taskkill within the caller's cleanup deadline and own its transport."""

    try:
        await terminator.start(
            "taskkill",
            "/PID",
            str(pid),
            "/T",
            "/F",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        await asyncio.shield(terminator.outcome_ready)
        assert terminator.transport is not None
        return terminator.transport.get_returncode() == 0
    finally:
        terminator.close()


class ShellProcess(asyncio.SubprocessProtocol):
    """Own a process transport and bounded capture until exit AND pipe EOF.

    Capturing through the public protocol API lets cleanup close the read end
    without waiting for inherited write handles or leaving a reader task behind.
    """

    def __init__(
        self,
        output_file: Any = None,
        *,
        max_capture_bytes: int = MAX_FULL_OUTPUT_BYTES,
        task_prefix: str = "pygent-shell",
    ) -> None:
        self.transport: asyncio.SubprocessTransport | None = None
        self.output_file = output_file
        self.outcome_ready: asyncio.Future[None] = (
            asyncio.get_running_loop().create_future()
        )
        self.connection_closed = False
        self.captured = 0
        self.truncated = False
        self.error: Exception | None = None
        self.close_requested = False
        self.max_capture_bytes = max_capture_bytes
        self.task_prefix = task_prefix
        self._write_ready = asyncio.Event()
        self._write_ready.set()
        self.startup: (
            asyncio.Task[tuple[asyncio.SubprocessTransport, ShellProcess]] | None
        ) = None

    async def start(self, *args: str, **kwargs: Any) -> None:
        self.startup = asyncio.create_task(
            asyncio.get_running_loop().subprocess_exec(lambda: self, *args, **kwargs),
            name=f"{self.task_prefix}-startup",
        )
        # A late startup failure must be observed even after the caller's deadline.
        self.startup.add_done_callback(
            lambda task: None if task.cancelled() else task.exception()
        )
        await asyncio.shield(self.startup)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.SubprocessTransport, transport)
        if self.close_requested:
            self.transport.close()

    def pipe_data_received(self, fd: int, data: bytes) -> None:
        if self.output_file is None:
            return
        remaining = self.max_capture_bytes - self.captured
        try:
            saved = data[:remaining]
            self.output_file.write(saved)
            self.captured += len(saved)
            self.truncated |= len(data) > remaining
        except OSError as exc:
            self._capture_failed(exc)

    def pipe_connection_lost(self, fd: int, exc: Exception | None) -> None:
        if exc is not None:
            self._capture_failed(exc)

    def pause_writing(self) -> None:
        # The child stopped reading its stdin; do not buffer without bound.
        self._write_ready.clear()

    def resume_writing(self) -> None:
        self._write_ready.set()

    def stdin_pipe(self) -> Any:
        transport = self.transport
        if transport is None:
            return None
        return transport.get_pipe_transport(0)

    async def write_stdin(
        self,
        data: bytes,
        *,
        timeout: float = STDIN_WRITE_TIMEOUT_SECONDS,
    ) -> bool:
        """Write to the child's stdin; False when the pipe is closed or the write stalls."""

        pipe = self.stdin_pipe()
        if pipe is None or self.close_requested or pipe.is_closing():
            return False
        try:
            pipe.write(data)
        except (OSError, RuntimeError):
            return False
        try:
            async with asyncio.timeout(timeout):
                while not self._write_ready.is_set():
                    await self._write_ready.wait()
        except TimeoutError:
            return False
        return not self.close_requested and not pipe.is_closing()

    def _capture_failed(self, exc: Exception) -> None:
        self.error = exc
        self.output_file = None
        if not self.outcome_ready.done():
            self.outcome_ready.set_result(None)

    def connection_lost(self, exc: Exception | None) -> None:
        self.connection_closed = True
        if exc is not None and self.error is None:
            self.error = exc
        if not self.outcome_ready.done():
            self.outcome_ready.set_result(None)

    def close(self) -> None:
        # Disable writes before the caller closes its temporary output file.
        self.close_requested = True
        self.output_file = None
        if self.transport is not None:
            self.transport.close()
            if self.startup is not None and not self.startup.done():
                self.startup.cancel()

    async def aclose(self, is_windows: bool) -> bool:
        """Spend one cleanup budget, then close local handles unconditionally."""
        try:
            async with asyncio.timeout(PROCESS_CLEANUP_SECONDS):
                if self.transport is None and self.startup is not None:
                    # Let pipe connection finish without cancelling asyncio's
                    # own startup waiter, which may otherwise wait for EOF.
                    await asyncio.shield(self.startup)
                if self.connection_closed or self.transport is None:
                    return True
                terminated = await self._terminate_tree(is_windows)
                await asyncio.shield(self.outcome_ready)
                return terminated and self.connection_closed
        except (TimeoutError, OSError):
            return False
        finally:
            self.close()

    async def _terminate_tree(self, is_windows: bool) -> bool:
        assert self.transport is not None
        pid = self.transport.get_pid()
        if is_windows:
            # taskkill cannot reliably identify descendants of an exited parent.
            # Do not target a potentially reused PID; report incomplete cleanup.
            if self.transport.get_returncode() is not None:
                return False
            return await terminate_windows_process_tree(pid, type(self)(None))

        # A POSIX process group can outlive its leader. Always signal the group.
        kill_process_group = os.killpg  # type: ignore[attr-defined]
        try:
            kill_process_group(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        try:
            async with asyncio.timeout(PROCESS_KILL_GRACE_SECONDS):
                await asyncio.shield(self.outcome_ready)
        except TimeoutError:
            pass
        # Remaining descendants may have closed stdout but ignored SIGTERM.
        try:
            kill_process_group(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass
        return True


async def run_shell_command(
    process: ShellProcess,
    output_file: Any,
    *,
    argv: Sequence[str],
    process_kwargs: dict[str, Any],
    cwd: str,
    shell_name: str,
    is_windows: bool,
    max_output_bytes: int,
    max_capture_bytes: int,
    output_prefix: str,
) -> str:
    """Run one shell command to completion and return the bounded projection."""

    from pygent.tool.executors import ToolExecutionError, current_tool_execution

    context = current_tool_execution()
    published_bytes = -1
    cancelled = False
    try:
        try:
            await process.start(*argv, **process_kwargs)
        except FileNotFoundError as exc:
            raise ToolExecutionError(
                f"{shell_name} executable was not found",
                kind="process_error",
                code="executable_not_found",
                side_effect_committed=False,
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                "command could not be started",
                kind="process_error",
                code="process_start_failed",
                side_effect_committed=False,
            ) from exc
        while not process.outcome_ready.done():
            done, _ = await asyncio.wait({process.outcome_ready}, timeout=0.05)
            if (
                context is not None
                and context.publish_output is not None
                and process.captured != published_bytes
            ):
                position = output_file.tell()
                data, _ = read_limited_output(output_file, max_output_bytes)
                output_file.seek(position)
                await context.publish_output(decode_output(data, max_output_bytes))
                published_bytes = process.captured
            if done:
                break
    except asyncio.CancelledError:
        cancelled = True
    finally:
        cleanup = asyncio.create_task(
            process.aclose(is_windows),
            name=f"pygent-{shell_name}-cleanup",
        )
        # Repeated caller cancellation must neither restart the budget
        # nor abandon the task that owns the transport and output file.
        while True:
            try:
                cleanup_complete = await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                if cleanup.cancelled():
                    raise
                cancelled = True
        if cancelled:
            if context is not None and context.publish_output is not None:
                data, _ = read_limited_output(output_file, max_output_bytes)
                await context.publish_output(decode_output(data, max_output_bytes))
            raise asyncio.CancelledError

    if process.error is not None:
        raise process.error
    transport = process.transport
    output_file.flush()
    data, truncated = read_limited_output(output_file, max_output_bytes)
    full_path = None
    full_error = None
    if truncated:
        full_path, full_error = save_full_output(
            output_file,
            cwd,
            prefix=output_prefix,
            pid=transport.get_pid() if transport else None,
        )
    returncode = transport.get_returncode() if transport else None
    exit_code: int | str = returncode if returncode is not None else -1
    formatted = format_result(
        exit_code,
        decode_output(data, max_output_bytes),
        max_output_bytes=max_output_bytes,
        max_capture_bytes=max_capture_bytes,
        truncated=truncated,
        full_output_path=full_path,
        full_output_error=full_error,
        capture_truncated=process.truncated,
        cleanup_complete=cleanup_complete,
    )
    if context is not None and context.publish_output is not None:
        await context.publish_output(formatted)
    return formatted


__all__ = [
    "MAX_FULL_OUTPUT_BYTES",
    "MAX_OUTPUT_BYTES",
    "OUTPUT_COPY_CHUNK_BYTES",
    "PROCESS_CLEANUP_SECONDS",
    "PROCESS_KILL_GRACE_SECONDS",
    "ShellProcess",
    "decode_output",
    "format_result",
    "read_limited_output",
    "run_shell_command",
    "save_full_output",
    "terminate_windows_process_tree",
]
