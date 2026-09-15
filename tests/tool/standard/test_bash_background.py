from __future__ import annotations

import asyncio
import json
import sys

import pytest

from pygent.tool.standard import BashTools


class PythonTools(BashTools):
    def _command_args(self, command):
        return [sys.executable, "-c", command]


@pytest.mark.asyncio
async def test_call_timeout_overrides_default_without_changing_next_call(tmp_path):
    from pygent.tool import ToolTaskHandle

    async with PythonTools(workspace_root=tmp_path, timeout=0) as tools:
        result = await tools.bash("print('done')", timeout=5)
        assert isinstance(result, str) and "done" in result
        handle = await tools.bash("print('next')")
        assert isinstance(handle, ToolTaskHandle)
        assert (await handle.result()).status == "succeeded"
        immediate = await tools.bash("print('zero')", timeout=0)
        assert isinstance(immediate, ToolTaskHandle)
        assert (await immediate.result()).status == "succeeded"
        background = await tools.bash("print('background')", timeout=5, is_background=True)
        assert isinstance(background, ToolTaskHandle)
        assert (await background.result()).status == "succeeded"
        fallback = await tools.bash("print('fallback')", timeout=None)
        assert isinstance(fallback, ToolTaskHandle)
        assert (await fallback.result()).status == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [-1, True, float("nan"), float("inf"), "2"])
async def test_invalid_call_timeout_rejected_before_process_start(tmp_path, timeout):
    async with PythonTools(workspace_root=tmp_path) as tools:
        with pytest.raises(ValueError):
            await tools.bash("open('started', 'w').close()", timeout=timeout)
        assert not (tmp_path / "started").exists()


@pytest.mark.asyncio
async def test_native_timeout_keeps_one_process_and_captured_output(tmp_path):
    tools = PythonTools(workspace_root=tmp_path, timeout=0.15)
    try:
        handle = await tools.bash(
            "import pathlib,time; pathlib.Path('count').open('a').write('x'); "
            "print('before',flush=True); time.sleep(.5); print('after')"
        )
        assert handle.task_id
        snapshot = await tools.tool_task_get(handle.task_id)
        for _ in range(100):
            if "before" in (snapshot["output"] or ""):
                break
            await asyncio.sleep(0.02)
            snapshot = await tools.tool_task_get(handle.task_id)
        json.dumps(snapshot)
        assert "before" in snapshot["output"]
        result = await handle.result()
        assert result.status == "succeeded"
        assert "after" in result.output
        assert (tmp_path / "count").read_text() == "x"
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_background_stop_uses_same_manager(tmp_path):
    tools = PythonTools(workspace_root=tmp_path)
    try:
        handle = await tools.bash(
            "import time; print('ready',flush=True); time.sleep(30)", is_background=True
        )
        await asyncio.sleep(0.15)
        stopped = await tools.tool_task_stop(handle.task_id)
        assert stopped["cancel_requested"] is True
        assert (await handle.result()).status in {"cancelled", "unknown"}
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_cancelling_native_observer_does_not_stop_task(tmp_path):
    tools = PythonTools(workspace_root=tmp_path)
    try:
        handle = await tools.bash(
            "import time; time.sleep(.3); print('done')", is_background=True
        )
        observer = asyncio.create_task(handle.result())
        await asyncio.sleep(0.05)
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        assert (await handle.result()).status == "succeeded"
    finally:
        await tools.aclose()


@pytest.mark.asyncio
async def test_close_cleans_owned_tasks_but_preserves_shared_manager(tmp_path):
    from pygent.tool.executors import ExecutorRegistry, InMemoryToolTaskManager

    owner = PythonTools(workspace_root=tmp_path)
    handle = await owner.bash("import time; time.sleep(30)", is_background=True)
    await asyncio.sleep(0.1)
    await owner.aclose()
    assert (await handle.result()).status in {"cancelled", "unknown"}

    manager = InMemoryToolTaskManager(ExecutorRegistry())
    try:
        shared = PythonTools(workspace_root=tmp_path, task_manager=manager)
        handle = await shared.bash(
            "import time; time.sleep(.3); print('done')", is_background=True
        )
        await shared.aclose()
        assert (await handle.result()).status == "succeeded"
    finally:
        await manager.close(cancel=True)


@pytest.mark.asyncio
async def test_short_native_result_and_declared_failure(tmp_path):
    from pygent.tool import ToolExecutionError

    async with PythonTools(workspace_root=tmp_path) as tools:
        assert "hello" in await tools.bash("print('hello')")
        with pytest.raises(ToolExecutionError) as raised:
            await tools.bash("print('hello')", working_directory="missing")
        assert raised.value.code == "not_a_directory"


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True])
def test_invalid_native_wait_configuration(tmp_path, timeout):
    with pytest.raises(ValueError):
        PythonTools(workspace_root=tmp_path, timeout=timeout)
