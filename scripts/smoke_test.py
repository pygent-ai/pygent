"""
Smoke test: 在全新环境中验证 pip install 后 pygent 可正常 import 并使用。
不依赖外部 API，仅验证核心对象的创建与导入。
用法: python scripts/smoke_test.py（需在项目根目录执行，或先 pip install .）
"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    errors = []

    # 1. import pygent
    try:
        import pygent
        print("[OK] import pygent")
    except ImportError as e:
        errors.append(f"import pygent 失败: {e}")
        print(f"[FAIL] {errors[-1]}")
        return 1

    # 2. version
    try:
        ver = getattr(pygent, "__version__", None)
        if ver:
            print(f"[OK] pygent.__version__ = {ver}")
        else:
            print("[WARN] pygent.__version__ 未定义")
    except Exception as e:
        errors.append(f"访问 __version__ 失败: {e}")

    # 3. 核心子模块
    try:
        from pygent.agent import BaseAgent
        from pygent.context import BaseContext
        from pygent.message import UserMessage, AssistantMessage
        from pygent.module.tool import ToolManager
        from pygent.llm import AsyncRequestsClient
        print("[OK] 导入 agent, context, message, tool, llm")
    except ImportError as e:
        errors.append(f"导入子模块失败: {e}")
        print(f"[FAIL] {errors[-1]}")
        return 1

    # 4. 创建对象
    try:
        ctx = BaseContext(system_prompt="test")
        ctx.add_message(UserMessage(content="hello"))
        assert ctx.last_message is not None
        print("[OK] BaseContext + UserMessage")
    except Exception as e:
        errors.append(f"创建 Context/Message 失败: {e}")
        print(f"[FAIL] {errors[-1]}")
        return 1

    # 5. ToolManager
    try:
        tm = ToolManager()
        assert tm is not None
        print("[OK] ToolManager()")
    except Exception as e:
        errors.append(f"创建 ToolManager 失败: {e}")
        print(f"[FAIL] {errors[-1]}")
        return 1

    # 6. BaseAgent 子类 + 端到端运行（使用 Mock LLM，无需 API key）
    try:
        from pygent.llm import BaseAsyncClient
        from pygent.message import AssistantMessage

        class EchoClient(BaseAsyncClient):
            """Mock LLM：直接返回固定回复，不调用真实 API。用于安装验证。"""

            async def forward(self, context, **kwargs):
                msg = AssistantMessage(content="Hello from pygent!")
                context.add_message(msg)
                return msg

        class MinimalAgent(BaseAgent):
            def __init__(self):
                super().__init__()
                self.llm = EchoClient(
                    base_url="http://localhost",
                    api_key="",
                    model_name="echo",
                )

            async def forward(self, user_input: str):
                ctx = BaseContext(system_prompt="You are helpful.")
                ctx.add_message(UserMessage(content=user_input))
                await self.llm.forward(ctx)
                return ctx.last_message.content

        import asyncio

        agent = MinimalAgent()
        result = asyncio.run(agent.forward("hi"))
        content = result.data if hasattr(result, "data") else result
        assert "Hello from pygent" in str(content)
        print("[OK] Agent 端到端运行（Mock LLM）")
    except Exception as e:
        errors.append(f"Agent 端到端运行失败: {e}")
        print(f"[FAIL] {errors[-1]}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n[PASS] 所有 smoke test 通过（含 Agent 端到端）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
