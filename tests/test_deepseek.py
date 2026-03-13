"""测试 DeepSeek API 的 AsyncRequestsClient"""
import asyncio
import os

from pygent.llm import AsyncRequestsClient
from pygent.context import BaseContext
from pygent.message import UserMessage


async def test_deepseek():
    base_url = os.environ.get("DS_BASE_URL", "https://api.deepseek.com")
    api_key = os.environ.get("DS_API_KEY", "")
    model = os.environ.get("DS_MODEL_NAME", "deepseek-chat")

    if not api_key:
        print("请设置环境变量 DS_API_KEY")
        return

    ctx = BaseContext(system_prompt="You are a helpful assistant. Reply briefly.")
    ctx.add_message(UserMessage("Say hello in one sentence."))

    client = AsyncRequestsClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        timeout=30,
    )

    print("Testing non-stream...")
    try:
        msg = await client.forward(ctx)
        print("Response:", msg.content.data[:300] if msg.content.data else "(empty)")
        print("Non-stream OK")
    except Exception as e:
        print("Non-stream Error:", type(e).__name__, str(e))
        return

    ctx2 = BaseContext(system_prompt="Reply briefly.")
    ctx2.add_message(UserMessage("Count from 1 to 3."))
    client2 = AsyncRequestsClient(
        base_url=base_url,
        api_key=api_key,
        model_name=model,
        stream=True,
        timeout=30,
    )

    print("\nTesting stream...")
    chunks = []

    def on_chunk(c):
        chunks.append(c)
        print(c.content.data, end="", flush=True)

    try:
        msg2 = await client2.forward(ctx2, on_chunk=on_chunk)
        print()
        print("Stream OK, total chunks:", len(chunks))
    except Exception as e:
        print("\nStream Error:", type(e).__name__, str(e))


if __name__ == "__main__":
    asyncio.run(test_deepseek())
