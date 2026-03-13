"""测试 BaseMessageChunk 及 BaseMessage + BaseMessageChunk 合并逻辑"""
import pytest
from pygent.message import (
    BaseMessage,
    BaseMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    UserMessage,
    UserMessageChunk,
    AssistantMessage,
    AssistantMessageChunk,
    ToolMessage,
    ToolMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    ToolCallChunk,
)


# ==================== BaseMessage + BaseMessageChunk ====================


class TestBaseMessagePlusChunk:
    """BaseMessage + BaseMessageChunk => BaseMessage"""

    def test_assistant_message_plus_chunks(self):
        msg = AssistantMessage("")
        chunk1 = AssistantMessageChunk("Hello")
        chunk2 = AssistantMessageChunk(" World")
        result = msg + chunk1 + chunk2
        assert isinstance(result, AssistantMessage)
        assert result.content.data == "Hello World"

    def test_system_message_plus_chunk(self):
        msg = SystemMessage("")
        chunk = SystemMessageChunk("System prompt")
        result = msg + chunk
        assert isinstance(result, SystemMessage)
        assert result.content.data == "System prompt"

    def test_user_message_plus_chunk(self):
        msg = UserMessage("")
        chunk = UserMessageChunk("User input")
        result = msg + chunk
        assert isinstance(result, UserMessage)
        assert result.content.data == "User input"


class TestChunkPlusChunk:
    """BaseMessageChunk + BaseMessageChunk => BaseMessageChunk"""

    def test_assistant_chunk_plus_chunk(self):
        c1 = AssistantMessageChunk("Hi")
        c2 = AssistantMessageChunk(" there")
        merged = c1 + c2
        assert merged.content.data == "Hi there"

    def test_accumulate_chunks_then_add_to_message(self):
        acc = AssistantMessageChunk("")
        for s in ["A", "B", "C"]:
            acc = acc + AssistantMessageChunk(s)
        final = AssistantMessage("") + acc
        assert final.content.data == "ABC"

    def test_base_chunk_plus_specific_chunk_delegates(self):
        """BaseMessageChunk + AssistantMessageChunk 应委托给 AssistantMessageChunk"""
        base = BaseMessageChunk("assistant", "left")
        spec = AssistantMessageChunk("right")
        merged = base + spec
        assert merged.content.data == "leftright"
        assert isinstance(merged, AssistantMessageChunk)


class TestToolCallChunk:
    def test_merge_tool_call_chunks(self):
        tc1 = ToolCallChunk(index=0, tool_name="get_", arguments='{"a":')
        tc2 = ToolCallChunk(
            index=0, tool_call_id="call_1", tool_name="weather", arguments='1}'
        )
        tc = tc1 + tc2
        full = tc.to_tool_call()
        assert full.tool_call_id.data == "call_1"
        assert full.tool_name.data == "get_weather"
        assert "a" in full.arguments.data

    def test_tool_call_chunk_index_mismatch_raises(self):
        tc1 = ToolCallChunk(index=0, tool_name="foo")
        tc2 = ToolCallChunk(index=1, tool_name="bar")
        with pytest.raises(ValueError, match="index 不匹配"):
            tc1 + tc2


class TestAssistantMessageChunkWithToolCalls:
    def test_chunk_with_tool_call_chunks_merges_to_message(self):
        ach = AssistantMessageChunk(
            "",
            tool_call_chunks=[
                ToolCallChunk(0, tool_call_id="id1", tool_name="foo", arguments="{}")
            ],
        )
        am = AssistantMessage("") + ach
        assert len(am.tool_calls.data) == 1
        assert am.tool_calls.data[0].tool_name.data == "foo"
        assert am.tool_calls.data[0].tool_call_id.data == "id1"

    def test_streaming_tool_call_merge(self):
        """模拟流式工具调用：多个 chunk 合并"""
        c1 = AssistantMessageChunk(
            "", tool_call_chunks=[ToolCallChunk(0, tool_call_id="id1", tool_name="get_")]
        )
        c2 = AssistantMessageChunk(
            "", tool_call_chunks=[ToolCallChunk(0, tool_name="weather", arguments='{"')]
        )
        c3 = AssistantMessageChunk(
            "", tool_call_chunks=[ToolCallChunk(0, arguments='city":"Beijing"}')]
        )
        merged = c1 + c2 + c3
        am = AssistantMessage("") + merged
        assert len(am.tool_calls.data) == 1
        tc = am.tool_calls.data[0]
        assert tc.tool_call_id.data == "id1"
        assert tc.tool_name.data == "get_weather"
        assert tc.arguments.data.get("city") == "Beijing"


class TestToolMessageChunk:
    def test_tool_message_plus_chunk(self):
        msg = ToolMessage("partial", tool_call_id="call_123")
        chunk = ToolMessageChunk(" result")
        result = msg + chunk
        assert isinstance(result, ToolMessage)
        assert result.content.data == "partial result"
        assert result.tool_call_id.data == "call_123"


class TestFunctionMessageChunk:
    def test_function_message_plus_chunk(self):
        msg = FunctionMessage("", name="my_func")
        chunk = FunctionMessageChunk('{"result": 1}')
        result = msg + chunk
        assert isinstance(result, FunctionMessage)
        assert result.content.data == '{"result": 1}'
        assert result.name.data == "my_func"

    def test_function_chunk_plus_chunk(self):
        c1 = FunctionMessageChunk("part1", name="fn")
        c2 = FunctionMessageChunk("part2")
        merged = c1 + c2
        assert merged.content.data == "part1part2"
        assert merged.name.data == "fn"

    def test_function_chunk_name_mismatch_raises(self):
        c1 = FunctionMessageChunk("", name="func_a")
        c2 = FunctionMessageChunk("", name="func_b")
        with pytest.raises(ValueError, match="name 不匹配"):
            c1 + c2


# ==================== 边界与异常 ====================


class TestEdgeCases:
    """边界情况与异常行为"""

    def test_base_message_plus_base_message_raises(self):
        """BaseMessage + BaseMessage 应抛出 TypeError"""
        m1 = BaseMessage("user", "hi")
        m2 = BaseMessage("assistant", "hello")
        with pytest.raises(TypeError, match="BaseMessage \\+ BaseMessage"):
            m1 + m2

    def test_chunk_plus_message_returns_message(self):
        """BaseMessageChunk + BaseMessage 通过 __radd__ 应返回 BaseMessage"""
        chunk = AssistantMessageChunk("world")
        msg = AssistantMessage("Hello ")
        result = chunk + msg
        assert isinstance(result, AssistantMessage)
        assert result.content.data == "Hello world"

    def test_chunk_role_mismatch_raises(self):
        """不同 role 的 chunk 合并应抛出 ValueError"""
        c1 = AssistantMessageChunk("a")
        c2 = UserMessageChunk("b")
        with pytest.raises(ValueError, match="role 不匹配"):
            c1 + c2

    def test_empty_chunk_merge(self):
        """空 content 的 chunk 合并"""
        c1 = AssistantMessageChunk("")
        c2 = AssistantMessageChunk("only")
        merged = c1 + c2
        assert merged.content.data == "only"

    def test_message_with_existing_content_plus_chunk(self):
        """已有内容的 Message + Chunk"""
        msg = AssistantMessage("Hello")
        chunk = AssistantMessageChunk(", world!")
        result = msg + chunk
        assert result.content.data == "Hello, world!"


# ==================== 多工具调用 ====================


class TestMultipleToolCalls:
    """多个 tool_call 的流式合并"""

    def test_two_tool_calls_streaming(self):
        """模拟两个工具调用的流式输出"""
        chunks = [
            AssistantMessageChunk(
                "",
                tool_call_chunks=[
                    ToolCallChunk(0, tool_call_id="id1", tool_name="search", arguments='{"q":'),
                    ToolCallChunk(1, tool_call_id="id2", tool_name="calc", arguments='{"expr":'),
                ],
            ),
            AssistantMessageChunk(
                "",
                tool_call_chunks=[
                    ToolCallChunk(0, arguments='"test"}'),
                    ToolCallChunk(1, arguments='"1+1"}'),
                ],
            ),
        ]
        merged = chunks[0] + chunks[1]
        am = AssistantMessage("") + merged
        assert len(am.tool_calls.data) == 2
        assert am.tool_calls.data[0].tool_name.data == "search"
        assert am.tool_calls.data[0].arguments.data.get("q") == "test"
        assert am.tool_calls.data[1].tool_name.data == "calc"
        assert am.tool_calls.data[1].arguments.data.get("expr") == "1+1"


# ==================== ToolCallChunk 边界 ====================


class TestToolCallChunkEdgeCases:
    def test_tool_call_chunk_to_tool_call_empty_args(self):
        """空 arguments 转为 ToolCall"""
        tc = ToolCallChunk(0, tool_call_id="x", tool_name="f", arguments="")
        full = tc.to_tool_call()
        assert full.tool_call_id.data == "x"
        assert full.tool_name.data == "f"
        assert full.arguments.data == {}

    def test_tool_call_chunk_invalid_json_args(self):
        """非 JSON 的 arguments 应存入 raw_arguments"""
        tc = ToolCallChunk(0, tool_call_id="x", tool_name="f", arguments="not json")
        full = tc.to_tool_call()
        assert full.arguments.data.get("raw_arguments") == "not json"

    def test_tool_message_chunk_only_tool_call_id(self):
        """Chunk 仅有 tool_call_id 时合并到 Message"""
        msg = ToolMessage("", tool_call_id="tid_1")
        chunk = ToolMessageChunk("result")
        result = msg + chunk
        assert result.tool_call_id.data == "tid_1"
        assert result.content.data == "result"

    def test_tool_message_empty_tool_call_id_raises(self):
        """Message 与 Chunk 都无 tool_call_id 时应报错"""
        msg = ToolMessage("x", tool_call_id="")
        chunk = ToolMessageChunk("y")
        with pytest.raises(ValueError, match="必须包含 tool_call_id"):
            msg + chunk


# ==================== 流式模拟 ====================


class TestStreamingSimulation:
    """模拟真实流式输出场景"""

    def test_simulate_llm_streaming_text(self):
        """模拟 LLM 逐 token 输出文本"""
        tokens = ["你", "好", "，", "世", "界", "！"]
        acc = AssistantMessageChunk("")
        for t in tokens:
            acc = acc + AssistantMessageChunk(t)
        final = AssistantMessage("") + acc
        assert final.content.data == "你好，世界！"

    def test_simulate_streaming_with_tool_then_text(self):
        """先输出 tool_call，再输出文本"""
        c1 = AssistantMessageChunk(
            "", tool_call_chunks=[ToolCallChunk(0, tool_call_id="c1", tool_name="f", arguments="{}")]
        )
        c2 = AssistantMessageChunk("Done.")
        merged = c1 + c2
        am = AssistantMessage("") + merged
        assert len(am.tool_calls.data) == 1
        assert am.content.data == "Done."
