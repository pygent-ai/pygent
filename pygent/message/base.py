from typing import Any, Dict, List, Optional, Union, Type, get_type_hints, Iterator
from abc import ABC, abstractmethod
from datetime import datetime
import json
import inspect
import hashlib

from pygent.common import PygentData, PygentOperator, PygentString, PygentDict, PygentList, PygentBool, PygentInt, PygentFloat


# ==================== 消息相关类 ====================

class MessageRole(PygentString):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class BaseMessage(PygentData):
    """基础消息类"""
    role: PygentString
    content: PygentString
    name: Optional[PygentString] = None
    metadata: Optional[PygentDict] = None
    
    def __init__(
        self,
        role: Union[str, PygentString],
        content: Union[str, PygentString],
        name: Optional[Union[str, PygentString]] = None,
        metadata: Optional[Union[Dict, PygentDict]] = None,
        **kwargs
    ):
        super().__init__({})
        
        # 初始化角色
        if isinstance(role, PygentString):
            self.role = role
        else:
            self.role = PygentString(str(role))
        
        # 初始化内容
        if isinstance(content, PygentString):
            self.content = content
        else:
            self.content = PygentString(str(content))
        
        # 初始化名称（可选）
        if name is not None:
            if isinstance(name, PygentString):
                self.name = name
            else:
                self.name = PygentString(str(name))
        
        # 初始化元数据（可选）
        if metadata is not None:
            if isinstance(metadata, PygentDict):
                self.metadata = metadata
            else:
                self.metadata = PygentDict(metadata)
        
        # 处理额外参数
        for key, value in kwargs.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool, list, dict)):
                    self.data[key] = value
                elif isinstance(value, PygentData):
                    self.data[key] = value.data
                else:
                    self.data[key] = str(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（API请求格式）"""
        result = {
            "role": self.role.data,
            "content": self.content.data
        }
        
        if self.name is not None:
            result["name"] = self.name.data
        
        if self.metadata is not None:
            # 元数据通常不作为API参数发送
            pass
        
        # 添加额外字段
        for key, value in self.data.items():
            if key not in ['role', 'content', 'name', 'metadata']:
                result[key] = value
        
        return result
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为OpenAI消息格式"""
        result = self.to_dict()
        
        # OpenAI特定的格式调整
        if self.role.data == "function":
            # OpenAI函数消息格式
            result["role"] = "function"
        elif self.role.data == "tool":
            # OpenAI工具消息格式
            result["role"] = "tool"
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseMessage':
        """从字典创建消息"""
        return cls(
            role=data.get("role", ""),
            content=data.get("content", ""),
            name=data.get("name"),
            metadata=data.get("metadata")
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        name_str = f" ({self.name.data})" if self.name else ""
        content_preview = self.content.data[:50] + "..." if len(self.content.data) > 50 else self.content.data
        return f"{self.role.data}{name_str}: {content_preview}"
    
    def __repr__(self) -> str:
        return f"BaseMessage(role={repr(self.role.data)}, content={repr(self.content.data[:30])}...)"

    def __add__(self, other: Union['BaseMessage', 'BaseMessageChunk']) -> 'BaseMessage':
        """
        BaseMessage + BaseMessageChunk => BaseMessage
        用于流式输出时累积 chunk，返回合并后的完整消息。
        """
        if isinstance(other, BaseMessageChunk):
            return other.__radd__(self)
        if isinstance(other, BaseMessage):
            # BaseMessage + BaseMessage 暂不支持，可扩展为合并
            raise TypeError("BaseMessage + BaseMessage 不支持，请使用 BaseMessage + BaseMessageChunk")
        return NotImplemented


class BaseMessageChunk(PygentData):
    """
    消息的流式增量块。
    用于大模型流式输出时，content 及 tool_call_chunks 分多次返回。
    支持 BaseMessage + BaseMessageChunk => BaseMessage 及 BaseMessageChunk + BaseMessageChunk => BaseMessageChunk。
    """
    role: PygentString
    content: PygentString
    name: Optional[PygentString] = None
    metadata: Optional[PygentDict] = None

    def __init__(
        self,
        role: Union[str, PygentString],
        content: Union[str, PygentString] = "",
        name: Optional[Union[str, PygentString]] = None,
        metadata: Optional[Union[Dict, PygentDict]] = None,
        **kwargs
    ):
        super().__init__({})
        self.role = PygentString(role) if not isinstance(role, PygentString) else role
        self.content = PygentString(content) if not isinstance(content, PygentString) else content
        self.name = PygentString(name) if name is not None else None
        self.metadata = PygentDict(metadata) if metadata is not None else None
        for key, value in kwargs.items():
            if value is not None:
                self.data[key] = value

    def _merge_content(self, other: 'BaseMessageChunk') -> str:
        """合并 content（字符串拼接）"""
        c1 = self.content.data if self.content else ""
        c2 = other.content.data if other.content else ""
        return c1 + c2

    def _merge_metadata(self, other: 'BaseMessageChunk') -> Optional[PygentDict]:
        """合并 metadata（后者覆盖前者同键）"""
        d1 = dict(self.metadata.data) if self.metadata else {}
        d2 = dict(other.metadata.data) if other.metadata else {}
        if not d1 and not d2:
            return None
        merged = {**d1, **d2}
        return PygentDict(merged)

    def _to_base_message_kwargs(self) -> Dict[str, Any]:
        """提取转为 BaseMessage 所需的参数字典"""
        kwargs = {"role": self.role, "content": self.content}
        if self.name is not None:
            kwargs["name"] = self.name
        if self.metadata is not None:
            kwargs["metadata"] = self.metadata
        for key in self.data:
            if key not in ('role', 'content', 'name', 'metadata'):
                kwargs[key] = self.data[key]
        return kwargs

    def __add__(self, other: Union['BaseMessage', 'BaseMessageChunk']) -> Union['BaseMessage', 'BaseMessageChunk']:
        """BaseMessageChunk + BaseMessageChunk => BaseMessageChunk"""
        if isinstance(other, BaseMessageChunk):
            if type(other) is not BaseMessageChunk and issubclass(type(other), BaseMessageChunk):
                return other._merge_chunk(self, swap_order=True)
            return self._merge_chunk(other)
        if isinstance(other, BaseMessage):
            return other.__add__(self)  # 转为 BaseMessage + BaseMessageChunk
        return NotImplemented

    def __radd__(self, other: 'BaseMessage') -> 'BaseMessage':
        """BaseMessage + BaseMessageChunk => BaseMessage（由 BaseMessage.__add__ 调用）"""
        if not isinstance(other, BaseMessage):
            return NotImplemented
        return self._merge_into_message(other)

    def _merge_chunk(self, other: 'BaseMessageChunk', swap_order: bool = False) -> 'BaseMessageChunk':
        """合并两个 BaseMessageChunk，由子类覆盖以处理 tool_call_chunks 等。swap_order=True 表示从委托调用，left=other, right=self"""
        left, right = (other, self) if swap_order else (self, other)
        if left.role.data != right.role.data:
            raise ValueError(f"BaseMessageChunk role 不匹配: {left.role.data} != {right.role.data}")
        content = left._merge_content(right)
        name = right.name if right.name is not None else left.name
        metadata = left._merge_metadata(right)
        kwargs = {k: v for k, v in left.data.items() if k not in ('role', 'content', 'name', 'metadata')}
        for k, v in right.data.items():
            if k not in ('role', 'content', 'name', 'metadata'):
                kwargs[k] = v
        return BaseMessageChunk(role=left.role, content=content, name=name, metadata=metadata, **kwargs)

    def _merge_into_message(self, msg: 'BaseMessage') -> 'BaseMessage':
        """将 chunk 合并进 BaseMessage，由子类覆盖以处理 tool_calls 等"""
        content = (msg.content.data if msg.content else "") + (self.content.data if self.content else "")
        kwargs = {"role": msg.role, "content": content}
        if msg.name is not None:
            kwargs["name"] = msg.name
        if msg.metadata is not None or self.metadata is not None:
            d1 = dict(msg.metadata.data) if msg.metadata else {}
            d2 = dict(self.metadata.data) if self.metadata else {}
            kwargs["metadata"] = PygentDict({**d1, **d2})
        for key in msg.data:
            if key not in ('role', 'content', 'name', 'metadata'):
                kwargs[key] = getattr(msg, key, msg.data[key])
        for key in self.data:
            if key not in ('role', 'content', 'name', 'metadata'):
                kwargs[key] = self.data[key]
        return msg.__class__(**kwargs)

    def __str__(self) -> str:
        name_str = f" ({self.name.data})" if self.name else ""
        content_preview = (self.content.data[:50] + "...") if len(self.content.data) > 50 else self.content.data
        return f"{self.role.data}Chunk{name_str}: {content_preview}"


class SystemMessage(BaseMessage):
    """系统消息"""
    def __init__(self, content: Union[str, PygentString], **kwargs):
        super().__init__(role=MessageRole.SYSTEM, content=content, **kwargs)


class SystemMessageChunk(BaseMessageChunk):
    """系统消息的流式块"""
    def __init__(self, content: Union[str, PygentString] = "", **kwargs):
        super().__init__(role=MessageRole.SYSTEM, content=content, **kwargs)

    def _merge_chunk(self, other: 'BaseMessageChunk', swap_order: bool = False) -> 'SystemMessageChunk':
        merged = super()._merge_chunk(other, swap_order)
        return SystemMessageChunk(content=merged.content, name=merged.name, metadata=merged.metadata)

    def _merge_into_message(self, msg: 'BaseMessage') -> 'SystemMessage':
        content = (msg.content.data if msg.content else "") + (self.content.data if self.content else "")
        return SystemMessage(content=content)


class UserMessage(BaseMessage):
    """用户消息"""
    def __init__(self, content: Union[str, PygentString], name: Optional[Union[str, PygentString]] = None, **kwargs):
        super().__init__(role=MessageRole.USER, content=content, name=name, **kwargs)


class UserMessageChunk(BaseMessageChunk):
    """用户消息的流式块"""
    def __init__(self, content: Union[str, PygentString] = "", name: Optional[Union[str, PygentString]] = None, **kwargs):
        super().__init__(role=MessageRole.USER, content=content, name=name, **kwargs)

    def _merge_chunk(self, other: 'BaseMessageChunk', swap_order: bool = False) -> 'UserMessageChunk':
        merged = super()._merge_chunk(other, swap_order)
        return UserMessageChunk(content=merged.content, name=merged.name, metadata=merged.metadata)

    def _merge_into_message(self, msg: 'BaseMessage') -> 'UserMessage':
        content = (msg.content.data if msg.content else "") + (self.content.data if self.content else "")
        name = msg.name if hasattr(msg, 'name') else None
        return UserMessage(content=content, name=name)


class AssistantMessage(BaseMessage):
    """助手消息"""
    def __init__(
        self,
        content: Union[str, PygentString],
        name: Optional[Union[str, PygentString]] = None,
        tool_calls: Optional[List['ToolCall']] = None,
        **kwargs
    ):
        super().__init__(role=MessageRole.ASSISTANT, content=content, name=name, **kwargs)
        
        # 工具调用（可选）
        if tool_calls is not None:
            self.tool_calls = PygentList(tool_calls)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        
        if hasattr(self, 'tool_calls'):
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls.data]

        return result


# ToolCall/ToolCallChunk 需在 AssistantMessageChunk 之前定义（依赖关系）
class ToolCall(PygentData):
    """工具调用"""
    tool_call_id: PygentString
    tool_name: PygentString
    arguments: PygentDict

    def __init__(
        self,
        tool_call_id: Union[str, PygentString],
        tool_name: Union[str, PygentString],
        arguments: Union[Dict, str, PygentDict],
        **kwargs
    ):
        super().__init__({})
        self.tool_call_id = PygentString(tool_call_id) if not isinstance(tool_call_id, PygentString) else tool_call_id
        self.tool_name = PygentString(tool_name) if not isinstance(tool_name, PygentString) else tool_name
        if isinstance(arguments, PygentDict):
            self.arguments = arguments
        elif isinstance(arguments, str):
            try:
                self.arguments = PygentDict(json.loads(arguments))
            except json.JSONDecodeError:
                self.arguments = PygentDict({"input": arguments})
        elif isinstance(arguments, dict):
            self.arguments = PygentDict(arguments)
        else:
            self.arguments = PygentDict({"arguments": str(arguments)})
        for key, value in kwargs.items():
            if value is not None:
                self.data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.tool_call_id.data,
            "type": "function",
            "function": {
                "name": self.tool_name.data,
                "arguments": json.dumps(self.arguments.data) if self.arguments.data else "{}"
            }
        }
        for key, value in self.data.items():
            if key not in ['tool_call_id', 'tool_name', 'arguments']:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        if "id" in data and "function" in data:
            tool_call_id = data["id"]
            tool_name = data["function"]["name"]
            arguments_str = data["function"]["arguments"]
            try:
                arguments = json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError):
                arguments = {"raw_arguments": arguments_str}
            return cls(tool_call_id=tool_call_id, tool_name=tool_name, arguments=arguments)
        return cls(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {})
        )

    def __str__(self) -> str:
        args_preview = str(self.arguments.data)[:50] + "..." if len(str(self.arguments.data)) > 50 else self.arguments.data
        return f"ToolCall({self.tool_name.data}: {args_preview})"


class ToolCallChunk(PygentData):
    """
    工具调用的流式增量块。
    用于大模型流式输出时，工具调用的 id、name、arguments 可能分多次返回。
    """
    def __init__(
        self,
        index: int = 0,
        tool_call_id: Optional[Union[str, PygentString]] = None,
        tool_name: Optional[Union[str, PygentString]] = None,
        arguments: Optional[Union[str, PygentString]] = None,
        **kwargs
    ):
        super().__init__({})
        self.index = index
        self.tool_call_id = PygentString(tool_call_id) if tool_call_id is not None else None
        self.tool_name = PygentString(tool_name) if tool_name is not None else None
        self.arguments = PygentString(arguments) if arguments is not None else None
        for key, value in kwargs.items():
            if value is not None:
                self.data[key] = value

    def __add__(self, other: 'ToolCallChunk') -> 'ToolCallChunk':
        if not isinstance(other, ToolCallChunk):
            return NotImplemented
        if self.index != other.index:
            raise ValueError(f"ToolCallChunk index 不匹配: {self.index} != {other.index}")
        tool_call_id = (other.tool_call_id.data if other.tool_call_id else None) or (self.tool_call_id.data if self.tool_call_id else None) or ""
        tool_name = (self.tool_name.data if self.tool_name else "") + (other.tool_name.data if other.tool_name else "")
        args_str = (self.arguments.data if self.arguments else "") + (other.arguments.data if other.arguments else "")
        return ToolCallChunk(index=self.index, tool_call_id=tool_call_id, tool_name=tool_name, arguments=args_str)

    def to_tool_call(self) -> ToolCall:
        args_str = self.arguments.data if self.arguments else "{}"
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {"raw_arguments": args_str}
        return ToolCall(
            tool_call_id=self.tool_call_id.data if self.tool_call_id else "",
            tool_name=self.tool_name.data if self.tool_name else "",
            arguments=arguments
        )


class AssistantMessageChunk(BaseMessageChunk):
    """助手消息的流式块，支持 content 与 tool_call_chunks 的增量合并"""
    def __init__(
        self,
        content: Union[str, PygentString] = "",
        name: Optional[Union[str, PygentString]] = None,
        tool_call_chunks: Optional[List[ToolCallChunk]] = None,
        **kwargs
    ):
        super().__init__(role=MessageRole.ASSISTANT, content=content, name=name, **kwargs)
        self.tool_call_chunks = PygentList(tool_call_chunks or [])

    def _merge_chunk(self, other: 'BaseMessageChunk', swap_order: bool = False) -> 'AssistantMessageChunk':
        left, right = (other, self) if swap_order else (self, other)
        content = left._merge_content(right)
        name = right.name if right.name is not None else left.name
        metadata = left._merge_metadata(right)
        chunks = list(left.tool_call_chunks.data) if hasattr(left, 'tool_call_chunks') else []
        if isinstance(right, AssistantMessageChunk) and hasattr(right, 'tool_call_chunks'):
            for rc in right.tool_call_chunks.data:
                found = False
                for i, c in enumerate(chunks):
                    if c.index == rc.index:
                        chunks[i] = c + rc
                        found = True
                        break
                if not found:
                    chunks.append(rc)
        chunks.sort(key=lambda x: x.index)
        return AssistantMessageChunk(content=content, name=name, metadata=metadata, tool_call_chunks=chunks)

    def _merge_into_message(self, msg: 'BaseMessage') -> 'AssistantMessage':
        content = (msg.content.data if msg.content else "") + (self.content.data if self.content else "")
        name = msg.name if hasattr(msg, 'name') else None
        tool_calls = list(getattr(msg, 'tool_calls', PygentList()).data) if hasattr(msg, 'tool_calls') else []
        if hasattr(self, 'tool_call_chunks') and self.tool_call_chunks.data:
            for tc_chunk in self.tool_call_chunks.data:
                while len(tool_calls) <= tc_chunk.index:
                    tool_calls.append(None)
                existing = tool_calls[tc_chunk.index]
                if existing is None:
                    tool_calls[tc_chunk.index] = tc_chunk.to_tool_call()
                else:
                    merged = ToolCallChunk(
                        index=tc_chunk.index,
                        tool_call_id=existing.tool_call_id.data,
                        tool_name=existing.tool_name.data,
                        arguments=json.dumps(existing.arguments.data) if existing.arguments else "{}"
                    ) + tc_chunk
                    tool_calls[tc_chunk.index] = merged.to_tool_call()
            tool_calls = [tc for tc in tool_calls if tc is not None]
        return AssistantMessage(content=content, name=name, tool_calls=tool_calls)


class ToolMessage(BaseMessage):
    """工具消息"""
    def __init__(
        self,
        content: Union[str, PygentString],
        tool_call_id: Union[str, PygentString],
        **kwargs
    ):
        super().__init__(role=MessageRole.TOOL, content=content, **kwargs)
        self.tool_call_id = PygentString(tool_call_id)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = super().to_dict()
        result["tool_call_id"] = self.tool_call_id.data
        return result


class ToolMessageChunk(BaseMessageChunk):
    """工具消息的流式块"""
    def __init__(
        self,
        content: Union[str, PygentString] = "",
        tool_call_id: Optional[Union[str, PygentString]] = None,
        **kwargs
    ):
        super().__init__(role=MessageRole.TOOL, content=content, **kwargs)
        self.tool_call_id = PygentString(tool_call_id) if tool_call_id is not None else None

    def _merge_chunk(self, other: 'BaseMessageChunk', swap_order: bool = False) -> 'ToolMessageChunk':
        merged = super()._merge_chunk(other, swap_order)
        left, right = (other, self) if swap_order else (self, other)
        tid = (left.tool_call_id.data if isinstance(left, ToolMessageChunk) and left.tool_call_id else None) or (right.tool_call_id.data if isinstance(right, ToolMessageChunk) and right.tool_call_id else None) or ""
        return ToolMessageChunk(content=merged.content, tool_call_id=tid, metadata=merged.metadata)

    def _merge_into_message(self, msg: 'BaseMessage') -> 'ToolMessage':
        content = (msg.content.data if msg.content else "") + (self.content.data if self.content else "")
        tool_call_id = getattr(msg, 'tool_call_id', None)
        tid = (tool_call_id.data if tool_call_id else "") or (self.tool_call_id.data if self.tool_call_id else "")
        if not tid:
            raise ValueError("ToolMessage 必须包含 tool_call_id")
        return ToolMessage(content=content, tool_call_id=tid)


class FunctionMessage(BaseMessage):
    """函数消息（兼容旧版）"""
    def __init__(
        self,
        content: Union[str, PygentString],
        name: Union[str, PygentString],
        **kwargs
    ):
        super().__init__(role=MessageRole.FUNCTION, content=content, name=name, **kwargs)


class FunctionMessageChunk(BaseMessageChunk):
    """函数消息的流式块"""
    def __init__(
        self,
        content: Union[str, PygentString] = "",
        name: Optional[Union[str, PygentString]] = None,
        **kwargs
    ):
        super().__init__(role=MessageRole.FUNCTION, content=content, name=name, **kwargs)

    def _merge_chunk(self, other: 'BaseMessageChunk', swap_order: bool = False) -> 'FunctionMessageChunk':
        left, right = (other, self) if swap_order else (self, other)
        if isinstance(right, FunctionMessageChunk) and left.name is not None and right.name is not None and left.name.data != right.name.data:
            raise ValueError(f"FunctionMessageChunk name 不匹配: {left.name.data} != {right.name.data}")
        merged = super()._merge_chunk(other, swap_order)
        return FunctionMessageChunk(content=merged.content, name=merged.name, metadata=merged.metadata)

    def _merge_into_message(self, msg: 'BaseMessage') -> 'FunctionMessage':
        content = (msg.content.data if msg.content else "") + (self.content.data if self.content else "")
        name = (msg.name.data if msg.name else "") or (self.name.data if self.name else "")
        if not name:
            raise ValueError("FunctionMessage 必须包含 name")
        return FunctionMessage(content=content, name=name)


# ==================== 工具调用相关类（FunctionCall 等） ====================

class FunctionCall(PygentData):
    """函数调用（兼容旧版）"""
    name: PygentString
    arguments: PygentDict
    
    def __init__(
        self,
        name: Union[str, PygentString],
        arguments: Union[Dict, str, PygentDict],
        **kwargs
    ):
        super().__init__({})
        
        if isinstance(name, PygentString):
            self.name = name
        else:
            self.name = PygentString(str(name))
        
        if isinstance(arguments, PygentDict):
            self.arguments = arguments
        elif isinstance(arguments, str):
            try:
                parsed_args = json.loads(arguments)
                self.arguments = PygentDict(parsed_args)
            except json.JSONDecodeError:
                self.arguments = PygentDict({"input": arguments})
        elif isinstance(arguments, dict):
            self.arguments = PygentDict(arguments)
        else:
            self.arguments = PygentDict({"arguments": str(arguments)})
        
        for key, value in kwargs.items():
            if value is not None:
                self.data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name.data,
            "arguments": json.dumps(self.arguments.data) if self.arguments.data else "{}"
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        args_preview = str(self.arguments.data)[:50] + "..." if len(str(self.arguments.data)) > 50 else self.arguments.data
        return f"FunctionCall({self.name.data}: {args_preview})"


# ==================== 元数据相关类 ====================

class MessageMetadata(PygentDict):
    """消息元数据"""
    def __init__(self, data: Optional[Dict] = None, **kwargs):
        init_data = data or {}
        init_data.update(kwargs)
        super().__init__(init_data)
    
    @property
    def timestamp(self) -> Optional[str]:
        """获取时间戳"""
        return self.data.get("timestamp")
    
    @timestamp.setter
    def timestamp(self, value: str):
        """设置时间戳"""
        self.data["timestamp"] = value
    
    @property
    def source(self) -> Optional[str]:
        """获取消息来源"""
        return self.data.get("source")
    
    @source.setter
    def source(self, value: str):
        """设置消息来源"""
        self.data["source"] = value
    
    @property
    def tokens(self) -> Optional[int]:
        """获取token数量"""
        return self.data.get("tokens")
    
    @tokens.setter
    def tokens(self, value: int):
        """设置token数量"""
        self.data["tokens"] = value
    
    def add_custom_field(self, key: str, value: Any):
        """添加自定义字段"""
        self.data[key] = value
    
    def get_custom_field(self, key: str, default: Any = None) -> Any:
        """获取自定义字段"""
        return self.data.get(key, default)


# ==================== 导出 ====================

__all__ = [
    # 消息相关
    'BaseMessage',
    'BaseMessageChunk',
    'SystemMessage',
    'SystemMessageChunk',
    'UserMessage',
    'UserMessageChunk',
    'AssistantMessage',
    'AssistantMessageChunk',
    'ToolMessage',
    'ToolMessageChunk',
    'FunctionMessage',
    'FunctionMessageChunk',
    'MessageRole',
    
    # 工具调用
    'ToolCall',
    'ToolCallChunk',
    'FunctionCall',
    
    # 元数据
    'MessageMetadata',
]
