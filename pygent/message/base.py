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


class SystemMessage(BaseMessage):
    """系统消息"""
    def __init__(self, content: Union[str, PygentString], **kwargs):
        super().__init__(role=MessageRole.SYSTEM, content=content, **kwargs)


class UserMessage(BaseMessage):
    """用户消息"""
    def __init__(self, content: Union[str, PygentString], name: Optional[Union[str, PygentString]] = None, **kwargs):
        super().__init__(role=MessageRole.USER, content=content, name=name, **kwargs)


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


class FunctionMessage(BaseMessage):
    """函数消息（兼容旧版）"""
    def __init__(
        self,
        content: Union[str, PygentString],
        name: Union[str, PygentString],
        **kwargs
    ):
        super().__init__(role=MessageRole.FUNCTION, content=content, name=name, **kwargs)


# ==================== 工具调用相关类 ====================

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
        
        # 初始化工具调用ID
        if isinstance(tool_call_id, PygentString):
            self.tool_call_id = tool_call_id
        else:
            self.tool_call_id = PygentString(str(tool_call_id))
        
        # 初始化工具名称
        if isinstance(tool_name, PygentString):
            self.tool_name = tool_name
        else:
            self.tool_name = PygentString(str(tool_name))
        
        # 初始化参数（可能是字符串或字典）
        if isinstance(arguments, PygentDict):
            self.arguments = arguments
        elif isinstance(arguments, str):
            try:
                # 尝试解析JSON字符串
                parsed_args = json.loads(arguments)
                self.arguments = PygentDict(parsed_args)
            except json.JSONDecodeError:
                # 如果不是有效的JSON，作为普通字符串处理
                self.arguments = PygentDict({"input": arguments})
        elif isinstance(arguments, dict):
            self.arguments = PygentDict(arguments)
        else:
            self.arguments = PygentDict({"arguments": str(arguments)})
        
        # 处理额外参数
        for key, value in kwargs.items():
            if value is not None:
                self.data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "id": self.tool_call_id.data,
            "type": "function",
            "function": {
                "name": self.tool_name.data,
                "arguments": json.dumps(self.arguments.data) if self.arguments.data else "{}"
            }
        }
        
        # 添加额外字段
        for key, value in self.data.items():
            if key not in ['tool_call_id', 'tool_name', 'arguments']:
                result[key] = value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """从字典创建工具调用"""
        # OpenAI格式
        if "id" in data and "function" in data:
            tool_call_id = data["id"]
            tool_name = data["function"]["name"]
            arguments_str = data["function"]["arguments"]
            
            try:
                arguments = json.loads(arguments_str)
            except (json.JSONDecodeError, TypeError):
                arguments = {"raw_arguments": arguments_str}
            
            return cls(tool_call_id=tool_call_id, tool_name=tool_name, arguments=arguments)
        
        # 简化格式
        return cls(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {})
        )
    
    def __str__(self) -> str:
        """字符串表示"""
        args_preview = str(self.arguments.data)[:50] + "..." if len(str(self.arguments.data)) > 50 else self.arguments.data
        return f"ToolCall({self.tool_name.data}: {args_preview})"


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
    'SystemMessage',
    'UserMessage',
    'AssistantMessage',
    'ToolMessage',
    'FunctionMessage',
    'MessageRole',
    
    # 工具调用
    'ToolCall',
    'FunctionCall',
    
    # 元数据
    'MessageMetadata',
]
