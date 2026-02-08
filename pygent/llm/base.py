from typing import Any, Dict, List, Optional, Union, Iterator
from abc import ABC, abstractmethod

from pygent.common import PygentData, PygentOperator, PygentString, PygentDict, PygentInt, PygentFloat, PygentBool
from pygent.context import BaseContext


class BaseClient(PygentOperator, ABC):
    """基础模型客户端（抽象基类）- 单次调用版本"""
    
    # 定义PygentData字段
    base_url: PygentString
    api_key: PygentString
    model_name: PygentString
    timeout: PygentInt
    max_retries: PygentInt
    temperature: PygentFloat
    max_tokens: PygentInt
    stream: PygentBool
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 30,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """
        初始化客户端
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            model_name: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            temperature: 温度参数
            max_tokens: 最大输出token数
            stream: 是否使用流式响应
            **kwargs: 其他配置参数
        """
        super().__init__()
        
        # 初始化配置字段
        self.base_url = PygentString(base_url)
        self.api_key = PygentString(api_key)
        self.model_name = PygentString(model_name)
        self.timeout = PygentInt(timeout)
        self.max_retries = PygentInt(max_retries)
        self.temperature = PygentFloat(temperature)
        self.max_tokens = PygentInt(max_tokens or 2048)
        self.stream = PygentBool(stream)
        
        # 设置额外配置
        self.config = PygentDict(kwargs)
        
        # 统计信息（可选）
        self.total_requests = PygentInt(0)
        self.total_errors = PygentInt(0)

    def forward(self, context: BaseContext) -> BaseContext:
        raise NotImplementedError

    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(model={self.model_name.data}, url={self.base_url.data})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_url={repr(self.base_url.data)}, model_name={repr(self.model_name.data)})"


class BaseAsyncClient(PygentOperator, ABC):
    """基础模型客户端（抽象基类）- 单次调用版本"""

    # 定义PygentData字段
    base_url: PygentString
    api_key: PygentString
    model_name: PygentString
    timeout: PygentFloat
    max_retries: PygentInt
    temperature: PygentFloat
    max_tokens: PygentInt
    stream: PygentBool

    def __init__(
            self,
            base_url: str,
            api_key: str,
            model_name: str,
            timeout: int = 30,
            max_retries: int = 3,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stream: bool = False,
            **kwargs
    ):
        """
        初始化客户端

        Args:
            base_url: API基础URL
            api_key: API密钥
            model_name: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            temperature: 温度参数
            max_tokens: 最大输出token数
            stream: 是否使用流式响应
            **kwargs: 其他配置参数
        """
        super().__init__()

        # 初始化配置字段
        self.base_url = PygentString(base_url)
        self.api_key = PygentString(api_key)
        self.model_name = PygentString(model_name)
        self.timeout = PygentFloat(timeout)
        self.max_retries = PygentInt(max_retries)
        self.temperature = PygentFloat(temperature)
        self.max_tokens = PygentInt(max_tokens or 2048)
        self.stream = PygentBool(stream)

        # 设置额外配置
        self.config = PygentDict(kwargs)

        # 统计信息（可选）
        self.total_requests = PygentInt(0)
        self.total_errors = PygentInt(0)

    async def forward(self, context: BaseContext) -> BaseContext:
        raise NotImplementedError

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(model={self.model_name.data}, url={self.base_url.data})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_url={repr(self.base_url.data)}, model_name={repr(self.model_name.data)})"

