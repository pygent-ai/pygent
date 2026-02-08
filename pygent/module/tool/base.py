from typing import Any, Dict, List, Optional, Union, Callable, Type, get_type_hints, get_origin
from enum import Enum
import inspect
import json
from dataclasses import dataclass, field
from datetime import datetime

from pygent.module import PygentModule
from pygent.common import (
    PygentData,
    PygentString,
    PygentDict,
    PygentList,
    PygentBool,
    PygentInt,
    PygentFloat,
)


class ToolCategory(Enum):
    """工具类别枚举"""
    SEARCH = "search"
    CALCULATION = "calculation"
    DATABASE = "database"
    FILE = "file"
    NETWORK = "network"
    SYSTEM = "system"
    UTILITY = "utility"
    AI = "ai"
    CUSTOM = "custom"


class ToolPermission(Enum):
    """工具权限级别"""
    PUBLIC = "public"      # 完全公开
    LIMITED = "limited"    # 有限访问
    PRIVATE = "private"   # 私有工具
    ADMIN = "admin"       # 管理员权限


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "unknown"
    category: ToolCategory = ToolCategory.UTILITY
    permission: ToolPermission = ToolPermission.PUBLIC
    tags: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # 每分钟调用限制
    timeout: float = 30.0  # 超时时间（秒）
    requires_auth: bool = False  # 是否需要认证
    deprecated: bool = False  # 是否已弃用
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ToolParameter(PygentData):
    """工具参数定义"""
    def __init__(
        self,
        name: str,
        type: Union[Type, str],
        description: str = "",
        required: bool = True,
        default: Any = None,
        enum: Optional[List[Any]] = None,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        pattern: Optional[str] = None,  # 正则表达式
        **kwargs
    ):
        super().__init__({
            "name": name,
            "type": self._type_to_string(type),
            "description": description,
            "required": required,
            "default": default,
            "enum": enum,
            "min_value": min_value,
            "max_value": max_value,
            "pattern": pattern,
            **kwargs
        })
    
    def _type_to_string(self, type_obj: Union[Type, str]) -> str:
        """将Python类型转换为字符串表示（也接受已是字符串的 type）"""
        if isinstance(type_obj, str):
            return type_obj
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null"
        }
        
        if type_obj in type_map:
            return type_map[type_obj]
        
        # 处理 typing.List[T], typing.Dict[K,V] 等
        origin = get_origin(type_obj)
        if origin is list:
            return "array"
        if origin is dict:
            return "object"
        
        # 处理可选类型 Optional[X]
        if hasattr(type_obj, '__origin__') and type_obj.__origin__ == Union:
            # 查找非None的类型
            for arg in type_obj.__args__:
                if arg != type(None):
                    return self._type_to_string(arg)
        
        # 默认为字符串
        return "string"
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为OpenAI参数模式"""
        schema = {
            "type": self.data["type"],
            "description": self.data["description"]
        }
        
        # 添加约束条件
        constraints = {}
        
        if self.data["enum"]:
            schema["enum"] = self.data["enum"]
        
        if self.data["min_value"] is not None:
            if self.data["type"] in ["integer", "number"]:
                constraints["minimum"] = self.data["min_value"]
        
        if self.data["max_value"] is not None:
            if self.data["type"] in ["integer", "number"]:
                constraints["maximum"] = self.data["max_value"]
        
        if self.data["pattern"]:
            if self.data["type"] == "string":
                constraints["pattern"] = self.data["pattern"]
        
        if constraints:
            schema.update(constraints)
        
        return schema


class BaseTool(PygentModule):
    """工具基类，支持大模型工具调用"""
    
    # 元数据字段（必须通过PygentData包装）
    metadata: PygentDict
    
    # 参数定义
    parameters: PygentDict
    
    # 工具配置
    config: PygentDict
    
    # 工具状态
    enabled: PygentBool
    call_count: PygentInt
    last_called: PygentString
    error_count: PygentInt
    
    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        **kwargs
    ):
        """
        初始化工具
        
        Args:
            name: 工具名称
            description: 工具描述
            version: 工具版本
            **kwargs: 其他元数据
        """
        super().__init__()
        
        # 初始化元数据
        metadata = ToolMetadata(
            name=name,
            description=description,
            version=version,
            **{k: v for k, v in kwargs.items() if k in ToolMetadata.__annotations__}
        )
        
        self.metadata = PygentDict(metadata.__dict__)
        
        # 初始化参数定义
        self.parameters = PygentDict({})
        
        # 初始化配置
        self.config = PygentDict({})
        
        # 初始化状态
        self.enabled = PygentBool(True)
        self.call_count = PygentInt(0)
        self.last_called = PygentString("")
        self.error_count = PygentInt(0)
        
        # 自动发现并注册参数
        self._discover_parameters()
    
    def _discover_parameters(self) -> None:
        """从forward方法的类型注解自动发现参数"""
        # 获取forward方法的签名
        forward_method = self.forward
        if forward_method == BaseTool.forward:
            return  # 基类方法，不自动发现
        
        sig = inspect.signature(forward_method)
        type_hints = get_type_hints(forward_method)
        
        parameters = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            # 跳过 *args / **kwargs（装饰器生成的 forward(self, *args, **kwargs)）
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # 获取参数类型
            param_type = type_hints.get(param_name, str)
            
            # 检查是否有默认值
            required = param.default == inspect.Parameter.empty
            
            # 创建参数定义
            param_def = ToolParameter(
                name=param_name,
                type=param_type,
                required=required,
                default=param.default if not required else None
            )
            
            parameters[param_name] = param_def.data
        
        if parameters:
            self.parameters.data.update(parameters)
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        验证参数
        
        Args:
            parameters: 输入的参数字典
            
        Returns:
            错误字典，键为参数名，值为错误消息列表
        """
        errors = {}
        
        for param_name, param_def in self.parameters.data.items():
            param_value = parameters.get(param_name)
            param_definition = param_def if isinstance(param_def, dict) else param_def.data
            
            # 检查必需参数
            if param_definition.get("required", True) and param_name not in parameters:
                if param_name not in errors:
                    errors[param_name] = []
                errors[param_name].append("参数是必需的")
                continue
            
            # 如果参数为None且有默认值，使用默认值
            if param_value is None and "default" in param_definition:
                parameters[param_name] = param_definition["default"]
                param_value = param_definition["default"]
            
            # 类型检查
            if param_value is not None:
                param_type = param_definition.get("type", "string")
                
                # 类型验证
                type_errors = self._validate_type(param_value, param_type, param_name, param_definition)
                if type_errors:
                    if param_name not in errors:
                        errors[param_name] = []
                    errors[param_name].extend(type_errors)
        
        # 检查是否有未知参数
        known_params = set(self.parameters.data.keys())
        input_params = set(parameters.keys())
        unknown_params = input_params - known_params
        
        if unknown_params:
            errors.setdefault("_unknown", []).append(f"未知参数: {', '.join(unknown_params)}")
        
        return errors
    
    def _validate_type(
        self, 
        value: Any, 
        expected_type: str, 
        param_name: str,
        param_def: Dict[str, Any]
    ) -> List[str]:
        """验证参数类型"""
        errors = []
        
        type_checkers = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int),
            "number": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }
        
        checker = type_checkers.get(expected_type)
        if checker and not checker(value):
            errors.append(f"期望类型 {expected_type}, 实际类型 {type(value).__name__}")
        
        # 枚举验证
        if "enum" in param_def and param_def["enum"]:
            if value not in param_def["enum"]:
                errors.append(f"值必须是以下之一: {param_def['enum']}")
        
        # 数值范围验证
        if expected_type in ["integer", "number"]:
            if "min_value" in param_def and param_def["min_value"] is not None:
                if value < param_def["min_value"]:
                    errors.append(f"值必须大于等于 {param_def['min_value']}")
            
            if "max_value" in param_def and param_def["max_value"] is not None:
                if value > param_def["max_value"]:
                    errors.append(f"值必须小于等于 {param_def['max_value']}")
        
        # 字符串模式验证
        if expected_type == "string" and "pattern" in param_def and param_def["pattern"]:
            import re
            if not re.match(param_def["pattern"], str(value)):
                errors.append(f"字符串必须匹配模式: {param_def['pattern']}")
        
        return errors
    
    def forward(self, *args, **kwargs) -> Any:
        """
        工具执行方法
        
        注意：子类必须实现此方法
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            工具执行结果
        """
        raise NotImplementedError("子类必须实现 forward 方法")
    
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        调用工具（主入口）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            包含结果和状态的字典
        """
        # 检查工具是否启用
        if not self.enabled.data:
            return self._create_error_response("工具已被禁用")
        
        try:
            # 更新调用统计
            self.call_count.data += 1
            self.last_called.data = datetime.now().isoformat()
            
            # 验证参数
            if kwargs:
                errors = self.validate_parameters(kwargs)
                if errors:
                    return self._create_error_response("参数验证失败", details=errors)
            
            # 执行工具
            result = self.forward(*args, **kwargs)
            
            # 创建成功响应
            return self._create_success_response(result)
            
        except Exception as e:
            # 更新错误统计
            self.error_count.data += 1
            
            # 创建错误响应
            return self._create_error_response(str(e), exception=e)
    
    def _create_success_response(self, result: Any) -> Dict[str, Any]:
        """创建成功响应"""
        return {
            "success": True,
            "result": result,
            "metadata": {
                "tool": self.metadata.data["name"],
                "version": self.metadata.data["version"],
                "call_id": f"{self.metadata.data['name']}_{self.call_count.data}",
                "timestamp": datetime.now().isoformat(),
                "execution_time": None,  # 可以在子类中填充
            },
            "status": {
                "call_count": self.call_count.data,
                "error_count": self.error_count.data,
                "enabled": self.enabled.data,
            }
        }
    
    def _create_error_response(self, error: str, details: Any = None, exception: Exception = None) -> Dict[str, Any]:
        """创建错误响应"""
        response = {
            "success": False,
            "error": error,
            "metadata": {
                "tool": self.metadata.data["name"],
                "version": self.metadata.data["version"],
                "call_id": f"{self.metadata.data['name']}_{self.call_count.data}",
                "timestamp": datetime.now().isoformat(),
            },
            "status": {
                "call_count": self.call_count.data,
                "error_count": self.error_count.data,
                "enabled": self.enabled.data,
            }
        }
        
        if details:
            response["details"] = details
        
        if exception:
            response["exception"] = {
                "type": exception.__class__.__name__,
                "message": str(exception)
            }
        
        return response
    
    def to_openai_function(self) -> Dict[str, Any]:
        """转换为OpenAI Function格式"""
        # 构建参数模式
        properties = {}
        required_params = []
        
        for param_name, param_def in self.parameters.data.items():
            param_definition = param_def if isinstance(param_def, dict) else param_def.data
            properties[param_name] = ToolParameter(**param_definition).to_openai_schema()
            
            if param_definition.get("required", True):
                required_params.append(param_name)
        
        return {
            "name": self.metadata.data["name"],
            "description": self.metadata.data["description"],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }
    
    def to_langchain_tool(self) -> Dict[str, Any]:
        """转换为LangChain工具格式"""
        return {
            "name": self.metadata.data["name"],
            "description": self.metadata.data["description"],
            "func": self.__call__,
            "args_schema": self._create_args_schema()
        }
    
    def _create_args_schema(self) -> Type:
        """创建Pydantic参数模式（用于LangChain）"""
        try:
            from pydantic import BaseModel, Field
            from typing import Optional as PydanticOptional
            
            # 动态创建Pydantic模型类
            fields = {}
            
            for param_name, param_def in self.parameters.data.items():
                param_definition = param_def if isinstance(param_def, dict) else param_def.data
                
                # 构建字段配置
                field_config = {
                    "description": param_definition.get("description", ""),
                    "default": ... if param_definition.get("required", True) else param_definition.get("default")
                }
                
                # 添加额外约束
                extra = {}
                if "enum" in param_definition and param_definition["enum"]:
                    extra["enum"] = param_definition["enum"]
                
                if "min_value" in param_definition:
                    extra["ge"] = param_definition["min_value"]
                
                if "max_value" in param_definition:
                    extra["le"] = param_definition["max_value"]
                
                if extra:
                    field_config.update(extra)
                
                # 确定字段类型
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": float,
                    "boolean": bool,
                    "array": List,
                    "object": Dict,
                }
                
                field_type = type_map.get(param_definition.get("type", "string"), str)
                if not param_definition.get("required", True):
                    field_type = PydanticOptional[field_type]
                
                fields[param_name] = (field_type, Field(**field_config))
            
            # 动态创建模型类：__annotations__ 为 name -> type，类属性为 name -> Field(...)
            model_attrs = {"__annotations__": {k: v[0] for k, v in fields.items()}}
            for k, v in fields.items():
                model_attrs[k] = v[1]
            ModelClass = type(
                f"{self.metadata.data['name']}Args",
                (BaseModel,),
                model_attrs,
            )
            return ModelClass
            
        except ImportError:
            print("警告: 未安装pydantic，无法创建LangChain参数模式")
            return None
    
    def get_schema(self) -> Dict[str, Any]:
        """获取完整的工具模式"""
        return {
            "metadata": self.metadata.data,
            "parameters": self.parameters.data,
            "config": self.config.data,
            "openai_function": self.to_openai_function(),
            "status": {
                "enabled": self.enabled.data,
                "call_count": self.call_count.data,
                "error_count": self.error_count.data,
                "last_called": self.last_called.data,
            }
        }
    
    def enable(self) -> None:
        """启用工具"""
        self.enabled.data = True
        print(f"工具 '{self.metadata.data['name']}' 已启用")
    
    def disable(self) -> None:
        """禁用工具"""
        self.enabled.data = False
        print(f"工具 '{self.metadata.data['name']}' 已禁用")
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.call_count.data = 0
        self.error_count.data = 0
        self.last_called.data = ""
        print(f"工具 '{self.metadata.data['name']}' 统计信息已重置")
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.data.update(config)
        print(f"工具 '{self.metadata.data['name']}' 配置已更新")
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.metadata.data['name']} (v{self.metadata.data['version']}): {self.metadata.data['description']}"


