from typing import Any, Dict, List, Optional, Union, Callable, Type, get_type_hints, get_origin
import inspect
import functools
import re

from .base import BaseTool, ToolParameter, ToolMetadata, ToolCategory, ToolPermission


# ==================== 工具函数 ====================

def _python_type_to_string(py_type: Type) -> str:
    """Python类型转换为字符串表示"""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
        bytes: "string",  # 字节通常作为base64字符串
    }
    
    # 处理基础类型
    if py_type in type_map:
        return type_map[py_type]
    
    # 处理Union类型（如Optional）
    if hasattr(py_type, '__origin__'):
        if py_type.__origin__ == Union:
            # 找到非None的类型
            for arg in py_type.__args__:
                if arg != type(None):
                    return _python_type_to_string(arg)
    
    # 处理List[T], Dict[K, V]等
    origin = get_origin(py_type)
    if origin:
        if origin == list:
            return "array"
        elif origin == dict:
            return "object"
    
    # 处理类型别名
    if hasattr(py_type, '__name__'):
        type_name = py_type.__name__.lower()
        if type_name in ['str', 'string']:
            return "string"
        elif type_name in ['int', 'integer']:
            return "integer"
        elif type_name in ['float', 'number']:
            return "number"
        elif type_name in ['bool', 'boolean']:
            return "boolean"
    
    # 默认为字符串
    return "string"


def _extract_function_description(docstring: Optional[str]) -> str:
    """从文档字符串提取函数描述"""
    if not docstring:
        return ""
    
    lines = docstring.strip().split('\n')
    description_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith((':param', ':type', ':return', ':raises', 
                                       'Args:', 'Arguments:', 'Parameters:', 
                                       'Returns:', 'Raises:', 'Yields:')):
            break
        description_lines.append(line)
    
    return ' '.join(description_lines).strip() or ""


def _parse_docstring(docstring: str) -> Dict[str, Any]:
    """
    解析文档字符串，提取参数信息
    
    支持多种文档格式：
    - Google风格
    - Numpy风格
    - reStructuredText风格
    - 简单风格
    """
    if not docstring:
        return {}
    
    result = {
        'params': {},     # 参数描述
        'types': {},      # 参数类型
        'enums': {},      # 枚举值
        'ranges': {},     # 取值范围
        'patterns': {},   # 正则模式
        'required': {},   # 是否必需
    }
    
    lines = docstring.strip().split('\n')
    current_section = None
    param_buffer = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 跳过空行但保持buffer
        if not line:
            if param_buffer:
                _process_param_buffer(param_buffer, result)
                param_buffer = []
            continue
        
        # 检测章节开始
        section_lower = line.lower()
        if section_lower in ['args:', 'arguments:', 'parameters:']:
            current_section = 'google_args'
            if param_buffer:
                _process_param_buffer(param_buffer, result)
                param_buffer = []
            continue
        elif section_lower in ['returns:', 'returns', 'return:']:
            current_section = 'returns'
            if param_buffer:
                _process_param_buffer(param_buffer, result)
                param_buffer = []
            continue
        elif section_lower.startswith('param ') or section_lower.startswith('parameter '):
            current_section = 'numpy_params'
            if param_buffer:
                _process_param_buffer(param_buffer, result)
                param_buffer = []
            continue
        
        # 处理参数行
        if current_section == 'google_args':
            # Google风格: param: description
            if ':' in line and not line.startswith(':'):
                if param_buffer:
                    _process_param_buffer(param_buffer, result)
                    param_buffer = []
                param_buffer.append(line)
            elif param_buffer:
                # 多行描述，追加到上一个参数
                param_buffer[-1] = param_buffer[-1] + ' ' + line
        
        elif current_section == 'numpy_params':
            # Numpy风格: param_name : type description
            if ':' in line or param_buffer:
                param_buffer.append(line)
        
        # reStructuredText风格: :param param_name: description
        elif line.startswith(':param'):
            if param_buffer:
                _process_param_buffer(param_buffer, result)
                param_buffer = []
            param_buffer.append(line)
        
        # reStructuredText风格: :type param_name: type
        elif line.startswith(':type'):
            if ':' in line:
                parts = line.split(':', 1)
                param_info = parts[0].replace(':type', '').strip()
                type_str = parts[1].strip() if len(parts) > 1 else ""
                
                if param_info and type_str:
                    result['types'][param_info] = type_str
        
        # 简单风格（无章节标记）
        elif current_section is None and ':' in line and not line.startswith(':'):
            parts = line.split(':', 1)
            param_name = parts[0].strip()
            # 检查是否是参数行（不是普通的句子）
            if param_name and ' ' not in param_name and len(param_name) < 30:
                if param_buffer:
                    _process_param_buffer(param_buffer, result)
                    param_buffer = []
                param_buffer.append(line)
    
    # 处理最后一个buffer
    if param_buffer:
        _process_param_buffer(param_buffer, result)
    
    return result


def _process_param_buffer(buffer: List[str], result: Dict[str, Any]):
    """处理参数缓冲区"""
    for line in buffer:
        line = line.strip()
        
        # reStructuredText风格
        if line.startswith(':param'):
            if ':' in line:
                parts = line.split(':', 1)
                param_info = parts[0].replace(':param', '').strip()
                description = parts[1].strip() if len(parts) > 1 else ""
                
                # 提取类型和参数名
                if ' ' in param_info:
                    # 有类型信息: :param type param_name:
                    type_str, param_name = param_info.split()[:2]
                    result['types'][param_name] = type_str
                else:
                    # 只有参数名: :param param_name:
                    param_name = param_info
                
                result['params'][param_name] = description
                _extract_constraints(param_name, description, result)
        
        # Google风格
        elif ':' in line and not line.startswith(':'):
            parts = line.split(':', 1)
            if len(parts) == 2:
                param_name = parts[0].strip()
                description = parts[1].strip()
                
                result['params'][param_name] = description
                _extract_constraints(param_name, description, result)
        
        # Numpy风格
        elif ':' in line or len(buffer) > 1:
            # Numpy风格可能跨多行
            full_line = ' '.join(buffer)
            if ':' in full_line:
                # 参数名 : 类型 描述
                param_part, desc_part = full_line.split(':', 1) if ':' in full_line else (full_line, '')
                param_part = param_part.strip()
                desc_part = desc_part.strip()
                
                # 提取参数名和类型
                if ' ' in param_part:
                    param_name = param_part.split()[0].strip()
                    # 类型可能在括号中或空格后
                    type_match = re.search(r':\s*([^:]+)', param_part)
                    if type_match:
                        type_str = type_match.group(1).strip()
                        result['types'][param_name] = type_str
                    elif len(param_part.split()) > 1:
                        # 假设第二个单词是类型
                        type_str = param_part.split()[1].strip()
                        result['types'][param_name] = type_str
                else:
                    param_name = param_part
                
                if param_name:
                    result['params'][param_name] = desc_part
                    _extract_constraints(param_name, desc_part, result)


def _extract_constraints(param_name: str, description: str, result: Dict[str, Any]):
    """从描述中提取约束条件"""
    # 提取枚举值
    _extract_enum_from_description(param_name, description, result)
    
    # 提取取值范围
    _extract_range_from_description(param_name, description, result)
    
    # 提取正则模式
    _extract_pattern_from_description(param_name, description, result)
    
    # 提取是否必需
    if 'required' in description.lower() or 'must' in description.lower():
        result['required'][param_name] = True


def _extract_enum_from_description(param_name: str, description: str, result: Dict[str, Any]):
    """从描述中提取枚举值"""
    enum_patterns = [
        r'one of \[([^\]]+)\]',  # one of [A, B, C]
        r'options?: ([^,]+(?:\|[^,]+)+)',  # options: A|B|C
        r'values?: ([^,]+(?:\|[^,]+)+)',   # values: A|B|C
        r'either ([^,]+) or ([^,\.]+)',    # either A or B
        r'must be (?:either )?([^,\.]+)(?: or ([^,\.]+))?',  # must be A or B
        r'choices?: ([^,]+(?:\|[^,]+)+)',  # choices: A|B|C
    ]
    
    for pattern in enum_patterns:
        match = re.search(pattern, description, re.IGNORECASE)
        if match:
            if 'one of' in pattern:
                # 处理 [A, B, C] 格式
                enum_str = match.group(1)
                enum_values = [v.strip().strip("'\"") for v in enum_str.split(',')]
                result['enums'][param_name] = enum_values
                break
            elif 'either' in pattern or 'must be' in pattern:
                # 处理 either A or B 或 must be A or B 格式
                enum_values = []
                for i in range(1, 3):
                    if match.group(i):
                        enum_values.append(match.group(i).strip().strip("' \""))
                if enum_values:
                    result['enums'][param_name] = enum_values
                break
            else:
                # 处理 A|B|C 格式
                enum_str = match.group(1)
                enum_values = [v.strip().strip("'\"") for v in enum_str.split('|')]
                result['enums'][param_name] = enum_values
                break


def _extract_range_from_description(param_name: str, description: str, result: Dict[str, Any]):
    """从描述中提取取值范围"""
    range_patterns = [
        r'between (\d+(?:\.\d+)?) and (\d+(?:\.\d+)?)',  # between 1 and 10
        r'range[:\s]+(\d+(?:\.\d+)?)[-\s]+(\d+(?:\.\d+)?)',  # range: 1-10
        r'>=?\s*(\d+(?:\.\d+)?)',  # >= 0, > 0
        r'<=?\s*(\d+(?:\.\d+)?)',  # <= 100, < 100
        r'minimum[:\s]+(\d+(?:\.\d+)?)',  # minimum: 0
        r'maximum[:\s]+(\d+(?:\.\d+)?)',  # maximum: 100
        r'greater than (\d+(?:\.\d+)?)',  # greater than 0
        r'less than (\d+(?:\.\d+)?)',  # less than 100
    ]
    
    min_val = None
    max_val = None
    
    for pattern in range_patterns:
        matches = re.findall(pattern, description, re.IGNORECASE)
        if matches:
            if 'between' in pattern or 'range' in pattern:
                # between X and Y 或 range: X-Y
                min_val = float(matches[0][0])
                max_val = float(matches[0][1])
                break
            elif '>=' in pattern or 'minimum' in pattern:
                # >= X 或 minimum: X
                min_val = float(matches[0])
            elif '>' in pattern or 'greater' in pattern:
                # > X 或 greater than X
                min_val = float(matches[0]) + 0.001  # 稍微大于
            elif '<=' in pattern or 'maximum' in pattern:
                # <= X 或 maximum: X
                max_val = float(matches[0])
            elif '<' in pattern or 'less' in pattern:
                # < X 或 less than X
                max_val = float(matches[0]) - 0.001  # 稍微小于
    
    if min_val is not None or max_val is not None:
        result['ranges'][param_name] = (min_val, max_val)


def _extract_pattern_from_description(param_name: str, description: str, result: Dict[str, Any]):
    """从描述中提取正则模式"""
    # 匹配模式描述: pattern: XXX, format: XXX, must match XXX
    pattern_keywords = ['pattern', 'format', 'regex', 'match']
    
    for keyword in pattern_keywords:
        if keyword in description.lower():
            # 尝试提取具体的模式
            pattern_match = re.search(rf'{keyword}[:\s]+([^\s,\.;]+)', description, re.IGNORECASE)
            if pattern_match:
                pattern_str = pattern_match.group(1)
                # 常见的模式简写
                pattern_map = {
                    'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    'phone': r'^\+?[\d\s\-\(\)]+$',
                    'url': r'^(https?|ftp)://[^\s/$.?#].[^\s]*$',
                    'date': r'^\d{4}-\d{2}-\d{2}$',
                    'time': r'^\d{2}:\d{2}(:\d{2})?$',
                    'ip': r'^\d{1,3}(\.\d{1,3}){3}$',
                    'uuid': r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$',
                    'hex': r'^[0-9a-fA-F]+$',
                }
                
                if pattern_str.lower() in pattern_map:
                    result['patterns'][param_name] = pattern_map[pattern_str.lower()]
                else:
                    result['patterns'][param_name] = pattern_str
                break


# ==================== 主要装饰器 ====================

def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0.0",
    author: str = "unknown",
    category: Union[str, ToolCategory] = "utility",
    permission: Union[str, ToolPermission] = "public",
    tags: Optional[List[str]] = None,
    rate_limit: Optional[int] = None,
    timeout: float = 30.0,
    requires_auth: bool = False,
    deprecated: bool = False,
    enabled: bool = True,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    工具装饰器，将普通函数转换为BaseTool对象
    
    使用示例:
        @tool(name="calculator", description="数学计算工具")
        def calculate(expression: str) -> float:
            return eval(expression)
    
        # 直接调用
        result = calculate("2 + 3 * 4")
        
        # 获取工具对象
        tool_obj = calculate.tool
    """
    def decorator(func: Callable) -> Callable:
        # 确定工具名称
        tool_name = name or func.__name__
        
        # 确定工具描述
        tool_description = description or _extract_function_description(func.__doc__) or f"工具: {func.__name__}"
        
        # 处理类别和权限（如果是Enum则获取value）
        if isinstance(category, ToolCategory):
            category_value = category.value
        else:
            category_value = category
        
        if isinstance(permission, ToolPermission):
            permission_value = permission.value
        else:
            permission_value = permission
        
        # 获取函数签名和类型提示
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # 创建工具类
        tool_class_name = f"Tool_{func.__name__.title().replace('_', '')}"
        
        # 定义初始化方法
        def __init__(self):
            # 先调用父类初始化
            BaseTool.__init__(
                self,
                name=tool_name,
                description=tool_description,
                version=version,
                author=author,
                category=category_value,
                permission=permission_value,
                tags=tags or [],
                rate_limit=rate_limit,
                timeout=timeout,
                requires_auth=requires_auth,
                deprecated=deprecated,
                **kwargs
            )
        
        # 定义forward方法
        def forward(self, *args, **kwargs):
            return func(*args, **kwargs)
        
        # 动态创建工具类
        ToolClass = type(
            tool_class_name,
            (BaseTool,),
            {
                "__init__": __init__,
                "forward": forward,
                "__doc__": func.__doc__,
                "__module__": func.__module__,
                "__annotations__": type_hints,
            }
        )
        
        # 创建工具实例
        tool_instance = ToolClass()
        
        # 手动设置参数配置
        for param_name, param in sig.parameters.items():
            if param_name in ['self', 'cls']:
                continue
            
            # 获取参数类型
            param_type = type_hints.get(param_name, str)
            
            # 确定是否必需
            required = param.default == inspect.Parameter.empty
            
            # 构建参数配置
            param_config = {
                "name": param_name,
                "type": _python_type_to_string(param_type),
                "description": f"参数: {param_name}",
                "required": required,
            }
            
            # 如果不是必需参数，添加默认值
            if not required:
                param_config["default"] = param.default
            
            # 创建参数对象
            param_obj = ToolParameter(**param_config)
            tool_instance.parameters.data[param_name] = param_obj.data
        
        # 设置启用状态
        if not enabled:
            tool_instance.enabled.data = False
        
        # 设置配置
        if config:
            tool_instance.config.data.update(config)
        
        # 创建包装函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """
            包装函数，使其可以直接调用
            注意：这会跳过工具的参数验证等特性
            要使用完整功能，请使用 func.tool(*args, **kwargs)
            """
            return func(*args, **kwargs)
        
        # 将工具实例附加到包装器
        wrapper.tool = tool_instance
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        
        return wrapper
    
    return decorator


def auto_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
):
    """
    智能工具装饰器，自动从函数签名和文档字符串提取所有信息
    
    支持多种文档格式：
    1. Google风格：
        Args:
            param1: 参数1描述
            param2: 参数2描述
            
    2. Numpy风格：
        Parameters
        ----------
        param1 : type
            参数1描述
        param2 : type
            参数2描述
            
    3. reStructuredText风格：
        :param param1: 参数1描述
        :type param1: str
        :param param2: 参数2描述
        :type param2: int
    """
    def decorator(func: Callable) -> Callable:
        # 自动提取信息
        tool_name = name or func.__name__
        func_description = description or _extract_function_description(func.__doc__) or f"工具: {func.__name__}"
        
        # 解析文档字符串
        doc_info = _parse_docstring(func.__doc__) if func.__doc__ else {}
        
        # 获取函数签名和类型提示
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # 准备参数配置
        param_configs = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self' or param_name == 'cls':
                continue
            
            # 获取参数类型（优先从类型提示，其次从文档）
            param_type = type_hints.get(param_name, str)
            type_from_doc = doc_info.get('types', {}).get(param_name)
            
            # 如果文档中有类型信息且类型提示是默认的str，使用文档中的类型
            if type_from_doc and param_type == str:
                # 尝试将文档中的类型字符串转换为Python类型
                type_map = {
                    'str': str, 'string': str,
                    'int': int, 'integer': int,
                    'float': float, 'number': float,
                    'bool': bool, 'boolean': bool,
                    'list': list, 'array': list,
                    'dict': dict, 'object': dict,
                }
                param_type = type_map.get(type_from_doc.lower(), str)
            
            # 获取参数描述
            param_desc = doc_info.get('params', {}).get(param_name, f"参数: {param_name}")
            
            # 确定是否必需（优先从签名，其次从文档）
            required_signature = param.default == inspect.Parameter.empty
            required_doc = doc_info.get('required', {}).get(param_name, required_signature)
            required = required_doc  # 文档中的required信息更优先
            
            # 构建参数配置
            param_config = {
                "name": param_name,
                "type": _python_type_to_string(param_type),
                "description": param_desc,
                "required": required,
            }
            
            # 如果不是必需参数，添加默认值
            if not required_signature:  # 注意：这里用signature的required，因为只有signature有默认值
                param_config["default"] = param.default
            
            # 添加可能的枚举值
            if param_name in doc_info.get('enums', {}):
                param_config["enum"] = doc_info['enums'][param_name]
            
            # 添加可能的取值范围
            if param_name in doc_info.get('ranges', {}):
                min_val, max_val = doc_info['ranges'][param_name]
                if min_val is not None:
                    param_config["min_value"] = min_val
                if max_val is not None:
                    param_config["max_value"] = max_val
            
            # 添加可能的正则模式
            if param_name in doc_info.get('patterns', {}):
                param_config["pattern"] = doc_info['patterns'][param_name]
            
            param_configs[param_name] = param_config
        
        # 应用@tool装饰器
        tool_decorator = tool(
            name=tool_name,
            description=func_description,
            **kwargs
        )
        
        decorated_func = tool_decorator(func)
        
        # 更新参数配置（因为@tool装饰器已经设置了基础参数，我们需要更新它们）
        for param_name, param_config in param_configs.items():
            if param_name in decorated_func.tool.parameters.data:
                # 更新现有的参数配置
                existing_config = decorated_func.tool.parameters.data[param_name]
                if isinstance(existing_config, dict):
                    # 直接更新字典
                    existing_config.update(param_config)
                else:
                    # 创建新的ToolParameter
                    param_obj = ToolParameter(**param_config)
                    decorated_func.tool.parameters.data[param_name] = param_obj.data
            else:
                # 创建新的参数
                param_obj = ToolParameter(**param_config)
                decorated_func.tool.parameters.data[param_name] = param_obj.data
        
        return decorated_func
    
    return decorator


# ==================== 类方法装饰器 ====================

def tool_method(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
):
    """
    类方法装饰器，将类方法转换为工具
    
    使用示例:
        class MathTools:
            @tool_method(name="add", description="加法计算")
            def add(self, a: float, b: float) -> float:
                return a + b
    """
    def decorator(method: Callable) -> Callable:
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            return method(self, *args, **kwargs)
        
        # 存储装饰器信息，供类装饰器使用
        wrapper._is_tool_method = True
        wrapper._tool_config = {
            "name": name or method.__name__,
            "description": description or _extract_function_description(method.__doc__) or f"工具: {method.__name__}",
            **kwargs
        }
        wrapper._method = method  # 保存原始方法
        
        return wrapper
    
    return decorator


# ==================== 类装饰器 ====================

def tool_class(
    class_name: Optional[str] = None,
    description: Optional[str] = None,
    version: str = "1.0.0",
    **kwargs
):
    """
    类装饰器，将类中的所有@tool_method方法转换为工具
    
    使用示例:
        @tool_class(description="数学工具集")
        class MathTools:
            @tool_method(name="add", description="加法")
            def add(self, a: float, b: float) -> float:
                return a + b
    """
    def decorator(cls: Type) -> Type:
        # 创建工具管理器
        from .tool_manager import ToolManager
        
        # 定义获取工具管理器的方法
        def get_tool_manager(self):
            """延迟初始化工具管理器"""
            if not hasattr(self, '_tool_manager') or self._tool_manager is None:
                self._tool_manager = ToolManager()
                
                # 注册所有工具方法
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if callable(attr) and hasattr(attr, '_is_tool_method'):
                        # 创建工具实例
                        tool_instance = self._create_tool_from_method(attr)
                        if tool_instance:
                            self._tool_manager.register_tool(tool_instance)
            
            return self._tool_manager
        
        def _create_tool_from_method(self, method_wrapper):
            """从方法包装器创建工具实例"""
            if not hasattr(method_wrapper, '_tool_config'):
                return None
            
            config = method_wrapper._tool_config
            original_method = method_wrapper._method if hasattr(method_wrapper, '_method') else method_wrapper
            
            # 创建工具类
            tool_class_name = f"Tool_{config['name'].title().replace('_', '')}"
            
            # 定义forward方法（绑定self）
            def forward(self_tool, *args, **kwargs):
                return original_method(self, *args, **kwargs)
            
            # 动态创建工具类
            ToolClass = type(
                tool_class_name,
                (BaseTool,),
                {
                    "__init__": lambda self_tool: BaseTool.__init__(self_tool, **config),
                    "forward": forward,
                    "__doc__": original_method.__doc__,
                }
            )
            
            return ToolClass()
        
        def get_tool(self, tool_name: str):
            """获取指定工具"""
            return self.get_tool_manager().get_tool(tool_name)
        
        def call_tool(self, tool_name: str, *args, **kwargs):
            """调用工具"""
            return self.get_tool_manager().call_tool(tool_name, *args, **kwargs)
        
        def get_all_tools(self):
            """获取所有工具"""
            return self.get_tool_manager().get_all_schemas()
        
        def get_openai_functions(self):
            """获取OpenAI函数格式"""
            return self.get_tool_manager().get_openai_functions()
        
        # 添加方法到类
        cls.get_tool_manager = get_tool_manager
        cls._create_tool_from_method = _create_tool_from_method
        cls.get_tool = get_tool
        cls.call_tool = call_tool
        cls.get_all_tools = get_all_tools
        cls.get_openai_functions = get_openai_functions
        
        # 添加类级别元数据
        cls._tool_class_metadata = {
            "class_name": class_name or cls.__name__,
            "description": description or cls.__doc__ or f"工具类: {cls.__name__}",
            "version": version,
            **kwargs
        }
        
        # 初始化实例属性
        original_init = cls.__init__ if hasattr(cls, '__init__') else lambda self: None
        
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._tool_manager = None
        
        cls.__init__ = new_init
        
        return cls
    
    return decorator


# ==================== 工具注册表 ====================

class ToolRegistry:
    """工具注册表（单例）"""
    _instance = None
    _tools = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._tools = {}
            self._initialized = True
    
    def register(self, func_or_tool, name: Optional[str] = None) -> Any:
        """注册工具"""
        if callable(func_or_tool) and hasattr(func_or_tool, 'tool'):
            # 已经用@tool装饰过的函数
            tool_instance = func_or_tool.tool
            tool_name = name or tool_instance.metadata.data['name']
            self._tools[tool_name] = tool_instance
            
        elif isinstance(func_or_tool, BaseTool):
            # BaseTool实例
            tool_name = name or func_or_tool.metadata.data['name']
            self._tools[tool_name] = func_or_tool
            
        elif callable(func_or_tool):
            # 普通函数，用@tool装饰它
            try:
                tool_decorator = tool(name=name or func_or_tool.__name__)
                decorated_func = tool_decorator(func_or_tool)
                self.register(decorated_func)
                return decorated_func
            except Exception as e:
                raise ValueError(f"无法将函数转换为工具: {e}")
        else:
            raise ValueError("只能注册@tool装饰的函数或BaseTool实例")
        
        return func_or_tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_all(self) -> List[str]:
        """列出所有工具"""
        return list(self._tools.keys())
    
    def clear(self):
        """清空注册表"""
        self._tools.clear()
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """获取所有工具"""
        return self._tools.copy()
    
    def get_tool_manager(self) -> 'ToolManager':
        """获取工具管理器"""
        from .tool_manager import ToolManager
        manager = ToolManager()
        for name, tool_instance in self._tools.items():
            manager.register_tool(tool_instance)
        return manager


# 创建全局注册表实例
registry = ToolRegistry()


def register_tool(name: Optional[str] = None):
    """
    注册工具到全局注册表的装饰器
    
    使用示例:
        @tool(name="calculator", description="计算器")
        @register_tool()
        def calculate(expression: str) -> float:
            return eval(expression)
    """
    def decorator(func_or_tool):
        registry.register(func_or_tool, name)
        return func_or_tool
    
    return decorator


# ==================== 预定义装饰器 ====================

# 简化装饰器别名
calculator = functools.partial(tool, category=ToolCategory.CALCULATION.value, tags=["math", "calculator"])
searcher = functools.partial(tool, category=ToolCategory.SEARCH.value, tags=["search", "web"])
converter = functools.partial(tool, category=ToolCategory.UTILITY.value, tags=["convert", "format"])
validator = functools.partial(tool, category=ToolCategory.UTILITY.value, tags=["validate", "check"])
generator = functools.partial(tool, category=ToolCategory.AI.value, tags=["generate", "create"])
file_processor = functools.partial(tool, category=ToolCategory.FILE.value, tags=["file", "io"])
database = functools.partial(tool, category=ToolCategory.DATABASE.value, tags=["database", "db"])
network = functools.partial(tool, category=ToolCategory.NETWORK.value, tags=["network", "api"])


# ==================== 异步支持 ====================

def async_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
):
    """
    异步工具装饰器
    
    使用示例:
        @async_tool(name="async_processor")
        async def process_data(data: str) -> str:
            await asyncio.sleep(1)
            return f"Processed: {data}"
    """
    def decorator(async_func: Callable) -> Callable:
        # 创建同步包装器
        @functools.wraps(async_func)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            return asyncio.run(async_func(*args, **kwargs))
        
        # 应用@tool装饰器
        tool_decorator = tool(
            name=name or async_func.__name__,
            description=description or _extract_function_description(async_func.__doc__) or f"异步工具: {async_func.__name__}",
            **kwargs
        )
        
        decorated_sync = tool_decorator(sync_wrapper)
        
        # 创建异步包装器
        @functools.wraps(async_func)
        async def async_wrapper(*args, **kwargs):
            return await async_func(*args, **kwargs)
        
        # 复制工具属性
        async_wrapper.tool = decorated_sync.tool
        
        return async_wrapper
    
    return decorator


def async_auto_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs
):
    """
    智能异步工具装饰器
    """
    def decorator(async_func: Callable) -> Callable:
        # 创建同步包装器用于分析
        @functools.wraps(async_func)
        def sync_wrapper(*args, **kwargs):
            import asyncio
            return asyncio.run(async_func(*args, **kwargs))
        
        # 复制文档字符串
        sync_wrapper.__doc__ = async_func.__doc__
        
        # 应用@auto_tool装饰器
        auto_tool_decorator = auto_tool(
            name=name or async_func.__name__,
            description=description or _extract_function_description(async_func.__doc__) or f"异步工具: {async_func.__name__}",
            **kwargs
        )
        
        decorated_sync = auto_tool_decorator(sync_wrapper)
        
        # 创建异步包装器
        @functools.wraps(async_func)
        async def async_wrapper(*args, **kwargs):
            return await async_func(*args, **kwargs)
        
        # 复制工具属性
        async_wrapper.tool = decorated_sync.tool
        
        return async_wrapper
    
    return decorator


# ==================== 导出 ====================

__all__ = [
    # 主要装饰器
    'tool',
    'auto_tool',
    'tool_method',
    'tool_class',
    'async_tool',
    'async_auto_tool',
    
    # 注册功能
    'ToolRegistry',
    'registry',
    'register_tool',
    
    # 预定义装饰器
    'calculator',
    'searcher',
    'converter',
    'validator',
    'generator',
    'file_processor',
    'database',
    'network',
    
    # 工具函数
    '_python_type_to_string',
    '_extract_function_description',
    '_parse_docstring',
]
