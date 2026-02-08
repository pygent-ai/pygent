from typing import Any, Dict, List, Type, get_type_hints, get_origin, get_args
import json
import yaml
from pathlib import Path
from pygent.common import PygentOperator


class PygentModule(PygentOperator):
    """Pygent模块基类，提供更丰富的功能"""
    
    def __init__(self):
        super().__init__()
        self._modules = {}
        self._init_modules()
    
    def _init_modules(self):
        """初始化所有子模块"""
        for name, value in self.__dict__.items():
            if isinstance(value, PygentOperator):
                self._modules[name] = value
    
    def add_module(self, name: str, module: 'PygentOperator') -> None:
        """添加子模块"""
        self._modules[name] = module
        setattr(self, name, module)
    
    def modules(self) -> List['PygentOperator']:
        """获取所有子模块"""
        return list(self._modules.values())
    
    def named_modules(self) -> List[tuple]:
        """获取所有命名子模块"""
        return list(self._modules.items())
    
    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典（包含子模块）"""
        state = super().state_dict()
        
        # 添加子模块状态
        modules_state = {}
        for name, module in self._modules.items():
            modules_state[name] = module.state_dict()
        
        if modules_state:
            state['_modules'] = modules_state
        
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """加载状态字典（包含子模块）"""
        # 分离模块状态
        modules_state = state_dict.pop('_modules', {})
        
        # 加载自身状态
        super().load_state_dict(state_dict, strict)
        
        # 加载模块状态
        for name, module_state in modules_state.items():
            if name in self._modules:
                self._modules[name].load_state_dict(module_state, strict)
            elif strict:
                raise ValueError(f"Module '{name}' not found")
    
    def save(self, path: str, format: str = 'json', include_metadata: bool = True) -> str:
        """保存（覆盖以包含模块信息）"""
        # 在元数据中添加模块信息
        save_path = super().save(path, format, include_metadata)
        
        if include_metadata:
            # 重新加载文件，添加模块信息
            path_obj = Path(path)
            with open(path_obj, 'r', encoding='utf-8') as f:
                if format == 'json':
                    save_data = json.load(f)
                elif format == 'yaml':
                    save_data = yaml.safe_load(f)
            
            # 添加模块信息
            save_data['module_names'] = list(self._modules.keys())
            
            # 重新保存
            with open(path_obj, 'w', encoding='utf-8') as f:
                if format == 'json':
                    json.dump(save_data, f, indent=2, default=str)
                elif format == 'yaml':
                    yaml.dump(save_data, f, default_flow_style=False)
        
        return save_path
