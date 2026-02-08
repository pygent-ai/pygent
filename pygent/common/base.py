from typing import Any, Dict, List, Tuple, Set, Union, TypeVar, Generic, get_type_hints, get_origin
from pathlib import Path
import pickle
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum
import json
import base64

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class PygentData:
    """基础数据类"""
    data: Any
    
    def __init__(self, data: Any = None):
        self.data = data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.data)})"
    
    def __str__(self) -> str:
        return str(self.data)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.data, default=str)
    
    def to_dict(self) -> Any:
        """转换为Python字典（如果可能）"""
        return self.data
    
    def copy(self) -> 'PygentData':
        """创建副本"""
        return self.__class__(self.data.copy() if hasattr(self.data, 'copy') else self.data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PygentData':
        """从JSON创建"""
        return cls(json.loads(json_str))


class PygentString(str, PygentData):
    """字符串类型，继承自 str，拥有所有 str 的方法"""

    _data_holder: List[str]  # mutable holder so .data can be set (e.g. load_state_dict)

    def __new__(cls, data: Any = ""):
        s = str(data) if data is not None else ""
        return str.__new__(cls, s)

    def __init__(self, data: Any = ""):
        # _data_holder keeps mutable ref so .data can be set (e.g. load_state_dict)
        object.__setattr__(self, "_data_holder", [str(self)])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> str:
        return getattr(self, "_data_holder", [str.__str__(self)])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = str(value) if value is not None else ""

    def __str__(self) -> str:
        return self.data

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"PygentString({repr(self.data)})"

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item):
        return item in self.data

    def upper(self) -> 'PygentString':
        return PygentString(self.data.upper())

    def lower(self) -> 'PygentString':
        return PygentString(self.data.lower())

    def strip(self, chars: str = None) -> 'PygentString':
        return PygentString(self.data.strip(chars) if chars is not None else self.data.strip())

    def replace(self, old: str, new: str, count: int = -1) -> 'PygentString':
        return PygentString(self.data.replace(old, new, count))

    def length(self) -> int:
        """获取字符串长度 (alias for len(self))"""
        return len(self.data)

    def contains(self, substring: str) -> bool:
        """检查是否包含子串 (alias for 'in')"""
        return substring in self.data


class PygentInt(int, PygentData):
    """整数类型，继承自 int，拥有所有 int 的方法"""

    _data_holder: List[int]

    def __new__(cls, data: Any = 0):
        return int.__new__(cls, int(data) if data is not None else 0)

    def __init__(self, data: Any = 0):
        object.__setattr__(self, "_data_holder", [int(self)])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> int:
        return getattr(self, "_data_holder", [int(self)])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = int(value) if value is not None else 0

    def __add__(self, other: Any) -> 'PygentInt':
        o = other.data if isinstance(other, PygentInt) else other
        return PygentInt(int(self) + o)

    def __sub__(self, other: Any) -> 'PygentInt':
        o = other.data if isinstance(other, PygentInt) else other
        return PygentInt(int(self) - o)

    def __mul__(self, other: Any) -> 'PygentInt':
        o = other.data if isinstance(other, PygentInt) else other
        return PygentInt(int(self) * o)

    def __truediv__(self, other: Any) -> 'PygentFloat':
        o = other.data if isinstance(other, PygentInt) else (other.data if isinstance(other, PygentFloat) else other)
        return PygentFloat(int(self) / o)

    def __floordiv__(self, other: Any) -> 'PygentInt':
        o = other.data if isinstance(other, PygentInt) else other
        return PygentInt(int(self) // o)

    def __mod__(self, other: Any) -> 'PygentInt':
        o = other.data if isinstance(other, PygentInt) else other
        return PygentInt(int(self) % o)

    def to_float(self) -> 'PygentFloat':
        """转换为浮点数"""
        return PygentFloat(float(int(self)))

    def to_binary(self) -> str:
        """转换为二进制字符串"""
        return bin(int(self))

    def to_hex(self) -> str:
        """转换为十六进制字符串"""
        return hex(int(self))

    def is_even(self) -> bool:
        """判断是否为偶数"""
        return int(self) % 2 == 0

    def is_odd(self) -> bool:
        """判断是否为奇数"""
        return int(self) % 2 != 0


class PygentFloat(float, PygentData):
    """浮点数类型，继承自 float，拥有所有 float 的方法"""

    _data_holder: List[float]

    def __new__(cls, data: Any = 0.0):
        return float.__new__(cls, float(data) if data is not None else 0.0)

    def __init__(self, data: Any = 0.0):
        object.__setattr__(self, "_data_holder", [float(self)])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> float:
        return getattr(self, "_data_holder", [float(self)])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = float(value) if value is not None else 0.0

    def __add__(self, other: Any) -> 'PygentFloat':
        o = other.data if isinstance(other, (PygentInt, PygentFloat)) else other
        return PygentFloat(float(self) + o)

    def __sub__(self, other: Any) -> 'PygentFloat':
        o = other.data if isinstance(other, (PygentInt, PygentFloat)) else other
        return PygentFloat(float(self) - o)

    def __mul__(self, other: Any) -> 'PygentFloat':
        o = other.data if isinstance(other, (PygentInt, PygentFloat)) else other
        return PygentFloat(float(self) * o)

    def __truediv__(self, other: Any) -> 'PygentFloat':
        o = other.data if isinstance(other, (PygentInt, PygentFloat)) else other
        return PygentFloat(float(self) / o)

    def to_int(self) -> 'PygentInt':
        """转换为整数"""
        return PygentInt(int(float(self)))

    def round(self, ndigits: int = 0) -> 'PygentFloat':
        """四舍五入"""
        return PygentFloat(round(float(self), ndigits))

    def ceil(self) -> 'PygentInt':
        """向上取整"""
        import math
        return PygentInt(math.ceil(float(self)))

    def floor(self) -> 'PygentInt':
        """向下取整"""
        import math
        return PygentInt(math.floor(float(self)))

    def is_integer(self) -> bool:
        """判断是否为整数"""
        return float(self).is_integer()


class PygentBool(PygentData):
    """布尔类型（Python 不允许子类化 bool，故仅继承 PygentData，实现与 bool 一致的方法）"""
    data: bool

    def __init__(self, data: Any = False):
        super().__init__(bool(data) if data is not None else False)

    def __bool__(self) -> bool:
        return self.data

    def __and__(self, other: Any) -> 'PygentBool':
        o = other.data if isinstance(other, PygentBool) else other
        return PygentBool(self.data and o)

    def __or__(self, other: Any) -> 'PygentBool':
        o = other.data if isinstance(other, PygentBool) else other
        return PygentBool(self.data or o)

    def __invert__(self) -> 'PygentBool':
        """逻辑非"""
        return PygentBool(not self.data)


class PygentList(list, PygentData, Generic[T]):
    """列表类型，继承自 list，拥有所有 list 的方法"""

    def __init__(self, data: List[T] = None):
        list.__init__(self, data if data is not None else [])
        PygentData.__init__(self, self)

    @property
    def data(self) -> list:
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self.clear()
        if value is not None:
            self.extend(value)

    def copy(self) -> 'PygentList[T]':
        """创建副本（返回 PygentList）"""
        return PygentList(list.copy(self))

    def filter(self, func) -> 'PygentList[T]':
        """过滤元素"""
        return PygentList([item for item in self if func(item)])

    def map(self, func) -> 'PygentList':
        """映射元素"""
        return PygentList([func(item) for item in self])


class PygentDict(dict, PygentData, Generic[K, V]):
    """字典类型，继承自 dict，拥有所有 dict 的方法"""

    def __init__(self, data: Dict[K, V] = None):
        dict.__init__(self, data if data is not None else {})
        PygentData.__init__(self, self)

    @property
    def data(self) -> dict:
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self.clear()
        if value is not None:
            self.update(value)

    def set(self, key: K, value: V) -> None:
        """设置值 (alias for self[key] = value)"""
        self[key] = value


class PygentTuple(tuple, PygentData):
    """元组类型，继承自 tuple，拥有所有 tuple 的方法"""

    _data_holder: List[Tuple]

    def __new__(cls, data: Tuple = None):
        t = tuple(data) if data is not None else ()
        return tuple.__new__(cls, t)

    def __init__(self, data: Tuple = None):
        t = tuple(data) if data is not None else ()
        object.__setattr__(self, "_data_holder", [t])
        PygentData.__init__(self, t)

    @property
    def data(self) -> tuple:
        return getattr(self, "_data_holder", [()])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = tuple(value) if value is not None else ()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item) -> bool:
        return item in self.data

    def count(self, item: Any) -> int:
        return self.data.count(item)

    def index(self, item: Any, start: int = 0, stop: int = None) -> int:
        return self.data.index(item, start, stop if stop is not None else len(self.data))


class PygentSet(set, PygentData):
    """集合类型，继承自 set，拥有所有 set 的方法"""

    def __init__(self, data: Set = None):
        set.__init__(self, data if data else set())
        PygentData.__init__(self, self)

    @property
    def data(self) -> set:
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self.clear()
        if value is not None:
            self.update(value)

    def union(self, *others: Union[Set, 'PygentSet']) -> 'PygentSet':
        """并集"""
        o = [x.data if isinstance(x, PygentSet) else x for x in others]
        return PygentSet(set.union(self, *o))

    def intersection(self, *others: Union[Set, 'PygentSet']) -> 'PygentSet':
        """交集"""
        o = [x.data if isinstance(x, PygentSet) else x for x in others]
        return PygentSet(set.intersection(self, *o))

    def difference(self, *others: Union[Set, 'PygentSet']) -> 'PygentSet':
        """差集"""
        o = [x.data if isinstance(x, PygentSet) else x for x in others]
        return PygentSet(set.difference(self, *o))

    def symmetric_difference(self, other: Union[Set, 'PygentSet']) -> 'PygentSet':
        """对称差集"""
        o = other.data if isinstance(other, PygentSet) else other
        return PygentSet(set.symmetric_difference(self, o))


class PygentBytes(bytes, PygentData):
    """字节类型，继承自 bytes，拥有所有 bytes 的方法"""

    _data_holder: List[bytes]

    def __new__(cls, data: Any = b""):
        return bytes.__new__(cls, bytes(data) if data is not None else b"")

    def __init__(self, data: Any = b""):
        object.__setattr__(self, "_data_holder", [bytes(self)])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> bytes:
        return getattr(self, "_data_holder", [bytes(self)])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = bytes(value) if value is not None else b""

    def __str__(self) -> str:
        return str(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, item) -> bool:
        return item in self.data

    def to_base64(self) -> str:
        """转换为Base64"""
        return base64.b64encode(self.data).decode('utf-8')

    @classmethod
    def from_base64(cls, base64_str: str) -> 'PygentBytes':
        """从Base64创建"""
        return cls(base64.b64decode(base64_str))

    def to_hex(self) -> str:
        """转换为十六进制"""
        return self.data.hex()

    @classmethod
    def from_hex(cls, hex_str: str) -> 'PygentBytes':
        """从十六进制创建"""
        return cls(bytes.fromhex(hex_str))

    def decode(self, encoding: str = 'utf-8', errors: str = 'strict') -> PygentString:
        """解码为字符串（签名与 bytes.decode 一致）"""
        return PygentString(self.data.decode(encoding, errors))


class PygentDateTime(datetime, PygentData):
    """日期时间类型，继承自 datetime，拥有所有 datetime 的方法"""

    _data_holder: List[datetime]

    def __new__(cls, data: datetime = None):
        dt = data if data is not None else datetime.now()
        return datetime.__new__(cls, dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond, dt.tzinfo)

    def __init__(self, data: datetime = None):
        dt = data if data is not None else datetime.now()
        object.__setattr__(self, "_data_holder", [dt])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> datetime:
        return getattr(self, "_data_holder", [self])[0]

    @data.setter
    def data(self, value: Any) -> None:
        if value is None:
            self._data_holder[0] = datetime.now()
        elif isinstance(value, datetime):
            self._data_holder[0] = value
        else:
            self._data_holder[0] = datetime.fromisoformat(str(value))

    @classmethod
    def now(cls, tz=None) -> 'PygentDateTime':
        """当前时间"""
        return cls(datetime.now(tz=tz))

    @classmethod
    def from_timestamp(cls, timestamp: float, tz=None) -> 'PygentDateTime':
        """从时间戳创建"""
        return cls(datetime.fromtimestamp(timestamp, tz=tz))

    @classmethod
    def from_isoformat(cls, iso_str: str) -> 'PygentDateTime':
        """从ISO格式字符串创建"""
        return cls(datetime.fromisoformat(iso_str))

    def to_timestamp(self) -> float:
        """转换为时间戳"""
        return self.data.timestamp()

    def to_isoformat(self) -> str:
        """转换为ISO格式字符串"""
        return self.data.isoformat()

    def format(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        """格式化输出"""
        return self.data.strftime(fmt)

    def date(self) -> 'PygentDate':
        """获取日期部分"""
        return PygentDate(self.data.date())

    def time(self) -> 'PygentTime':
        """获取时间部分"""
        return PygentTime(self.data.time())

    def replace(self, **kwargs) -> 'PygentDateTime':
        """替换部分时间"""
        return PygentDateTime(self.data.replace(**kwargs))


class PygentDate(date, PygentData):
    """日期类型，继承自 date，拥有所有 date 的方法"""

    _data_holder: List[date]

    def __new__(cls, data: date = None):
        d = data if data is not None else date.today()
        return date.__new__(cls, d.year, d.month, d.day)

    def __init__(self, data: date = None):
        d = data if data is not None else date.today()
        object.__setattr__(self, "_data_holder", [d])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> date:
        return getattr(self, "_data_holder", [self])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = date.fromisoformat(str(value)) if value is not None else date.today()

    @classmethod
    def today(cls) -> 'PygentDate':
        """今天"""
        return cls(date.today())

    @classmethod
    def from_isoformat(cls, iso_str: str) -> 'PygentDate':
        """从ISO格式字符串创建"""
        return cls(date.fromisoformat(iso_str))

    def to_isoformat(self) -> str:
        """转换为ISO格式"""
        return self.data.isoformat()

    def format(self, fmt: str = "%Y-%m-%d") -> str:
        """格式化输出"""
        return self.data.strftime(fmt)

    def __sub__(self, other: Any) -> Any:
        """计算日期差（天数）或 timedelta"""
        if isinstance(other, PygentDate):
            return (self.data - other.data).days
        return date.__sub__(self, other)


class PygentTime(time, PygentData):
    """时间类型，继承自 time，拥有所有 time 的方法"""

    _data_holder: List[time]

    def __new__(cls, data: time = None):
        t = data if data is not None else datetime.now().time()
        return time.__new__(cls, t.hour, t.minute, t.second, t.microsecond, t.tzinfo)

    def __init__(self, data: time = None):
        t = data if data is not None else datetime.now().time()
        object.__setattr__(self, "_data_holder", [t])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> time:
        return getattr(self, "_data_holder", [self])[0]

    @data.setter
    def data(self, value: Any) -> None:
        if value is None:
            self._data_holder[0] = datetime.now().time()
        elif isinstance(value, time):
            self._data_holder[0] = value
        else:
            self._data_holder[0] = time.fromisoformat(str(value))

    @classmethod
    def now(cls) -> 'PygentTime':
        """当前时间"""
        return cls(datetime.now().time())

    def format(self, fmt: str = "%H:%M:%S") -> str:
        """格式化输出"""
        return self.data.strftime(fmt)


class PygentDecimal(Decimal, PygentData):
    """高精度小数类型，继承自 Decimal，拥有所有 Decimal 的方法"""

    _data_holder: List[Decimal]

    def __new__(cls, data: Union[str, int, float, Decimal] = "0"):
        return Decimal.__new__(cls, str(data) if data is not None else "0")

    def __init__(self, data: Union[str, int, float, Decimal] = "0"):
        object.__setattr__(self, "_data_holder", [Decimal(self)])
        PygentData.__init__(self, self._data_holder[0])

    @property
    def data(self) -> Decimal:
        return getattr(self, "_data_holder", [Decimal(self)])[0]

    @data.setter
    def data(self, value: Any) -> None:
        self._data_holder[0] = Decimal(str(value)) if value is not None else Decimal("0")


class PygentEnum(PygentData):
    """枚举类型"""
    data: Enum
    
    def __init__(self, data: Enum):
        super().__init__(data)
    
    @property
    def name(self) -> str:
        """枚举名称"""
        return self.data.name
    
    @property
    def value(self) -> Any:
        """枚举值"""
        return self.data.value


class PygentNone(PygentData):
    """None类型"""
    data: None
    
    def __init__(self):
        super().__init__(None)
    
    def is_none(self) -> bool:
        """判断是否为None"""
        return True
    
    def __bool__(self) -> bool:
        return False


class PygentAny(PygentData, Generic[T]):
    """任意类型包装器"""
    data: T
    
    def __init__(self, data: T):
        super().__init__(data)
    
    def get_type(self) -> type:
        """获取数据类型"""
        return type(self.data)
    
    def isinstance(self, type_check: type) -> bool:
        """类型检查"""
        return isinstance(self.data, type_check)


# 工厂函数
def create_pygent_data(data: Any) -> PygentData:
    """根据数据类型自动创建对应的PygentData对象"""
    if data is None:
        return PygentNone()
    elif isinstance(data, bool):
        return PygentBool(data)
    elif isinstance(data, str):
        return PygentString(data)
    elif isinstance(data, int):
        return PygentInt(data)
    elif isinstance(data, float):
        return PygentFloat(data)
    elif isinstance(data, list):
        return PygentList(data)
    elif isinstance(data, dict):
        return PygentDict(data)
    elif isinstance(data, tuple):
        return PygentTuple(data)
    elif isinstance(data, set):
        return PygentSet(data)
    elif isinstance(data, bytes):
        return PygentBytes(data)
    elif isinstance(data, datetime):
        return PygentDateTime(data)
    elif isinstance(data, date):
        return PygentDate(data)
    elif isinstance(data, time):
        return PygentTime(data)
    elif isinstance(data, Decimal):
        return PygentDecimal(data)
    elif isinstance(data, Enum):
        return PygentEnum(data)
    else:
        return PygentAny(data)


class PygentOperator:
    """Pygent数据操作器，支持参数的保存和加载"""

    def __init__(self):
        self._pygent_fields = {}
        self._init_fields()

    def to(self, *args, **kwargs) -> 'PygentOperator':
        """兼容接口，返回 self 便于链式调用"""
        return self

    def train(self, mode: bool = True) -> 'PygentOperator':
        """兼容接口，返回 self 便于链式调用"""
        return self

    def eval(self) -> 'PygentOperator':
        """兼容接口，返回 self 便于链式调用"""
        return self

    def _init_fields(self):
        """初始化所有PygentData字段"""
        # 获取类的类型注解
        type_hints = get_type_hints(self.__class__)

        for field_name, field_type in type_hints.items():
            # 检查是否是PygentData类型
            if self._is_pygent_data_type(field_type):
                # 如果字段还没有被初始化，使用默认值
                if not hasattr(self, field_name):
                    default_value = self._get_default_value(field_type)
                    setattr(self, field_name, default_value)

                # 记录字段信息
                self._pygent_fields[field_name] = {
                    'type': field_type,
                    'value': getattr(self, field_name)
                }

    def _is_pygent_data_type(self, field_type) -> bool:
        """检查是否为PygentData类型（含子类，如 PygentInt, PygentString）"""
        try:
            origin = get_origin(field_type)
            if origin is not None:
                if isinstance(origin, type) and issubclass(origin, PygentData):
                    return True
                return False
            if isinstance(field_type, type) and issubclass(field_type, PygentData):
                return True
            return False
        except TypeError:
            return False

    def _get_default_value(self, field_type):
        """获取字段类型的默认值"""
        origin = get_origin(field_type)

        if origin:
            # 处理泛型类型
            if origin.__name__ == 'PygentList':
                return origin()
            elif origin.__name__ == 'PygentDict':
                return origin()
            else:
                return origin()
        else:
            # 直接实例化
            return field_type() if hasattr(field_type, '__call__') else None

    def state_dict(self) -> Dict[str, Any]:
        """获取所有PygentData字段的状态字典"""
        state = {}

        for field_name in self._pygent_fields:
            field_value = getattr(self, field_name, None)
            if field_value is not None:
                # 如果是PygentData对象，获取其数据
                if isinstance(field_value, PygentData):
                    state[field_name] = {
                        'data': field_value.data,
                        'type': field_value.__class__.__name__
                    }
                else:
                    state[field_name] = field_value

        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """
        加载状态字典

        Args:
            state_dict: 状态字典
            strict: 是否严格模式，如果为True则要求字段完全匹配
        """
        # 获取当前类的所有PygentData字段
        expected_fields = set(self._pygent_fields.keys())
        loaded_fields = set(state_dict.keys())

        if strict:
            # 检查字段是否匹配
            if expected_fields != loaded_fields:
                missing = expected_fields - loaded_fields
                unexpected = loaded_fields - expected_fields

                error_msgs = []
                if missing:
                    error_msgs.append(f"Missing fields: {missing}")
                if unexpected:
                    error_msgs.append(f"Unexpected fields: {unexpected}")

                if error_msgs:
                    raise ValueError("State dict does not match: " + ", ".join(error_msgs))

        # 加载字段
        for field_name, field_value in state_dict.items():
            if hasattr(self, field_name):
                current_value = getattr(self, field_name)

                if isinstance(current_value, PygentData):
                    # 如果是PygentData对象，加载数据
                    if isinstance(field_value, dict) and 'data' in field_value:
                        # 完整的状态字典格式
                        current_value.data = field_value['data']
                    else:
                        # 简化的数据格式
                        current_value.data = field_value
                else:
                    # 直接设置值
                    setattr(self, field_name, field_value)
            else:
                if strict:
                    raise ValueError(f"Field '{field_name}' not found in operator")

    def save(self, path: str, format: str = 'json', include_metadata: bool = True) -> str:
        """
        保存所有PygentData字段到文件

        Args:
            path: 文件路径
            format: 保存格式，支持 'json', 'pickle', 'yaml'
            include_metadata: 是否包含元数据（类型信息等）

        Returns:
            保存的文件路径
        """
        path_obj = Path(path)

        # 准备保存数据
        if include_metadata:
            save_data = {
                'version': '1.0',
                'operator_class': self.__class__.__name__,
                'timestamp': PygentDateTime.now().to_isoformat(),
                'checksum': self._calculate_checksum(),
                'state_dict': self.state_dict()
            }
        else:
            save_data = self.state_dict()

        # 根据格式保存
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(path_obj, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)

        elif format == 'pickle':
            with open(path_obj, 'wb') as f:
                pickle.dump(save_data, f)

        elif format == 'yaml':
            import yaml
            with open(path_obj, 'w', encoding='utf-8') as f:
                yaml.dump(save_data, f, default_flow_style=False)

        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Saved operator state to: {path_obj.absolute()}")
        return str(path_obj.absolute())

    def load(self, path: str, format: str = 'auto', strict: bool = True) -> None:
        """
        从文件加载PygentData字段

        Args:
            path: 文件路径
            format: 文件格式，'auto'会自动检测
            strict: 是否严格模式
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # 自动检测格式
        if format == 'auto':
            format = path_obj.suffix.lower()[1:]  # 去掉点号
            if format not in ['json', 'pickle', 'yaml']:
                format = 'pickle'  # 默认使用pickle

        # 加载数据
        if format == 'json':
            with open(path_obj, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

        elif format == 'pickle':
            with open(path_obj, 'rb') as f:
                loaded_data = pickle.load(f)

        elif format == 'yaml':
            import yaml
            with open(path_obj, 'r', encoding='utf-8') as f:
                loaded_data = yaml.safe_load(f)

        else:
            raise ValueError(f"Unsupported format: {format}")

        # 提取状态字典
        if isinstance(loaded_data, dict) and 'state_dict' in loaded_data:
            # 包含元数据的格式
            state_dict = loaded_data['state_dict']

            # 验证checksum（可选）
            if strict and 'checksum' in loaded_data:
                current_checksum = self._calculate_checksum()
                if loaded_data['checksum'] != current_checksum:
                    print(f"Warning: Checksum mismatch. File may be from a different configuration.")

            # 输出加载信息
            if 'operator_class' in loaded_data:
                print(f"Loading {loaded_data['operator_class']} from {path}")
            if 'timestamp' in loaded_data:
                print(f"Saved at: {loaded_data['timestamp']}")
        else:
            # 简化的状态字典格式
            state_dict = loaded_data

        # 加载状态
        self.load_state_dict(state_dict, strict=strict)
        print(f"Successfully loaded state from: {path}")

    def _calculate_checksum(self) -> str:
        """计算状态字典的校验和"""
        import hashlib
        import json

        state_str = json.dumps(self.state_dict(), sort_keys=True, default=str)
        return hashlib.md5(state_str.encode()).hexdigest()

    def parameters(self) -> Dict[str, Any]:
        """获取所有参数（PyTorch风格）"""
        return {k: v.data for k, v in self._get_pygent_fields().items()}

    def named_parameters(self) -> List[tuple]:
        """获取所有命名参数"""
        return [(k, v.data) for k, v in self._get_pygent_fields().items()]

    def _get_pygent_fields(self) -> Dict[str, PygentData]:
        """获取所有PygentData字段"""
        fields = {}
        for field_name in self._pygent_fields:
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, PygentData):
                fields[field_name] = field_value
        return fields

    def __repr__(self) -> str:
        """字符串表示"""
        class_name = self.__class__.__name__
        fields_info = []

        for field_name in self._pygent_fields:
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, PygentData):
                fields_info.append(f"{field_name}={repr(field_value.data)}")
            else:
                fields_info.append(f"{field_name}={repr(field_value)}")

        fields_str = ", ".join(fields_info)
        return f"{class_name}({fields_str})"

