from __future__ import annotations

import base64
import copy as copy_module
import hashlib
import importlib
import json
import pickle
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, Iterator, List, Set, Tuple, TypeVar, Union, get_origin, get_type_hints


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class PygentData(Generic[T]):
    """Base value container used by Pygent stateful objects."""

    data: T

    def __init__(self, data: T = None):
        self.data = data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data!r})"

    def __str__(self) -> str:
        return str(self.data)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __eq__(self, other: Any) -> bool:
        return self.data == _unwrap_data(other)

    def __hash__(self) -> int:
        try:
            return hash(self.data)
        except TypeError:
            return id(self)

    def copy(self) -> "PygentData[T]":
        return self.__class__(copy_module.deepcopy(self.data))

    def to_json(self) -> str:
        return json.dumps(_to_json_value(self.data), ensure_ascii=False)

    def to_dict(self) -> Any:
        return _to_plain_value(self.data)

    @classmethod
    def from_json(cls, json_str: str) -> "PygentData":
        return cls(_from_json_value(json.loads(json_str)))


class PygentString(PygentData[str]):
    """String value object with explicit, mutable state."""

    data: str

    def __init__(self, data: Any = ""):
        super().__init__("" if data is None else str(_unwrap_data(data)))

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __contains__(self, item: Any) -> bool:
        return item in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other: Any) -> "PygentString":
        return PygentString(self.data + str(_unwrap_data(other)))

    def __radd__(self, other: Any) -> "PygentString":
        return PygentString(str(_unwrap_data(other)) + self.data)

    def __lt__(self, other: Any) -> bool:
        return self.data < str(_unwrap_data(other))

    def __le__(self, other: Any) -> bool:
        return self.data <= str(_unwrap_data(other))

    def __gt__(self, other: Any) -> bool:
        return self.data > str(_unwrap_data(other))

    def __ge__(self, other: Any) -> bool:
        return self.data >= str(_unwrap_data(other))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.data, name)

    def upper(self) -> "PygentString":
        return PygentString(self.data.upper())

    def lower(self) -> "PygentString":
        return PygentString(self.data.lower())

    def strip(self, chars: str | None = None) -> "PygentString":
        return PygentString(self.data.strip(chars) if chars is not None else self.data.strip())

    def replace(self, old: str, new: str, count: int = -1) -> "PygentString":
        return PygentString(self.data.replace(old, new, count))

    def length(self) -> int:
        return len(self.data)

    def contains(self, substring: str) -> bool:
        return substring in self.data


class PygentInt(PygentData[int]):
    data: int

    def __init__(self, data: Any = 0):
        super().__init__(0 if data is None else int(_unwrap_data(data)))

    def __int__(self) -> int:
        return self.data

    def __float__(self) -> float:
        return float(self.data)

    def __index__(self) -> int:
        return self.data

    def _number(self, other: Any) -> Any:
        return _unwrap_data(other)

    def __add__(self, other: Any) -> "PygentInt":
        return PygentInt(self.data + self._number(other))

    def __radd__(self, other: Any) -> "PygentInt":
        return PygentInt(self._number(other) + self.data)

    def __sub__(self, other: Any) -> "PygentInt":
        return PygentInt(self.data - self._number(other))

    def __rsub__(self, other: Any) -> "PygentInt":
        return PygentInt(self._number(other) - self.data)

    def __mul__(self, other: Any) -> "PygentInt":
        return PygentInt(self.data * self._number(other))

    def __rmul__(self, other: Any) -> "PygentInt":
        return PygentInt(self._number(other) * self.data)

    def __truediv__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self.data / self._number(other))

    def __floordiv__(self, other: Any) -> "PygentInt":
        return PygentInt(self.data // self._number(other))

    def __mod__(self, other: Any) -> "PygentInt":
        return PygentInt(self.data % self._number(other))

    def __lt__(self, other: Any) -> bool:
        return self.data < self._number(other)

    def __le__(self, other: Any) -> bool:
        return self.data <= self._number(other)

    def __gt__(self, other: Any) -> bool:
        return self.data > self._number(other)

    def __ge__(self, other: Any) -> bool:
        return self.data >= self._number(other)

    def to_float(self) -> "PygentFloat":
        return PygentFloat(float(self.data))

    def to_binary(self) -> str:
        return bin(self.data)

    def to_hex(self) -> str:
        return hex(self.data)

    def is_even(self) -> bool:
        return self.data % 2 == 0

    def is_odd(self) -> bool:
        return self.data % 2 != 0


class PygentFloat(PygentData[float]):
    data: float

    def __init__(self, data: Any = 0.0):
        super().__init__(0.0 if data is None else float(_unwrap_data(data)))

    def __float__(self) -> float:
        return self.data

    def _number(self, other: Any) -> Any:
        return _unwrap_data(other)

    def __add__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self.data + self._number(other))

    def __radd__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self._number(other) + self.data)

    def __sub__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self.data - self._number(other))

    def __rsub__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self._number(other) - self.data)

    def __mul__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self.data * self._number(other))

    def __rmul__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self._number(other) * self.data)

    def __truediv__(self, other: Any) -> "PygentFloat":
        return PygentFloat(self.data / self._number(other))

    def __lt__(self, other: Any) -> bool:
        return self.data < self._number(other)

    def __le__(self, other: Any) -> bool:
        return self.data <= self._number(other)

    def __gt__(self, other: Any) -> bool:
        return self.data > self._number(other)

    def __ge__(self, other: Any) -> bool:
        return self.data >= self._number(other)

    def to_int(self) -> PygentInt:
        return PygentInt(int(self.data))

    def round(self, ndigits: int = 0) -> "PygentFloat":
        return PygentFloat(round(self.data, ndigits))

    def ceil(self) -> PygentInt:
        import math

        return PygentInt(math.ceil(self.data))

    def floor(self) -> PygentInt:
        import math

        return PygentInt(math.floor(self.data))

    def is_integer(self) -> bool:
        return self.data.is_integer()


class PygentBool(PygentData[bool]):
    data: bool

    def __init__(self, data: Any = False):
        super().__init__(False if data is None else bool(_unwrap_data(data)))

    def __bool__(self) -> bool:
        return self.data

    def __and__(self, other: Any) -> "PygentBool":
        return PygentBool(self.data and bool(_unwrap_data(other)))

    def __or__(self, other: Any) -> "PygentBool":
        return PygentBool(self.data or bool(_unwrap_data(other)))

    def __invert__(self) -> "PygentBool":
        return PygentBool(not self.data)


class PygentList(list, PygentData[List[T]], Generic[T]):
    def __init__(self, data: Iterable[T] | None = None):
        list.__init__(self, [] if data is None else list(_unwrap_data(data)))

    @property
    def data(self) -> list:
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self.clear()
        if value is not None:
            self.extend(_unwrap_data(value))

    def copy(self) -> "PygentList[T]":
        return PygentList(copy_module.deepcopy(list(self)))

    def __bool__(self) -> bool:
        return len(self) > 0

    def filter(self, func) -> "PygentList[T]":
        return PygentList(item for item in self if func(item))

    def map(self, func) -> "PygentList":
        return PygentList(func(item) for item in self)


class PygentDict(dict, PygentData[Dict[K, V]], Generic[K, V]):
    def __init__(self, data: Dict[K, V] | None = None):
        dict.__init__(self, {} if data is None else dict(_unwrap_data(data)))

    @property
    def data(self) -> dict:
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self.clear()
        if value is not None:
            self.update(_unwrap_data(value))

    def copy(self) -> "PygentDict[K, V]":
        return PygentDict(copy_module.deepcopy(dict(self)))

    def __bool__(self) -> bool:
        return len(self) > 0

    def set(self, key: K, value: V) -> None:
        self[key] = value


class PygentTuple(PygentData[Tuple[Any, ...]]):
    data: tuple

    def __init__(self, data: Iterable[Any] | None = None):
        super().__init__(tuple(data) if data is not None else ())

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data)

    def __contains__(self, item: Any) -> bool:
        return item in self.data

    def __getitem__(self, index):
        return self.data[index]

    def count(self, item: Any) -> int:
        return self.data.count(item)

    def index(self, item: Any, start: int = 0, stop: int | None = None) -> int:
        return self.data.index(item, start, len(self.data) if stop is None else stop)


class PygentSet(set, PygentData[Set[Any]]):
    def __init__(self, data: Iterable[Any] | None = None):
        set.__init__(self, set() if data is None else set(_unwrap_data(data)))

    @property
    def data(self) -> set:
        return self

    @data.setter
    def data(self, value: Any) -> None:
        if value is self:
            return
        self.clear()
        if value is not None:
            self.update(_unwrap_data(value))

    def copy(self) -> "PygentSet":
        return PygentSet(copy_module.deepcopy(set(self)))

    def __bool__(self) -> bool:
        return len(self) > 0

    def union(self, *others: Union[Set, "PygentSet"]) -> "PygentSet":
        return PygentSet(set.union(self, *[_unwrap_data(x) for x in others]))

    def intersection(self, *others: Union[Set, "PygentSet"]) -> "PygentSet":
        return PygentSet(set.intersection(self, *[_unwrap_data(x) for x in others]))

    def difference(self, *others: Union[Set, "PygentSet"]) -> "PygentSet":
        return PygentSet(set.difference(self, *[_unwrap_data(x) for x in others]))

    def symmetric_difference(self, other: Union[Set, "PygentSet"]) -> "PygentSet":
        return PygentSet(set.symmetric_difference(self, _unwrap_data(other)))


class PygentBytes(PygentData[bytes]):
    data: bytes

    def __init__(self, data: Any = b""):
        value = _unwrap_data(data)
        if value is None:
            value = b""
        if isinstance(value, str):
            value = value.encode()
        super().__init__(bytes(value))

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[int]:
        return iter(self.data)

    def __contains__(self, item: Any) -> bool:
        return item in self.data

    def __getitem__(self, key):
        return self.data[key]

    def to_base64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    @classmethod
    def from_base64(cls, base64_str: str) -> "PygentBytes":
        return cls(base64.b64decode(base64_str))

    def to_hex(self) -> str:
        return self.data.hex()

    @classmethod
    def from_hex(cls, hex_str: str) -> "PygentBytes":
        return cls(bytes.fromhex(hex_str))

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> PygentString:
        return PygentString(self.data.decode(encoding, errors))


class PygentDateTime(PygentData[datetime]):
    data: datetime

    def __init__(self, data: datetime | str | None = None):
        super().__init__(_coerce_datetime(data))

    @classmethod
    def now(cls, tz=None) -> "PygentDateTime":
        return cls(datetime.now(tz=tz))

    @classmethod
    def from_timestamp(cls, timestamp: float, tz=None) -> "PygentDateTime":
        return cls(datetime.fromtimestamp(timestamp, tz=tz))

    @classmethod
    def from_isoformat(cls, iso_str: str) -> "PygentDateTime":
        return cls(datetime.fromisoformat(iso_str))

    def to_timestamp(self) -> float:
        return self.data.timestamp()

    def to_isoformat(self) -> str:
        return self.data.isoformat()

    def format(self, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
        return self.data.strftime(fmt)

    def date(self) -> "PygentDate":
        return PygentDate(self.data.date())

    def time(self) -> "PygentTime":
        return PygentTime(self.data.time())

    def replace(self, **kwargs) -> "PygentDateTime":
        return PygentDateTime(self.data.replace(**kwargs))


class PygentDate(PygentData[date]):
    data: date

    def __init__(self, data: date | str | None = None):
        super().__init__(_coerce_date(data))

    @classmethod
    def today(cls) -> "PygentDate":
        return cls(date.today())

    @classmethod
    def from_isoformat(cls, iso_str: str) -> "PygentDate":
        return cls(date.fromisoformat(iso_str))

    def to_isoformat(self) -> str:
        return self.data.isoformat()

    def format(self, fmt: str = "%Y-%m-%d") -> str:
        return self.data.strftime(fmt)

    def __sub__(self, other: Any) -> Any:
        other_value = _unwrap_data(other)
        if isinstance(other_value, date):
            return (self.data - other_value).days
        return self.data - other_value


class PygentTime(PygentData[time]):
    data: time

    def __init__(self, data: time | str | None = None):
        super().__init__(_coerce_time(data))

    @classmethod
    def now(cls) -> "PygentTime":
        return cls(datetime.now().time())

    def format(self, fmt: str = "%H:%M:%S") -> str:
        return self.data.strftime(fmt)


class PygentDecimal(PygentData[Decimal]):
    data: Decimal

    def __init__(self, data: Union[str, int, float, Decimal] = "0"):
        super().__init__(Decimal("0") if data is None else Decimal(str(_unwrap_data(data))))


class PygentEnum(PygentData[Enum]):
    data: Enum

    @property
    def name(self) -> str:
        return self.data.name

    @property
    def value(self) -> Any:
        return self.data.value


class PygentNone(PygentData[None]):
    data: None

    def __init__(self):
        super().__init__(None)

    def is_none(self) -> bool:
        return True

    def __bool__(self) -> bool:
        return False


class PygentAny(PygentData[T], Generic[T]):
    def get_type(self) -> type:
        return type(self.data)

    def isinstance(self, type_check: type) -> bool:
        return isinstance(self.data, type_check)


def create_pygent_data(data: Any) -> PygentData:
    if isinstance(data, PygentData):
        return data
    if data is None:
        return PygentNone()
    if isinstance(data, bool):
        return PygentBool(data)
    if isinstance(data, str):
        return PygentString(data)
    if isinstance(data, int):
        return PygentInt(data)
    if isinstance(data, float):
        return PygentFloat(data)
    if isinstance(data, list):
        return PygentList(data)
    if isinstance(data, dict):
        return PygentDict(data)
    if isinstance(data, tuple):
        return PygentTuple(data)
    if isinstance(data, set):
        return PygentSet(data)
    if isinstance(data, bytes):
        return PygentBytes(data)
    if isinstance(data, datetime):
        return PygentDateTime(data)
    if isinstance(data, date):
        return PygentDate(data)
    if isinstance(data, time):
        return PygentTime(data)
    if isinstance(data, Decimal):
        return PygentDecimal(data)
    if isinstance(data, Enum):
        return PygentEnum(data)
    return PygentAny(data)


class PygentOperator:
    """Stateful object that discovers annotated PygentData fields."""

    def __init__(self):
        self._pygent_fields: Dict[str, type[PygentData]] = {}
        self._init_fields()

    def to(self, *args, **kwargs) -> "PygentOperator":
        return self

    def train(self, mode: bool = True) -> "PygentOperator":
        return self

    def eval(self) -> "PygentOperator":
        return self

    def _init_fields(self) -> None:
        for field_name, field_type in self._field_annotations().items():
            data_type = self._resolve_pygent_data_type(field_type)
            if data_type is None:
                continue
            if not hasattr(self, field_name):
                setattr(self, field_name, data_type())
            self._pygent_fields[field_name] = data_type

    def _field_annotations(self) -> Dict[str, Any]:
        annotations: Dict[str, Any] = {}
        for cls in reversed(self.__class__.__mro__):
            if cls is object:
                continue
            try:
                annotations.update(get_type_hints(cls))
            except Exception:
                annotations.update(getattr(cls, "__annotations__", {}))
        return annotations

    @staticmethod
    def _resolve_pygent_data_type(field_type: Any) -> type[PygentData] | None:
        origin = get_origin(field_type)
        candidate = origin or field_type
        if isinstance(candidate, type) and issubclass(candidate, PygentData):
            return candidate
        return None

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for field_name in self._pygent_fields:
            field_value = getattr(self, field_name, None)
            if isinstance(field_value, PygentData):
                state[field_name] = {
                    "type": field_value.__class__.__name__,
                    "data": _to_json_value(field_value.data),
                }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        expected_fields = set(self._pygent_fields)
        loaded_fields = set(state_dict)
        if strict and expected_fields != loaded_fields:
            messages = []
            missing = expected_fields - loaded_fields
            unexpected = loaded_fields - expected_fields
            if missing:
                messages.append(f"Missing fields: {sorted(missing)}")
            if unexpected:
                messages.append(f"Unexpected fields: {sorted(unexpected)}")
            raise ValueError("State dict does not match: " + ", ".join(messages))

        for field_name, serialized_value in state_dict.items():
            if field_name not in self._pygent_fields:
                if strict:
                    raise ValueError(f"Field '{field_name}' not found in operator")
                continue

            current_value = getattr(self, field_name, None)
            payload = serialized_value.get("data") if isinstance(serialized_value, dict) and "data" in serialized_value else serialized_value
            value = _from_json_value(payload)

            if isinstance(current_value, PygentData):
                current_value.data = value
            else:
                setattr(self, field_name, self._pygent_fields[field_name](value))

    def save(self, path: str, format: str = "json", include_metadata: bool = True) -> str:
        path_obj = Path(path)
        state = self.state_dict()
        save_data = {
            "version": "2.0",
            "operator_class": self.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "checksum": self._calculate_checksum(state),
            "state_dict": state,
        } if include_metadata else state

        path_obj.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            with open(path_obj, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
        elif format == "pickle":
            with open(path_obj, "wb") as f:
                pickle.dump(save_data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        return str(path_obj.resolve())

    def load(self, path: str, format: str = "auto", strict: bool = True) -> None:
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        resolved_format = self._resolve_format(path_obj, format)
        if resolved_format == "json":
            with open(path_obj, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
        elif resolved_format == "pickle":
            with open(path_obj, "rb") as f:
                loaded_data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")

        state_dict = loaded_data["state_dict"] if isinstance(loaded_data, dict) and "state_dict" in loaded_data else loaded_data
        if strict and isinstance(loaded_data, dict) and "checksum" in loaded_data:
            expected = loaded_data["checksum"]
            actual = self._calculate_checksum(state_dict)
            if expected != actual:
                raise ValueError("State checksum mismatch")
        self.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def _resolve_format(path: Path, format: str) -> str:
        if format != "auto":
            return format
        suffix = path.suffix.lower()
        if suffix == ".json":
            return "json"
        if suffix in {".pkl", ".pickle"}:
            return "pickle"
        raise ValueError(f"Unsupported format: auto could not infer format from '{path.suffix}'")

    def _calculate_checksum(self, state: Dict[str, Any] | None = None) -> str:
        state_str = json.dumps(state or self.state_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(state_str.encode("utf-8")).hexdigest()

    def parameters(self) -> Dict[str, Any]:
        return {name: value.data for name, value in self._get_pygent_fields().items()}

    def named_parameters(self) -> List[tuple]:
        return [(name, value.data) for name, value in self._get_pygent_fields().items()]

    def _get_pygent_fields(self) -> Dict[str, PygentData]:
        return {
            field_name: field_value
            for field_name in self._pygent_fields
            if isinstance((field_value := getattr(self, field_name, None)), PygentData)
        }

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name).data!r}" for name in self._get_pygent_fields())
        return f"{self.__class__.__name__}({fields})"


def _unwrap_data(value: Any) -> Any:
    return value.data if isinstance(value, PygentData) else value


def _to_plain_value(value: Any) -> Any:
    value = _unwrap_data(value)
    if isinstance(value, dict):
        return {key: _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_value(item) for item in value)
    if isinstance(value, set):
        return {_to_plain_value(item) for item in value}
    return value


def _to_json_value(value: Any) -> Any:
    if isinstance(value, PygentData) and not isinstance(value, _JSON_NATIVE_PYGENT_TYPES):
        return _to_pygent_object_value(value)

    value = _unwrap_data(value)
    if isinstance(value, dict):
        return {str(_unwrap_data(key)): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, tuple):
        return {"__pygent_type__": "tuple", "items": [_to_json_value(item) for item in value]}
    if isinstance(value, set):
        return {"__pygent_type__": "set", "items": [_to_json_value(item) for item in sorted(value, key=repr)]}
    if isinstance(value, bytes):
        return {"__pygent_type__": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, datetime):
        return {"__pygent_type__": "datetime", "iso": value.isoformat()}
    if isinstance(value, date):
        return {"__pygent_type__": "date", "iso": value.isoformat()}
    if isinstance(value, time):
        return {"__pygent_type__": "time", "iso": value.isoformat()}
    if isinstance(value, Decimal):
        return {"__pygent_type__": "decimal", "value": str(value)}
    if isinstance(value, Enum):
        return {
            "__pygent_type__": "enum",
            "module": value.__class__.__module__,
            "class": value.__class__.__qualname__,
            "name": value.name,
            "value": _to_json_value(value.value),
        }
    return value


def _from_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_json_value(item) for item in value]
    if not isinstance(value, dict):
        return value

    marker = value.get("__pygent_type__")
    if marker == "tuple":
        return tuple(_from_json_value(item) for item in value["items"])
    if marker == "set":
        return set(_from_json_value(item) for item in value["items"])
    if marker == "bytes":
        return base64.b64decode(value["base64"])
    if marker == "datetime":
        return datetime.fromisoformat(value["iso"])
    if marker == "date":
        return date.fromisoformat(value["iso"])
    if marker == "time":
        return time.fromisoformat(value["iso"])
    if marker == "decimal":
        return Decimal(value["value"])
    if marker == "enum":
        return _from_json_value(value["value"])
    if marker == "pygent_object":
        return _from_pygent_object_value(value)
    return {key: _from_json_value(item) for key, item in value.items()}


_JSON_NATIVE_PYGENT_TYPES = (
    PygentString,
    PygentInt,
    PygentFloat,
    PygentBool,
    PygentList,
    PygentDict,
    PygentTuple,
    PygentSet,
    PygentBytes,
    PygentDateTime,
    PygentDate,
    PygentTime,
    PygentDecimal,
    PygentEnum,
    PygentNone,
    PygentAny,
)


def _to_pygent_object_value(value: PygentData) -> Dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    else:
        payload = value.data
    return {
        "__pygent_type__": "pygent_object",
        "module": value.__class__.__module__,
        "class": value.__class__.__qualname__,
        "payload": _to_json_value(payload),
    }


def _from_pygent_object_value(value: Dict[str, Any]) -> Any:
    payload = _from_json_value(value["payload"])
    cls = _import_qualified(value["module"], value["class"])

    try:
        from pygent.message.base import BaseMessage

        if isinstance(cls, type) and issubclass(cls, BaseMessage):
            return BaseMessage.from_serialized_dict(payload)
    except Exception:
        pass

    if hasattr(cls, "from_dict"):
        return cls.from_dict(payload)
    if isinstance(payload, dict):
        try:
            return cls(**payload)
        except TypeError:
            pass
    return cls(payload)


def _import_qualified(module_name: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _coerce_datetime(value: datetime | str | None) -> datetime:
    value = _unwrap_data(value)
    if value is None:
        return datetime.now()
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _coerce_date(value: date | str | None) -> date:
    value = _unwrap_data(value)
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _coerce_time(value: time | str | None) -> time:
    value = _unwrap_data(value)
    if value is None:
        return datetime.now().time()
    if isinstance(value, time):
        return value
    return time.fromisoformat(str(value))
