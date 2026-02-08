"""Tests for pygent.common.base module."""
import json
import os
import pickle
import tempfile
import unittest
import unittest.mock
import sys
from pathlib import Path
from datetime import datetime, date, time
from decimal import Decimal
from enum import Enum

# Ensure project root is on path when running tests directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

from pygent.common import (
    PygentData,
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
    PygentOperator,
    create_pygent_data,
)


class TestPygentData(unittest.TestCase):
    """Tests for PygentData base class."""

    def test_init_default(self):
        self.assertIsNone(PygentData().data)

    def test_init_with_data(self):
        d = PygentData(42)
        self.assertEqual(d.data, 42)

    def test_repr(self):
        self.assertIn("PygentData", repr(PygentData("x")))
        self.assertIn("x", repr(PygentData("x")))

    def test_str(self):
        self.assertEqual(str(PygentData("hello")), "hello")

    def test_to_json(self):
        self.assertEqual(PygentData({"a": 1}).to_json(), '{"a": 1}')
        self.assertEqual(PygentData([1, 2]).to_json(), "[1, 2]")

    def test_to_dict(self):
        self.assertEqual(PygentData({"k": "v"}).to_dict(), {"k": "v"})

    def test_copy_immutable(self):
        d = PygentData(5)
        c = d.copy()
        self.assertIsNot(d, c)
        self.assertEqual(c.data, 5)

    def test_copy_mutable(self):
        lst = [1, 2, 3]
        d = PygentData(lst)
        c = d.copy()
        self.assertIsNot(d.data, c.data)
        self.assertEqual(c.data, [1, 2, 3])
        c.data.append(4)
        self.assertEqual(len(d.data), 3)

    def test_from_json(self):
        d = PygentData.from_json('{"a": 1}')
        self.assertEqual(d.data, {"a": 1})


class TestPygentString(unittest.TestCase):
    """Tests for PygentString."""

    def test_init(self):
        self.assertEqual(PygentString("hi").data, "hi")
        self.assertEqual(PygentString(123).data, "123")

    def test_length(self):
        self.assertEqual(PygentString("hello").length(), 5)

    def test_upper_lower(self):
        self.assertEqual(PygentString("Hi").upper().data, "HI")
        self.assertEqual(PygentString("Hi").lower().data, "hi")

    def test_split(self):
        self.assertEqual(PygentString("a,b,c").split(","), ["a", "b", "c"])

    def test_strip(self):
        self.assertEqual(PygentString("  x  ").strip().data, "x")

    def test_replace(self):
        self.assertEqual(PygentString("foo").replace("o", "e").data, "fee")

    def test_contains_startswith_endswith(self):
        s = PygentString("hello world")
        self.assertTrue(s.contains("world"))
        self.assertFalse(s.contains("xyz"))
        self.assertTrue(s.startswith("hello"))
        self.assertTrue(s.endswith("world"))

    def test_to_json_from_json(self):
        s = PygentString("test")
        self.assertEqual(s.to_json(), '"test"')
        self.assertEqual(PygentString.from_json('"test"').data, "test")


class TestPygentNumber(unittest.TestCase):
    """Tests for PygentNumber, PygentInt, PygentFloat."""

    def test_pygent_int_arithmetic(self):
        a, b = PygentInt(10), PygentInt(3)
        self.assertEqual((a + b).data, 13)
        self.assertEqual((a - b).data, 7)
        self.assertEqual((a * b).data, 30)
        self.assertEqual((a / b).data, 10 / 3)

    def test_pygent_int_with_plain_int(self):
        a = PygentInt(5)
        self.assertEqual((a + 3).data, 8)
        self.assertEqual((a - 2).data, 3)

    def test_pygent_int_comparison(self):
        a, b = PygentInt(5), PygentInt(10)
        self.assertTrue(a < b)
        self.assertFalse(a > b)
        self.assertTrue(a == PygentInt(5))

    def test_pygent_int_to_float_binary_hex(self):
        n = PygentInt(10)
        self.assertEqual(n.to_float().data, 10.0)
        self.assertEqual(n.to_binary(), "0b1010")
        self.assertEqual(n.to_hex(), "0xa")

    def test_pygent_int_even_odd(self):
        self.assertTrue(PygentInt(4).is_even())
        self.assertFalse(PygentInt(4).is_odd())
        self.assertTrue(PygentInt(3).is_odd())

    def test_pygent_float(self):
        f = PygentFloat(3.7)
        self.assertEqual(f.to_int().data, 3)
        self.assertEqual(f.round(1).data, 3.7)
        self.assertEqual(f.round(0).data, 4.0)
        self.assertEqual(f.ceil().data, 4)
        self.assertEqual(f.floor().data, 3)
        self.assertFalse(f.is_integer())
        self.assertTrue(PygentFloat(4.0).is_integer())


class TestPygentBool(unittest.TestCase):
    """Tests for PygentBool."""

    def test_init(self):
        self.assertFalse(PygentBool(False).data)
        self.assertTrue(PygentBool(True).data)
        self.assertTrue(PygentBool(1).data)

    def test_and_or_invert(self):
        t, f = PygentBool(True), PygentBool(False)
        self.assertFalse((t & f).data)
        self.assertTrue((t | f).data)
        self.assertTrue((~f).data)
        self.assertFalse((~t).data)

    def test_bool(self):
        self.assertTrue(bool(PygentBool(True)))
        self.assertFalse(bool(PygentBool(False)))


class TestPygentList(unittest.TestCase):
    """Tests for PygentList."""

    def test_init(self):
        self.assertEqual(PygentList().data, [])
        self.assertEqual(PygentList(None).data, [])
        self.assertEqual(PygentList([1, 2]).data, [1, 2])

    def test_append_extend_insert(self):
        L = PygentList([1])
        L.append(2)
        self.assertEqual(L.data, [1, 2])
        L.extend([3, 4])
        self.assertEqual(L.data, [1, 2, 3, 4])
        L.insert(0, 0)
        self.assertEqual(L.data, [0, 1, 2, 3, 4])

    def test_remove_pop_clear(self):
        L = PygentList([1, 2, 3, 2])
        L.remove(2)
        self.assertEqual(L.data, [1, 3, 2])
        self.assertEqual(L.pop(), 2)
        L.clear()
        self.assertEqual(L.data, [])

    def test_count_index(self):
        L = PygentList([1, 2, 2, 3])
        self.assertEqual(L.count(2), 2)
        self.assertEqual(L.index(3), 3)

    def test_sort_reverse(self):
        L = PygentList([3, 1, 2])
        L.sort()
        self.assertEqual(L.data, [1, 2, 3])
        L.reverse()
        self.assertEqual(L.data, [3, 2, 1])

    def test_filter_map(self):
        L = PygentList([1, 2, 3, 4, 5])
        evens = L.filter(lambda x: x % 2 == 0)
        self.assertEqual(evens.data, [2, 4])
        doubled = L.map(lambda x: x * 2)
        self.assertEqual(doubled.data, [2, 4, 6, 8, 10])

    def test_len_getitem_setitem(self):
        L = PygentList([10, 20, 30])
        self.assertEqual(len(L), 3)
        self.assertEqual(L[1], 20)
        L[1] = 99
        self.assertEqual(L[1], 99)

    def test_copy(self):
        L = PygentList([1, 2, 3])
        c = L.copy()
        self.assertIsNot(L.data, c.data)
        self.assertEqual(c.data, [1, 2, 3])


class TestPygentDict(unittest.TestCase):
    """Tests for PygentDict."""

    def test_init(self):
        self.assertEqual(PygentDict().data, {})
        self.assertEqual(PygentDict(None).data, {})
        self.assertEqual(PygentDict({"a": 1}).data, {"a": 1})

    def test_keys_values_items(self):
        d = PygentDict({"x": 1, "y": 2})
        self.assertEqual(set(d.keys()), {"x", "y"})
        self.assertEqual(set(d.values()), {1, 2})
        self.assertEqual(set(d.items()), {("x", 1), ("y", 2)})

    def test_get_set_pop_clear(self):
        d = PygentDict({"a": 1})
        self.assertEqual(d.get("a"), 1)
        self.assertIsNone(d.get("b"))
        self.assertEqual(d.get("b", 0), 0)
        d.set("b", 2)
        self.assertEqual(d.get("b"), 2)
        self.assertEqual(d.pop("a"), 1)
        d.clear()
        self.assertEqual(len(d), 0)

    def test_update_with_dict(self):
        d = PygentDict({"a": 1})
        d.update({"b": 2, "c": 3})
        self.assertEqual(d.data, {"a": 1, "b": 2, "c": 3})

    def test_update_with_pygent_dict(self):
        d = PygentDict({"a": 1})
        other = PygentDict({"b": 2})
        d.update(other)
        self.assertEqual(d.data, {"a": 1, "b": 2})

    def test_contains_len(self):
        d = PygentDict({"k": "v"})
        self.assertIn("k", d)
        self.assertNotIn("x", d)
        self.assertEqual(len(d), 1)


class TestPygentTuple(unittest.TestCase):
    """Tests for PygentTuple."""

    def test_init(self):
        self.assertEqual(PygentTuple().data, ())
        self.assertEqual(PygentTuple(None).data, ())
        self.assertEqual(PygentTuple((1, 2, 3)).data, (1, 2, 3))
        self.assertEqual(PygentTuple([1, 2]).data, (1, 2))

    def test_count_index(self):
        t = PygentTuple((1, 2, 2, 3))
        self.assertEqual(t.count(2), 2)
        self.assertEqual(t.index(3), 3)

    def test_len_getitem(self):
        t = PygentTuple((10, 20, 30))
        self.assertEqual(len(t), 3)
        self.assertEqual(t[1], 20)


class TestPygentSet(unittest.TestCase):
    """Tests for PygentSet."""

    def test_init(self):
        self.assertEqual(PygentSet().data, set())
        self.assertEqual(PygentSet(None).data, set())
        self.assertEqual(PygentSet({1, 2, 3}).data, {1, 2, 3})

    def test_add_remove_discard_clear(self):
        s = PygentSet({1, 2})
        s.add(3)
        self.assertEqual(s.data, {1, 2, 3})
        s.remove(2)
        self.assertEqual(s.data, {1, 3})
        s.discard(99)
        s.clear()
        self.assertEqual(s.data, set())

    def test_union_with_set(self):
        a, b = PygentSet({1, 2}), {2, 3}
        u = a.union(b)
        self.assertEqual(u.data, {1, 2, 3})

    def test_union_with_pygent_set(self):
        a = PygentSet({1, 2})
        b = PygentSet({2, 3})
        u = a.union(b)
        self.assertEqual(u.data, {1, 2, 3})

    def test_intersection_difference_symmetric_with_pygent_set(self):
        a = PygentSet({1, 2, 3})
        b = PygentSet({2, 3, 4})
        self.assertEqual(a.intersection(b).data, {2, 3})
        self.assertEqual(a.difference(b).data, {1})
        self.assertEqual(a.symmetric_difference(b).data, {1, 4})

    def test_contains_len(self):
        s = PygentSet({1, 2})
        self.assertIn(1, s)
        self.assertNotIn(3, s)
        self.assertEqual(len(s), 2)


class TestPygentBytes(unittest.TestCase):
    """Tests for PygentBytes."""

    def test_init(self):
        self.assertEqual(PygentBytes().data, b"")
        self.assertEqual(PygentBytes(b"hello").data, b"hello")

    def test_base64(self):
        b = PygentBytes(b"hello")
        enc = b.to_base64()
        self.assertIsInstance(enc, str)
        dec = PygentBytes.from_base64(enc)
        self.assertEqual(dec.data, b"hello")

    def test_hex(self):
        b = PygentBytes(b"\x00\xff")
        self.assertEqual(b.to_hex(), "00ff")
        self.assertEqual(PygentBytes.from_hex("00ff").data, b"\x00\xff")

    def test_decode(self):
        b = PygentBytes(b"hello")
        s = b.decode()
        self.assertIsInstance(s, PygentString)
        self.assertEqual(s.data, "hello")


class TestPygentDateTime(unittest.TestCase):
    """Tests for PygentDateTime, PygentDate, PygentTime."""

    def test_now_from_timestamp_from_iso(self):
        now = PygentDateTime.now()
        self.assertIsInstance(now.data, datetime)
        ts = now.to_timestamp()
        dt = PygentDateTime.from_timestamp(ts)
        self.assertAlmostEqual(dt.data.timestamp(), ts)
        iso = "2024-01-15T10:30:00"
        dt2 = PygentDateTime.from_isoformat(iso)
        self.assertEqual(dt2.data.isoformat(), iso)

    def test_format_replace_date_time(self):
        dt = PygentDateTime.from_isoformat("2024-01-15T10:30:00")
        self.assertIn("2024", dt.format())
        d = dt.date()
        self.assertIsInstance(d.data, date)
        t = dt.time()
        self.assertIsInstance(t.data, time)
        dt2 = dt.replace(year=2025)
        self.assertEqual(dt2.data.year, 2025)

    def test_pygent_date(self):
        today = PygentDate.today()
        self.assertIsInstance(today.data, date)
        d = PygentDate.from_isoformat("2024-01-15")
        self.assertEqual(d.data.isoformat(), "2024-01-15")
        d2 = PygentDate.from_isoformat("2024-01-20")
        self.assertEqual((d2 - d), 5)

    def test_pygent_time(self):
        t = PygentTime.now()
        self.assertIsInstance(t.data, time)
        self.assertIn(":", t.format())


class TestPygentDecimal(unittest.TestCase):
    """Tests for PygentDecimal."""

    def test_init(self):
        self.assertEqual(PygentDecimal("10.5").data, Decimal("10.5"))
        self.assertEqual(PygentDecimal(10).data, Decimal(10))


class TestPygentEnum(unittest.TestCase):
    """Tests for PygentEnum."""

    def test_enum(self):
        class Color(Enum):
            RED = 1
            GREEN = 2
        e = PygentEnum(Color.RED)
        self.assertEqual(e.name, "RED")
        self.assertEqual(e.value, 1)


class TestPygentNone(unittest.TestCase):
    """Tests for PygentNone."""

    def test_none(self):
        n = PygentNone()
        self.assertIsNone(n.data)
        self.assertTrue(n.is_none())
        self.assertFalse(bool(n))


class TestPygentAny(unittest.TestCase):
    """Tests for PygentAny."""

    def test_any(self):
        a = PygentAny(42)
        self.assertEqual(a.data, 42)
        self.assertEqual(a.get_type(), int)
        self.assertTrue(a.isinstance(int))
        self.assertFalse(a.isinstance(str))


class TestCreatePygentData(unittest.TestCase):
    """Tests for create_pygent_data factory."""

    def test_none(self):
        self.assertIsInstance(create_pygent_data(None), PygentNone)

    def test_str_int_float_bool(self):
        self.assertIsInstance(create_pygent_data("x"), PygentString)
        self.assertIsInstance(create_pygent_data(1), PygentInt)
        self.assertIsInstance(create_pygent_data(1.0), PygentFloat)
        self.assertIsInstance(create_pygent_data(True), PygentBool)

    def test_list_dict_tuple_set_bytes(self):
        self.assertIsInstance(create_pygent_data([]), PygentList)
        self.assertIsInstance(create_pygent_data({}), PygentDict)
        self.assertIsInstance(create_pygent_data(()), PygentTuple)
        self.assertIsInstance(create_pygent_data(set()), PygentSet)
        self.assertIsInstance(create_pygent_data(b""), PygentBytes)

    def test_datetime_date_time_decimal_enum(self):
        self.assertIsInstance(create_pygent_data(datetime.now()), PygentDateTime)
        self.assertIsInstance(create_pygent_data(date.today()), PygentDate)
        self.assertIsInstance(create_pygent_data(datetime.now().time()), PygentTime)
        self.assertIsInstance(create_pygent_data(Decimal("0")), PygentDecimal)
        class E(Enum):
            X = 1
        self.assertIsInstance(create_pygent_data(E.X), PygentEnum)

    def test_any_fallback(self):
        self.assertIsInstance(create_pygent_data(object()), PygentAny)


class TestPygentOperator(unittest.TestCase):
    """Tests for PygentOperator."""

    def test_base_operator_init(self):
        """Bare PygentOperator has no PygentData fields."""
        op = PygentOperator()
        self.assertEqual(op._pygent_fields, {})
        self.assertIsInstance(op.state_dict(), dict)
        self.assertEqual(op.state_dict(), {})

    def test_operator_subclass_with_fields(self):
        """Subclass with PygentData fields gets them initialized."""

        class MyOperator(PygentOperator):
            name: PygentString
            count: PygentInt
            items: PygentList

        op = MyOperator()
        self.assertIn("name", op._pygent_fields)
        self.assertIn("count", op._pygent_fields)
        self.assertIn("items", op._pygent_fields)
        self.assertIsInstance(op.name, PygentString)
        self.assertIsInstance(op.count, PygentInt)
        self.assertIsInstance(op.items, PygentList)
        self.assertEqual(op.name.data, "")
        self.assertEqual(op.count.data, 0)
        self.assertEqual(op.items.data, [])

    def test_state_dict(self):
        """state_dict returns serializable state of PygentData fields."""

        class MyOperator(PygentOperator):
            name: PygentString
            value: PygentInt

        op = MyOperator()
        op.name.data = "test"
        op.value.data = 42
        state = op.state_dict()
        self.assertEqual(state["name"], {"data": "test", "type": "PygentString"})
        self.assertEqual(state["value"], {"data": 42, "type": "PygentInt"})

    def test_load_state_dict_strict(self):
        """load_state_dict in strict mode requires matching keys."""

        class MyOperator(PygentOperator):
            a: PygentString
            b: PygentInt

        op = MyOperator()
        op.load_state_dict({"a": {"data": "x", "type": "PygentString"}, "b": {"data": 1, "type": "PygentInt"}})
        self.assertEqual(op.a.data, "x")
        self.assertEqual(op.b.data, 1)

        with self.assertRaises(ValueError) as ctx:
            op.load_state_dict({"a": {"data": "x", "type": "PygentString"}})
        self.assertIn("Missing", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            op.load_state_dict({"a": {"data": "x", "type": "PygentString"}, "b": 1, "c": 2})
        self.assertIn("Unexpected", str(ctx.exception))

    def test_load_state_dict_simple_value_format(self):
        """load_state_dict accepts simple data format (not full state dict)."""

        class MyOperator(PygentOperator):
            name: PygentString

        op = MyOperator()
        op.load_state_dict({"name": "simple"})
        self.assertEqual(op.name.data, "simple")

    def test_load_state_dict_non_strict(self):
        """load_state_dict with strict=False ignores extra/missing fields."""

        class MyOperator(PygentOperator):
            a: PygentString

        op = MyOperator()
        op.load_state_dict({"a": {"data": "ok", "type": "PygentString"}, "extra": 99}, strict=False)
        self.assertEqual(op.a.data, "ok")

    def test_save_and_load_json(self):
        """Save to JSON and load back."""

        class MyOperator(PygentOperator):
            name: PygentString
            value: PygentInt

        op = MyOperator()
        op.name.data = "saved"
        op.value.data = 100

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.json")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="json", include_metadata=True)
            self.assertTrue(os.path.exists(path))

            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="json", strict=True)
            self.assertEqual(op2.name.data, "saved")
            self.assertEqual(op2.value.data, 100)

    def test_save_and_load_json_no_metadata(self):
        """Save state_dict only (no metadata) and load."""

        class MyOperator(PygentOperator):
            x: PygentString

        op = MyOperator()
        op.x.data = "minimal"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.json")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="json", include_metadata=False)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("x", data)
            self.assertNotIn("state_dict", data)

            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="json", strict=False)
            self.assertEqual(op2.x.data, "minimal")

    def test_save_returns_absolute_path(self):
        """save() returns the absolute path of the saved file."""
        op = PygentOperator()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.json")
            with unittest.mock.patch("builtins.print"):
                result = op.save(path, format="json")
            self.assertIsInstance(result, str)
            self.assertEqual(result, str(Path(path).resolve()))
            self.assertTrue(Path(result).is_absolute())

    def test_save_and_load_pickle(self):
        """Save to pickle and load back with metadata."""
        class MyOperator(PygentOperator):
            name: PygentString
            value: PygentInt

        op = MyOperator()
        op.name.data = "pickle_test"
        op.value.data = 999

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.pkl")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="pickle", include_metadata=True)
            self.assertTrue(os.path.exists(path))

            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="pickle", strict=True)
            self.assertEqual(op2.name.data, "pickle_test")
            self.assertEqual(op2.value.data, 999)

    def test_save_and_load_pickle_no_metadata(self):
        """Save state_dict only as pickle and load."""
        class MyOperator(PygentOperator):
            flag: PygentBool

        op = MyOperator()
        op.flag.data = True

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.pkl")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="pickle", include_metadata=False)
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.assertIn("flag", data)
            self.assertNotIn("state_dict", data)

            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="pickle", strict=False)
            self.assertTrue(op2.flag.data)

    @unittest.skipUnless(_HAS_YAML, "PyYAML not installed")
    def test_save_and_load_yaml(self):
        """Save to YAML and load back."""
        class MyOperator(PygentOperator):
            name: PygentString
            value: PygentInt

        op = MyOperator()
        op.name.data = "yaml_test"
        op.value.data = 77

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.yaml")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="yaml", include_metadata=True)
            self.assertTrue(os.path.exists(path))

            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="yaml", strict=True)
            self.assertEqual(op2.name.data, "yaml_test")
            self.assertEqual(op2.value.data, 77)

    def test_load_format_auto_json(self):
        """Load with format='auto' detects JSON from file extension."""
        class MyOperator(PygentOperator):
            name: PygentString

        op = MyOperator()
        op.name.data = "auto_json"

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.json")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="json")
            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="auto", strict=True)
            self.assertEqual(op2.name.data, "auto_json")

    def test_load_format_auto_pickle(self):
        """Load with format='auto' detects pickle from .pickle extension."""
        class MyOperator(PygentOperator):
            x: PygentInt

        op = MyOperator()
        op.x.data = 123

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.pickle")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="pickle")
            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="auto", strict=True)
            self.assertEqual(op2.x.data, 123)

    def test_save_creates_parent_directories(self):
        """save() creates parent directories if they do not exist."""
        op = PygentOperator()
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "subdir", "deep", "op.json")
            with unittest.mock.patch("builtins.print"):
                op.save(nested, format="json")
            self.assertTrue(os.path.exists(nested))
            self.assertTrue(os.path.isfile(nested))

    def test_save_and_load_with_complex_state(self):
        """Round-trip save/load with PygentDict and PygentList."""
        class MyOperator(PygentOperator):
            name: PygentString
            items: PygentList
            config: PygentDict

        op = MyOperator()
        op.name.data = "complex"
        op.items.data = [1, "two", {"nested": True}]
        op.config.data = {"a": 1, "b": "x"}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.json")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="json", include_metadata=True)
            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="json", strict=True)
            self.assertEqual(op2.name.data, "complex")
            self.assertEqual(op2.items.data, [1, "two", {"nested": True}])
            self.assertEqual(op2.config.data, {"a": 1, "b": "x"})

    def test_load_strict_true_rejects_mismatched_state(self):
        """load(strict=True) with state_dict missing a field raises."""
        class FullOperator(PygentOperator):
            a: PygentString
            b: PygentInt

        class PartialOperator(PygentOperator):
            a: PygentString

        full = FullOperator()
        full.a.data = "x"
        full.b.data = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "full.json")
            with unittest.mock.patch("builtins.print"):
                full.save(path, format="json", include_metadata=False)
            # Save only state_dict with one key
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"a": {"data": "x", "type": "PygentString"}}, f, indent=2)
            partial = PartialOperator()
            with unittest.mock.patch("builtins.print"):
                partial.load(path, format="json", strict=True)
            self.assertEqual(partial.a.data, "x")
            # FullOperator loading partial state should fail in strict mode
            full2 = FullOperator()
            with unittest.mock.patch("builtins.print"):
                with self.assertRaises(ValueError) as ctx:
                    full2.load(path, format="json", strict=True)
            self.assertIn("Missing", str(ctx.exception))

    def test_parameters_and_named_parameters(self):
        """parameters() and named_parameters() return PygentData values."""

        class MyOperator(PygentOperator):
            name: PygentString
            count: PygentInt

        op = MyOperator()
        op.name.data = "p"
        op.count.data = 5
        params = op.parameters()
        self.assertEqual(params["name"], "p")
        self.assertEqual(params["count"], 5)
        named = op.named_parameters()
        self.assertEqual(len(named), 2)
        names = [n[0] for n in named]
        self.assertIn("name", names)
        self.assertIn("count", names)

    def test_to_train_eval(self):
        """to(), train(), eval() return self for chaining."""
        op = PygentOperator()
        with unittest.mock.patch("builtins.print"):
            self.assertIs(op.to("cpu"), op)
            self.assertIs(op.train(True), op)
            self.assertIs(op.train(mode=False), op)
            self.assertIs(op.eval(), op)

    def test_repr(self):
        """__repr__ includes field names and values."""

        class MyOperator(PygentOperator):
            name: PygentString
            n: PygentInt

        op = MyOperator()
        op.name.data = "r"
        op.n.data = 2
        r = repr(op)
        self.assertIn("MyOperator", r)
        self.assertIn("name", r)
        self.assertIn("n", r)
        self.assertIn("r", r)
        self.assertIn("2", r)

    def test_load_file_not_found(self):
        """load() raises FileNotFoundError when file does not exist."""
        op = PygentOperator()
        with self.assertRaises(FileNotFoundError):
            op.load("/nonexistent/path/xyz.json")

    def test_save_unsupported_format(self):
        """save() raises ValueError for unsupported format."""
        op = PygentOperator()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.x")
            with self.assertRaises(ValueError) as ctx:
                op.save(path, format="xml")
            self.assertIn("Unsupported format", str(ctx.exception))

    def test_load_unsupported_format(self):
        """load() raises ValueError for unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy")
            path = f.name
        try:
            op = PygentOperator()
            with self.assertRaises(ValueError) as ctx:
                op.load(path, format="xyz")
            self.assertIn("Unsupported format", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_checksum_warning_on_load(self):
        """Loading into operator with fewer fields uses strict=False; extra keys ignored."""

        class MyOperator(PygentOperator):
            a: PygentString

        class OtherOperator(PygentOperator):
            a: PygentString
            b: PygentInt

        op = OtherOperator()
        op.a.data = "x"
        op.b.data = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "op.json")
            with unittest.mock.patch("builtins.print"):
                op.save(path, format="json")
            op2 = MyOperator()
            with unittest.mock.patch("builtins.print"):
                op2.load(path, format="json", strict=False)
            self.assertEqual(op2.a.data, "x")


if __name__ == "__main__":
    unittest.main()
