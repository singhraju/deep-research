"""Unit tests for ui.time_adapters — adapter selection and round-tripping."""

import datetime as dt

import pytest

from ui import time_adapters as ta


def test_select_adapter_yyyymm_int():
    adapter = ta.select_adapter(202301, 202604, "number")
    assert adapter is not None
    assert adapter.widget_kind == "month"
    assert adapter.name == "yyyymm_int"


def test_select_adapter_yyyymm_string():
    adapter = ta.select_adapter("202301", "202604", None)
    assert adapter is not None
    assert adapter.widget_kind == "month"
    assert adapter.name == "yyyymm_string"


def test_select_adapter_date():
    adapter = ta.select_adapter(dt.date(2024, 1, 1), dt.date(2026, 6, 1), "date")
    assert adapter is not None
    assert adapter.widget_kind == "date"
    assert adapter.name == "date"


def test_select_adapter_tz_datetime():
    adapter = ta.select_adapter(
        dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc),
        dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc),
        "timestamp_tz",
    )
    assert adapter is not None
    assert adapter.widget_kind == "datetime_tz"


def test_select_adapter_naive_datetime():
    adapter = ta.select_adapter(
        dt.datetime(2024, 1, 1),
        dt.datetime(2026, 6, 1),
        "timestamp_ntz",
    )
    assert adapter is not None
    assert adapter.widget_kind == "datetime"


def test_select_adapter_returns_none_when_unmatched():
    assert ta.select_adapter(object(), object(), None) is None


def test_yyyymm_int_round_trip():
    adapter = ta.YearMonthIntAdapter()
    parsed = adapter.parse(202606)
    assert parsed == dt.datetime(2026, 6, 1)
    assert adapter.serialize(parsed) == 202606


def test_yyyymm_string_round_trip():
    adapter = ta.YearMonthStringAdapter()
    parsed = adapter.parse("202606")
    assert parsed == dt.datetime(2026, 6, 1)
    assert adapter.serialize(parsed) == "202606"


def test_yyyymm_enumerate_range_inclusive():
    adapter = ta.YearMonthIntAdapter()
    months = adapter.enumerate_range(202401, 202404)
    assert [adapter.serialize(m) for m in months] == [202401, 202402, 202403, 202404]


def test_default_adapter_for_yaml_type():
    assert ta.default_adapter_for_yaml_type("number").name == "yyyymm_int"
    assert ta.default_adapter_for_yaml_type("date").name == "date"
    assert ta.default_adapter_for_yaml_type("timestamp_tz").name == "datetime_tz"
    assert ta.default_adapter_for_yaml_type("timestamp_ntz").name == "datetime"
    assert ta.default_adapter_for_yaml_type(None).name == "yyyymm_int"


def test_synthetic_recent_range_returns_24_months():
    min_dt, max_dt = ta.synthetic_recent_range(24)
    diff_months = (max_dt.year - min_dt.year) * 12 + (max_dt.month - min_dt.month)
    assert diff_months == 24


def test_yyyymm_int_rejects_out_of_range():
    adapter = ta.YearMonthIntAdapter()
    assert adapter.detect(189912, 200001, None) is False  # 189912 < min
    assert adapter.detect(202413, 202504, None) is False  # month 13 invalid
