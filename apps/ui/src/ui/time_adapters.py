"""Adapter pattern for handling diverse time-dimension storage formats.

Each adapter declares the widget kind it expects to render, can detect whether
a (min, max) sample from the warehouse looks like its format, parses native
values into a canonical ``datetime``, formats them for display, and serializes
a chosen ``datetime`` back to the column's native storage type so the
orchestrator can push the filter down.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Protocol, runtime_checkable

try:
    from dateutil import parser as _dateutil_parser
except ImportError:  # pragma: no cover - dateutil is a transitive dep, but be defensive
    _dateutil_parser = None


WidgetKind = Literal["month", "date", "datetime", "datetime_tz"]


@runtime_checkable
class TimeAdapter(Protocol):
    """Protocol for time-dimension adapters."""

    widget_kind: WidgetKind
    name: str

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool: ...

    def parse(self, db_value: Any) -> dt.datetime: ...

    def format_for_display(self, value: dt.datetime) -> str: ...

    def serialize(self, value: dt.datetime) -> Any: ...

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        """Optional helper for building enumerated dropdowns (e.g., month picker)."""
        ...


_YYYYMM_INT_MIN = 190001
_YYYYMM_INT_MAX = 210012
_YYYYMM_STR_RE = re.compile(r"^\d{6}$")


def _is_yyyymm_int(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and _YYYYMM_INT_MIN <= value <= _YYYYMM_INT_MAX
        and 1 <= (value % 100) <= 12
    )


def _is_yyyymm_str(value: Any) -> bool:
    if not isinstance(value, str) or not _YYYYMM_STR_RE.match(value):
        return False
    return _is_yyyymm_int(int(value))


def _yyyymm_to_datetime(value: int) -> dt.datetime:
    year, month = divmod(value, 100)
    return dt.datetime(year, month, 1)


def _datetime_to_yyyymm(value: dt.datetime) -> int:
    return value.year * 100 + value.month


def _enumerate_months(start: dt.datetime, end: dt.datetime) -> List[dt.datetime]:
    months: List[dt.datetime] = []
    cursor = dt.datetime(start.year, start.month, 1)
    final = dt.datetime(end.year, end.month, 1)
    while cursor <= final:
        months.append(cursor)
        if cursor.month == 12:
            cursor = dt.datetime(cursor.year + 1, 1, 1)
        else:
            cursor = dt.datetime(cursor.year, cursor.month + 1, 1)
    return months


@dataclass(frozen=True)
class YearMonthIntAdapter:
    widget_kind: WidgetKind = "month"
    name: str = "yyyymm_int"

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool:
        if _is_yyyymm_int(min_val) and _is_yyyymm_int(max_val):
            return True
        if yaml_data_type and yaml_data_type.lower() == "number" and min_val is None and max_val is None:
            return True
        return False

    def parse(self, db_value: Any) -> dt.datetime:
        if isinstance(db_value, dt.datetime):
            return db_value
        return _yyyymm_to_datetime(int(db_value))

    def format_for_display(self, value: dt.datetime) -> str:
        return value.strftime("%Y %B")

    def serialize(self, value: dt.datetime) -> int:
        return _datetime_to_yyyymm(value)

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        return _enumerate_months(self.parse(min_val), self.parse(max_val))


@dataclass(frozen=True)
class YearMonthStringAdapter:
    widget_kind: WidgetKind = "month"
    name: str = "yyyymm_string"

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool:
        return _is_yyyymm_str(min_val) and _is_yyyymm_str(max_val)

    def parse(self, db_value: Any) -> dt.datetime:
        return _yyyymm_to_datetime(int(db_value))

    def format_for_display(self, value: dt.datetime) -> str:
        return value.strftime("%Y %B")

    def serialize(self, value: dt.datetime) -> str:
        return f"{value.year:04d}{value.month:02d}"

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        return _enumerate_months(self.parse(min_val), self.parse(max_val))


@dataclass(frozen=True)
class DateAdapter:
    widget_kind: WidgetKind = "date"
    name: str = "date"

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool:
        return (
            isinstance(min_val, dt.date)
            and not isinstance(min_val, dt.datetime)
            and isinstance(max_val, dt.date)
            and not isinstance(max_val, dt.datetime)
        )

    def parse(self, db_value: Any) -> dt.datetime:
        if isinstance(db_value, dt.datetime):
            return db_value
        return dt.datetime(db_value.year, db_value.month, db_value.day)

    def format_for_display(self, value: dt.datetime) -> str:
        return value.strftime("%Y-%m-%d")

    def serialize(self, value: dt.datetime) -> dt.date:
        return dt.date(value.year, value.month, value.day)

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        return []


@dataclass(frozen=True)
class TzDatetimeAdapter:
    widget_kind: WidgetKind = "datetime_tz"
    name: str = "datetime_tz"

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool:
        return (
            isinstance(min_val, dt.datetime)
            and min_val.tzinfo is not None
            and isinstance(max_val, dt.datetime)
            and max_val.tzinfo is not None
        )

    def parse(self, db_value: Any) -> dt.datetime:
        return db_value

    def format_for_display(self, value: dt.datetime) -> str:
        return value.strftime("%Y-%m-%d %H:%M %Z")

    def serialize(self, value: dt.datetime) -> dt.datetime:
        return value

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        return []


@dataclass(frozen=True)
class NaiveDatetimeAdapter:
    widget_kind: WidgetKind = "datetime"
    name: str = "datetime"

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool:
        return (
            isinstance(min_val, dt.datetime)
            and min_val.tzinfo is None
            and isinstance(max_val, dt.datetime)
            and max_val.tzinfo is None
        )

    def parse(self, db_value: Any) -> dt.datetime:
        return db_value

    def format_for_display(self, value: dt.datetime) -> str:
        return value.strftime("%Y-%m-%d %H:%M:%S")

    def serialize(self, value: dt.datetime) -> dt.datetime:
        return value

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        return []


@dataclass(frozen=True)
class StringDateAdapter:
    widget_kind: WidgetKind = "date"
    name: str = "string_date"

    def detect(self, min_val: Any, max_val: Any, yaml_data_type: Optional[str]) -> bool:
        if _dateutil_parser is None:
            return False
        if not (isinstance(min_val, str) and isinstance(max_val, str)):
            return False
        if _is_yyyymm_str(min_val) or _is_yyyymm_str(max_val):
            return False
        try:
            _dateutil_parser.parse(min_val, dayfirst=False)
            _dateutil_parser.parse(max_val, dayfirst=False)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def parse(self, db_value: Any) -> dt.datetime:
        if isinstance(db_value, dt.datetime):
            return db_value
        return _dateutil_parser.parse(db_value, dayfirst=False)  # type: ignore[union-attr]

    def format_for_display(self, value: dt.datetime) -> str:
        return value.isoformat()

    def serialize(self, value: dt.datetime) -> str:
        return value.isoformat()

    def enumerate_range(self, min_val: Any, max_val: Any) -> List[dt.datetime]:
        return []


TIME_ADAPTERS: List[TimeAdapter] = [
    YearMonthIntAdapter(),
    YearMonthStringAdapter(),
    DateAdapter(),
    TzDatetimeAdapter(),
    NaiveDatetimeAdapter(),
    StringDateAdapter(),
]


def select_adapter(
    min_val: Any,
    max_val: Any,
    yaml_data_type: Optional[str] = None,
) -> Optional[TimeAdapter]:
    """Return the first registered adapter whose ``detect`` matches the probe.

    Returns ``None`` when nothing matches — the caller falls back to a free
    text input and surfaces a warning.
    """

    for adapter in TIME_ADAPTERS:
        try:
            if adapter.detect(min_val, max_val, yaml_data_type):
                return adapter
        except Exception:
            continue
    return None


def default_adapter_for_yaml_type(yaml_data_type: Optional[str]) -> TimeAdapter:
    """Pick a sensible fallback when no DB probe is available.

    Used when Snowflake is unreachable and we still need a renderable widget.
    """

    declared = (yaml_data_type or "").lower()
    if declared in {"date"}:
        return DateAdapter()
    if declared in {"timestamp_tz", "timestamp_ltz"}:
        return TzDatetimeAdapter()
    if declared in {"timestamp", "timestamp_ntz", "datetime"}:
        return NaiveDatetimeAdapter()
    return YearMonthIntAdapter()


def synthetic_recent_range(months_back: int = 24) -> tuple[dt.datetime, dt.datetime]:
    """Build a fallback ``(min, max)`` ending at the current month."""

    today = dt.date.today()
    max_dt = dt.datetime(today.year, today.month, 1)
    year = today.year
    month = today.month - months_back
    while month <= 0:
        month += 12
        year -= 1
    min_dt = dt.datetime(year, month, 1)
    return min_dt, max_dt


def first_day_of_month(value: dt.datetime) -> dt.datetime:
    """Return the first day of the month for the given datetime."""
    return dt.datetime(value.year, value.month, 1)


def add_months(value: dt.datetime, months: int) -> dt.datetime:
    """Add or subtract months from a datetime using calendar arithmetic.
    
    Handles year boundaries correctly. For example:
    - add_months(2025-12-15, 1) -> 2026-01-15
    - add_months(2025-01-15, -1) -> 2024-12-15
    """
    year = value.year
    month = value.month + months
    
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1
    
    day = min(value.day, calendar.monthrange(year, month)[1])
    
    return dt.datetime(
        year, month, day,
        value.hour, value.minute, value.second,
        value.microsecond, value.tzinfo
    )


def month_aligned_exclusive_end(start: dt.datetime, end: dt.datetime) -> dt.datetime:
    """Calculate the exclusive upper bound for a half-open range [start, exclusive_end).
    
    Both inputs are aligned to their respective month boundaries:
    - The start is set to the first day of its month
    - The end is advanced to the first day of the month *after* the end month
    
    This ensures that SQL filters like `>= start AND < exclusive_end` capture
    all records in the inclusive month range without midnight cutoff issues.
    
    Example:
        start = 2025-01-15, end = 2025-12-20
        Returns: 2026-01-01 (first day of month after December 2025)
    """
    aligned_start = first_day_of_month(start)
    aligned_end_inclusive = first_day_of_month(end)
    exclusive_end = add_months(aligned_end_inclusive, 1)
    return exclusive_end


def last_day_of_month(value: dt.datetime) -> dt.datetime:
    last = calendar.monthrange(value.year, value.month)[1]
    return dt.datetime(value.year, value.month, last)
