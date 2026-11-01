"""Record types: a single observation and the identity of a series.

Batch work goes through :class:`~infrapulse.data.frame.MetricFrame`. These records are for building
observations one at a time (collectors, tests, a future streaming API) and for naming series in
results. Both enforce the identifier rules documented in ``docs/data-model.md``.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

import numpy as np
import pandas as pd

from infrapulse.errors import DataValidationError

# Column names of the canonical tables. Labels become columns, so label keys may not reuse them.
RESERVED_NAMES: frozenset[str] = frozenset(
    {"timestamp", "metric", "value", "host", "resource", "unit", "count"}
)


def identifier_problem(text: str) -> str | None:
    """Return why ``text`` is not a valid identifier, or ``None`` if it is valid.

    Identifiers (metric, host, resource, unit, label keys and values) are compared exactly, so
    stray whitespace or control characters would silently split one series into several. They are
    rejected rather than stripped.
    """
    if not text:
        return "is empty"
    if text != text.strip():
        return "has leading or trailing whitespace"
    if not text.isprintable():
        return "contains control or other non-printable characters"
    return None


def label_key_problem(key: str) -> str | None:
    """Like :func:`identifier_problem`, but also rejects the reserved column names."""
    problem = identifier_problem(key)
    if problem is None and key in RESERVED_NAMES:
        return "is reserved for a core column"
    return problem


def freeze_labels(labels: Mapping[str, str]) -> Mapping[str, str]:
    """Validate ``labels`` and return a read-only copy sorted by key."""
    if not isinstance(labels, Mapping):
        raise TypeError(f"labels must be a mapping of str to str, got {type(labels).__name__}")
    for key, value in labels.items():
        _require_identifier("label key", key, problem_of=label_key_problem)
        _require_identifier(f"label {key!r}", value)
    return MappingProxyType(dict(sorted(labels.items())))


def series_text(metric: str, host: str, resource: str | None, labels: Mapping[str, str]) -> str:
    """Render a series identity as ``metric{host="...",resource="...",label="..."}``."""
    parts = [f'host="{_escape(host)}"']
    if resource is not None:
        parts.append(f'resource="{_escape(resource)}"')
    parts.extend(f'{key}="{_escape(value)}"' for key, value in labels.items())
    return f"{metric}{{{','.join(parts)}}}"


@dataclass(frozen=True, slots=True)
class SeriesKey:
    """Identity of one time series: metric, host, optional resource and labels.

    Two keys are equal when all four parts are equal; label order does not matter.
    """

    metric: str
    host: str
    resource: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier("metric", self.metric)
        _require_identifier("host", self.host)
        if self.resource is not None:
            _require_identifier("resource", self.resource)
        object.__setattr__(self, "labels", freeze_labels(self.labels))

    def __hash__(self) -> int:
        return hash((self.metric, self.host, self.resource, tuple(self.labels.items())))

    def __str__(self) -> str:
        return series_text(self.metric, self.host, self.resource, self.labels)


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """One numeric sample of one series at one instant.

    ``timestamp`` must be timezone-aware. It is kept as given and converted to UTC when the
    observation enters a :class:`~infrapulse.data.frame.MetricFrame`. ``value`` must be a finite
    real number (a bool counts as 0 or 1) and is stored as a float. A missing measurement is
    represented by not creating an observation, so the gap stays visible after resampling.
    """

    timestamp: datetime
    metric: str
    value: float
    host: str
    resource: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    unit: str | None = None

    def __post_init__(self) -> None:
        _require_aware(self.timestamp)
        _require_identifier("metric", self.metric)
        _require_identifier("host", self.host)
        if self.resource is not None:
            _require_identifier("resource", self.resource)
        if self.unit is not None:
            _require_identifier("unit", self.unit)
        object.__setattr__(self, "value", _finite_float(self.value))
        object.__setattr__(self, "labels", freeze_labels(self.labels))

    def __hash__(self) -> int:
        return hash(
            (
                self.timestamp,
                self.metric,
                self.value,
                self.host,
                self.resource,
                tuple(self.labels.items()),
                self.unit,
            )
        )

    @property
    def key(self) -> SeriesKey:
        """The series this observation belongs to."""
        return SeriesKey(self.metric, self.host, self.resource, self.labels)


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _require_identifier(
    name: str,
    value: object,
    *,
    problem_of: Callable[[str], str | None] = identifier_problem,
) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    problem = problem_of(value)
    if problem is not None:
        raise DataValidationError(f"{name} {problem}: {value!r}")


def _require_aware(timestamp: object) -> None:
    if not isinstance(timestamp, datetime):
        raise TypeError(f"timestamp must be a datetime, got {type(timestamp).__name__}")
    if pd.isna(timestamp):  # pd.NaT is a datetime subclass whose utcoffset() raises
        raise DataValidationError("timestamp is missing (NaT)")
    if timestamp.utcoffset() is None:
        raise DataValidationError(
            f"timestamp {timestamp.isoformat()} has no time zone; attach one, e.g. "
            "tzinfo=datetime.UTC or tzinfo=zoneinfo.ZoneInfo('Asia/Jakarta')"
        )


def _finite_float(value: object) -> float:
    if not isinstance(value, numbers.Real | np.bool_):
        raise TypeError(f"value must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise DataValidationError(f"value must be finite, got {number!r}")
    return number
