"""Put observations on a regular time grid without inventing values.

:func:`regularize` aggregates the samples of each series into fixed, epoch-aligned UTC buckets.
Buckets without samples stay empty (``value`` NaN, ``count`` 0). They are never interpolated or
forward-filled, because a gap can be exactly the outage an operator needs to see.
:func:`series_profile` and :func:`infer_interval` describe how each series was actually sampled.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd

from infrapulse.data.frame import MetricFrame, series_keys
from infrapulse.data.model import SeriesKey
from infrapulse.data.validation import (
    HOST,
    METRIC,
    RESOURCE,
    TIMESTAMP,
    UNIT,
    VALUE,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from infrapulse.errors import DataValidationError

Aggregation = Literal["mean", "median", "min", "max", "sum", "first", "last"]
COUNT = "count"

Int64Array = npt.NDArray[np.int64]
_AGGREGATIONS: tuple[str, ...] = get_args(Aggregation)
_INT64_MAX = int(np.iinfo(np.int64).max)


class RegularFrame:
    """Observations aggregated onto a regular, epoch-aligned UTC grid; built by :func:`regularize`.

    One row per series and bucket. ``timestamp`` is the bucket start (a bucket covers
    ``[timestamp, timestamp + interval)``), followed by the series columns, ``unit``, ``value``
    (the aggregate of the bucket's samples, NaN when there are none) and ``count`` (the number of
    samples; 0 marks a gap). Rows are sorted by series, then time.
    """

    __slots__ = ("_aggregation", "_data", "_interval", "_label_keys")

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        label_keys: tuple[str, ...],
        interval: pd.Timedelta,
        aggregation: Aggregation,
    ) -> None:
        expected = [TIMESTAMP, METRIC, HOST, RESOURCE, *label_keys, UNIT, VALUE, COUNT]
        if list(data.columns) != expected:
            raise ValueError(f"columns must be {expected}, got {list(data.columns)}")
        self._data = data.copy(deep=False)
        self._label_keys = label_keys
        self._interval = interval
        self._aggregation: Aggregation = aggregation

    @property
    def data(self) -> pd.DataFrame:
        """The grid. A copy: changing it never changes the frame."""
        return self._data.copy(deep=False)

    @property
    def label_keys(self) -> tuple[str, ...]:
        return self._label_keys

    @property
    def key_columns(self) -> tuple[str, ...]:
        return (METRIC, HOST, RESOURCE, *self._label_keys)

    @property
    def interval(self) -> pd.Timedelta:
        return self._interval

    @property
    def aggregation(self) -> Aggregation:
        return self._aggregation

    @property
    def observed(self) -> npt.NDArray[np.bool_]:
        """True for buckets that contain at least one sample."""
        return self._data[COUNT].to_numpy() > 0

    def series(self) -> tuple[SeriesKey, ...]:
        """The series on the grid, in row order."""
        return series_keys(self._data, self._label_keys)

    def gaps(self) -> pd.DataFrame:
        """Runs of consecutive empty buckets, one row per run.

        Columns: the series columns, ``start`` (start of the first empty bucket), ``end`` (end of
        the last empty bucket, exclusive) and ``missing_intervals`` (number of empty buckets).
        """
        data = self._data
        key_columns = list(self.key_columns)
        empty = data[COUNT].to_numpy() == 0
        first_of_series = np.diff(_series_ids(data, key_columns), prepend=-1) != 0
        last_of_series = np.append(first_of_series[1:], True)
        run_starts = np.flatnonzero(empty & (first_of_series | ~np.append(False, empty[:-1])))
        run_ends = np.flatnonzero(empty & (last_of_series | ~np.append(empty[1:], False)))
        runs = data.iloc[run_starts][key_columns].reset_index(drop=True)
        timestamps = data[TIMESTAMP]
        runs["start"] = timestamps.iloc[run_starts].reset_index(drop=True)
        runs["end"] = timestamps.iloc[run_ends].reset_index(drop=True) + self._interval
        runs["missing_intervals"] = run_ends - run_starts + 1
        return runs

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        empty = int((self._data[COUNT] == 0).sum())
        return (
            f"RegularFrame({len(self._data)} buckets of {self._interval}, "
            f"aggregation={self._aggregation!r}, {empty} empty)"
        )


def regularize(
    frame: MetricFrame,
    interval: str | timedelta,
    *,
    aggregation: Aggregation = "mean",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    max_buckets: int = 10_000_000,
) -> RegularFrame:
    """Aggregate every series into fixed ``interval`` buckets on a UTC grid anchored at the epoch.

    The grid does not depend on where the data starts, so a slice of a frame gets exactly the
    buckets of the whole frame. Each series runs from its first bucket to the last bucket of the
    frame: a host that stops reporting gets empty trailing buckets instead of silently ending.

    ``start`` (inclusive) and ``end`` (exclusive) restrict the samples to a window, and must be
    timezone-aware. With ``start``, every series of the frame gets every bucket from ``start``,
    including series without samples in the window. Partial first and last buckets show a smaller
    ``count``.

    Empty buckets keep ``value`` NaN and ``count`` 0; nothing is interpolated or filled.
    ``max_buckets`` stops a typo such as ``"1s"``, or a stray 1970 timestamp, from allocating an
    enormous grid.
    """
    if not isinstance(frame, MetricFrame):
        raise TypeError(f"frame must be a MetricFrame, got {type(frame).__name__}")
    step_size = _interval(interval)
    step = int(step_size.value)
    if aggregation not in _AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {list(_AGGREGATIONS)}, got {aggregation!r}")
    if isinstance(max_buckets, bool) or not isinstance(max_buckets, int) or max_buckets < 1:
        raise ValueError(f"max_buckets must be a positive integer, got {max_buckets!r}")
    start_ns = None if start is None else _utc_nanoseconds(start, "start")
    end_ns = None if end is None else _utc_nanoseconds(end, "end")
    if start_ns is not None and end_ns is not None and end_ns <= start_ns:
        raise ValueError("end must be later than start")

    data = frame.data
    key_columns = list(frame.key_columns)
    timestamps = data[TIMESTAMP].astype("int64").to_numpy()
    series = _series_ids(data, key_columns)
    n_series = int(series[-1]) + 1 if len(series) else 0
    inside = np.ones(len(data), dtype=bool)
    if start_ns is not None:
        inside &= timestamps >= start_ns
    if end_ns is not None:
        inside &= timestamps < end_ns
    buckets = timestamps - timestamps % step  # floor; numpy's % keeps the sign of the divisor

    if end_ns is not None:
        last = _floor(end_ns - 1, step)
    elif inside.any():
        last = int(np.max(buckets[inside]))
    else:
        return _empty_grid(frame.label_keys, step_size, aggregation)
    if start_ns is not None:
        first = np.full(n_series, _floor(start_ns, step), dtype=np.int64)
    else:
        first = np.full(n_series, _INT64_MAX, dtype=np.int64)
        np.minimum.at(first, series[inside], buckets[inside])
    included = first <= last  # without start, a series with no sample in the window is left out
    lengths = np.zeros(n_series, dtype=np.int64)
    lengths[included] = (last - first[included]) // step + 1
    total = int(lengths.sum())
    if total > max_buckets:
        earliest = pd.Timestamp(int(np.min(first[included])), tz="UTC").isoformat()
        raise ValueError(
            f"regularizing {int(included.sum())} series at {step_size} from {earliest} would "
            f"create {total:,} buckets, more than max_buckets={max_buckets:,}. Use a longer "
            "interval, narrow the window with start/end, look for stray timestamps (such as "
            "1970-01-01), or raise max_buckets."
        )

    offsets = np.cumsum(lengths) - lengths
    grid_series = np.repeat(np.arange(n_series), lengths)
    grid_timestamps = np.repeat(first, lengths) + (np.arange(total) - offsets[grid_series]) * step

    samples = pd.DataFrame(
        {"series": series[inside], "bucket": buckets[inside], VALUE: data[VALUE].to_numpy()[inside]}
    )
    grouped = samples.groupby(["series", "bucket"], sort=True)[VALUE]
    with np.errstate(over="ignore", invalid="ignore"):
        aggregated = grouped.agg(aggregation)
    sample_counts = grouped.size()
    aggregate_values = aggregated.to_numpy(dtype=np.float64)
    if not np.isfinite(aggregate_values).all():
        issue = ValidationIssue(
            "aggregation_overflow",
            Severity.ERROR,
            f"the {aggregation} of some buckets overflowed float64",
            count=int((~np.isfinite(aggregate_values)).sum()),
        )
        raise DataValidationError(
            f"cannot regularize: {issue.message}",
            report=ValidationReport((issue,), input_rows=len(data), output_rows=0),
        )
    sample_series = aggregated.index.get_level_values("series").to_numpy(dtype=np.int64)
    sample_buckets = aggregated.index.get_level_values("bucket").to_numpy(dtype=np.int64)
    slots = offsets[sample_series] + (sample_buckets - first[sample_series]) // step
    values = np.full(total, np.nan)
    values[slots] = aggregate_values
    counts = np.zeros(total, dtype=np.int64)
    counts[slots] = sample_counts.to_numpy(dtype=np.int64)

    first_rows = np.flatnonzero(np.diff(series, prepend=-1) != 0)
    grid = data.iloc[first_rows][[*key_columns, UNIT]].reset_index(drop=True)
    grid = grid.take(grid_series).reset_index(drop=True)
    grid.insert(0, TIMESTAMP, _utc_index(grid_timestamps))
    grid[VALUE] = values
    grid[COUNT] = counts
    return RegularFrame(
        grid, label_keys=frame.label_keys, interval=step_size, aggregation=aggregation
    )


def series_profile(frame: MetricFrame, *, tolerance: float = 0.25) -> pd.DataFrame:
    """Describe how each series was sampled, one row per series.

    Columns: the series columns and ``unit``; ``n_observations``; ``first`` and ``last``;
    ``median_interval``, ``min_interval`` and ``max_interval`` between consecutive samples (NaT
    for a single observation); ``irregular_fraction``, the share of intervals that differ from the
    median by more than ``tolerance`` times the median (NaN for a single observation); and
    ``constant``, true when a series has at least two samples and all of them are equal.
    """
    if not isinstance(frame, MetricFrame):
        raise TypeError(f"frame must be a MetricFrame, got {type(frame).__name__}")
    if isinstance(tolerance, bool) or not isinstance(tolerance, int | float) or tolerance < 0:
        raise ValueError(f"tolerance must be a non-negative number, got {tolerance!r}")
    data = frame.data
    key_columns = list(frame.key_columns)
    series = _series_ids(data, key_columns)
    first_rows = np.flatnonzero(np.diff(series, prepend=-1) != 0)
    last_rows = np.append(first_rows[1:] - 1, len(data) - 1)[: len(first_rows)]
    n_series = len(first_rows)
    counts = last_rows - first_rows + 1

    timestamps = data[TIMESTAMP].astype("int64").to_numpy()
    same_series = series[1:] == series[:-1]
    steps = np.diff(timestamps)[same_series]
    step_series = series[1:][same_series]
    stats = (
        pd.DataFrame({"series": step_series, "step": steps})
        .groupby("series")["step"]
        .agg(["median", "min", "max"])
        .reindex(range(n_series))
    )
    medians = stats["median"].to_numpy(dtype=np.float64)
    off_median = np.abs(steps - medians[step_series]) > tolerance * medians[step_series]
    has_steps = counts >= 2
    irregular = np.full(n_series, np.nan)
    irregular[has_steps] = np.bincount(
        step_series, weights=off_median.astype(np.float64), minlength=n_series
    )[has_steps] / (counts[has_steps] - 1)
    values = data[VALUE].to_numpy()
    if n_series:
        constant = has_steps & (
            np.minimum.reduceat(values, first_rows) == np.maximum.reduceat(values, first_rows)
        )
    else:
        constant = np.zeros(0, dtype=bool)

    profile = data.iloc[first_rows][[*key_columns, UNIT]].reset_index(drop=True)
    profile["n_observations"] = counts.astype(np.int64)
    profile["first"] = data[TIMESTAMP].iloc[first_rows].reset_index(drop=True)
    profile["last"] = data[TIMESTAMP].iloc[last_rows].reset_index(drop=True)
    for name in ("median", "min", "max"):
        profile[f"{name}_interval"] = pd.to_timedelta(
            stats[name].to_numpy(dtype=np.float64), unit="ns"
        ).as_unit("ns")
    profile["irregular_fraction"] = irregular
    profile["constant"] = constant
    return profile


def infer_interval(frame: MetricFrame) -> pd.Timedelta:
    """The typical sampling interval: the median of the per-series median intervals.

    The result is rounded to whole seconds when it is at least one second. Nothing calls this
    implicitly: pass the result to :func:`regularize` yourself, and check :func:`series_profile`
    first when series are sampled at different rates.
    """
    medians = series_profile(frame)["median_interval"].dropna()
    if medians.empty:
        issue = ValidationIssue(
            "insufficient_data",
            Severity.ERROR,
            "no series has two observations, so no sampling interval can be measured",
        )
        raise DataValidationError(
            f"cannot infer the sampling interval: {issue.message}",
            report=ValidationReport((issue,), input_rows=len(frame), output_rows=0),
        )
    interval = pd.Timedelta(medians.median())
    if interval >= pd.Timedelta(seconds=1):
        interval = interval.round("s")
    return interval


def _interval(interval: str | timedelta) -> pd.Timedelta:
    if not isinstance(interval, str | timedelta):
        raise TypeError(
            f"interval must be a duration string such as '5min' or a timedelta, "
            f"got {type(interval).__name__}"
        )
    try:
        step = pd.Timedelta(interval)
    except ValueError as exc:
        raise ValueError(
            f"invalid interval {interval!r}; use a fixed duration such as '30s', '5min' or '1h'"
        ) from exc
    if pd.isna(step) or step <= pd.Timedelta(0):
        raise ValueError(f"interval must be positive, got {interval!r}")
    return step.as_unit("ns")


def _utc_nanoseconds(moment: str | datetime, name: str) -> int:
    if not isinstance(moment, str | datetime):
        raise TypeError(f"{name} must be a datetime or an ISO 8601 string, got {moment!r}")
    stamp = pd.Timestamp(moment)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(
            f"{name} must be timezone-aware, e.g. '2026-01-05T00:00:00Z', got {moment!r}"
        )
    return int(stamp.tz_convert("UTC").as_unit("ns").value)


def _floor(nanoseconds: int, step: int) -> int:
    return nanoseconds - nanoseconds % step


def _series_ids(data: pd.DataFrame, key_columns: list[str]) -> Int64Array:
    """0, 1, 2, ... per series; ``data`` is sorted by series, so the ids are non-decreasing."""
    if data.empty:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(data.groupby(key_columns, sort=False).ngroup(), dtype=np.int64)


def _utc_index(nanoseconds: Int64Array) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(nanoseconds.view("datetime64[ns]")).tz_localize("UTC")


def _empty_grid(
    label_keys: tuple[str, ...], interval: pd.Timedelta, aggregation: Aggregation
) -> RegularFrame:
    columns: dict[str, pd.Series] = {TIMESTAMP: pd.Series([], dtype="datetime64[ns, UTC]")}
    for column in (METRIC, HOST, RESOURCE, *label_keys, UNIT):
        columns[column] = pd.Series([], dtype="str")
    columns[VALUE] = pd.Series([], dtype=np.float64)
    columns[COUNT] = pd.Series([], dtype=np.int64)
    return RegularFrame(
        pd.DataFrame(columns), label_keys=label_keys, interval=interval, aggregation=aggregation
    )
