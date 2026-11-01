"""The canonical batch container for observations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np
import pandas as pd

from infrapulse.data.model import (
    MetricObservation,
    SeriesKey,
    identifier_problem,
    label_key_problem,
)
from infrapulse.data.validation import (
    HOST,
    METRIC,
    RESOURCE,
    TIMESTAMP,
    UNIT,
    VALUE,
    Duplicates,
    OnInvalid,
    ValidationReport,
    canonical_columns,
    normalize_long_table,
)


class MetricFrame:
    """Validated observations in the canonical long layout: one row per sample.

    Columns are ``timestamp`` (``datetime64[ns, UTC]``), ``metric``, ``host``, ``resource``, one
    column per label key (sorted), ``unit`` (``str``; ``""`` means absent) and ``value`` (finite
    ``float64``). Rows are sorted by series, then time, and every series has one unit and at most
    one value per timestamp. See ``docs/data-model.md``.

    Build a frame with :meth:`from_dataframe` or :meth:`from_observations`. Both validate the input
    and attach a :class:`~infrapulse.data.validation.ValidationReport` describing every change
    made. The constructor accepts only data that is already canonical and checks the invariants.
    """

    __slots__ = ("_data", "_label_keys", "_report")

    def __init__(
        self,
        data: pd.DataFrame,
        *,
        label_keys: Iterable[str] = (),
        report: ValidationReport | None = None,
    ) -> None:
        keys = tuple(label_keys)
        check_canonical(data, keys)
        self._data = data.copy(deep=False)
        self._label_keys = keys
        if report is None:
            report = ValidationReport(input_rows=len(data), output_rows=len(data))
        self._report = report

    @classmethod
    def from_dataframe(
        cls,
        data: pd.DataFrame,
        *,
        timestamp: str = TIMESTAMP,
        metric: str = METRIC,
        value: str = VALUE,
        host: str = HOST,
        resource: str | None = RESOURCE,
        unit: str | None = UNIT,
        labels: Sequence[str] = (),
        assume_timezone: str | None = None,
        on_invalid: OnInvalid = "raise",
        duplicates: Duplicates = "raise",
    ) -> Self:
        """Validate a long table (one row per sample) and normalize it.

        The column parameters map your column names to the roles, e.g. ``metric="__name__"``.
        ``resource`` and ``unit`` are used when a column of that name exists; ``labels`` lists
        the columns that further identify a series. Any other column is an error, so a forgotten
        label cannot silently merge distinct series.

        Raises :class:`~infrapulse.errors.DataValidationError` with the full report when the data
        has problems that ``on_invalid`` and ``duplicates`` do not allow to be dropped or resolved.
        """
        table, label_keys, report = normalize_long_table(
            data,
            timestamp=timestamp,
            metric=metric,
            value=value,
            host=host,
            resource=resource,
            unit=unit,
            labels=labels,
            assume_timezone=assume_timezone,
            on_invalid=on_invalid,
            duplicates=duplicates,
        )
        return cls(table, label_keys=label_keys, report=report)

    @classmethod
    def from_observations(
        cls, observations: Iterable[MetricObservation], *, duplicates: Duplicates = "raise"
    ) -> Self:
        """Build a frame from records; they go through the same validation as a table."""
        records = list(observations)
        for record in records:
            if not isinstance(record, MetricObservation):
                raise TypeError(f"expected MetricObservation, got {type(record).__name__}")
        label_keys = sorted({key for record in records for key in record.labels})
        columns: dict[str, pd.Series] = {
            TIMESTAMP: pd.Series([record.timestamp for record in records], dtype=object),
            METRIC: pd.Series([record.metric for record in records], dtype=object),
            HOST: pd.Series([record.host for record in records], dtype=object),
            RESOURCE: pd.Series([record.resource for record in records], dtype=object),
            UNIT: pd.Series([record.unit for record in records], dtype=object),
            VALUE: pd.Series([record.value for record in records], dtype=np.float64),
        }
        for key in label_keys:
            columns[key] = pd.Series([record.labels.get(key) for record in records], dtype=object)
        return cls.from_dataframe(pd.DataFrame(columns), labels=label_keys, duplicates=duplicates)

    @property
    def data(self) -> pd.DataFrame:
        """The observations. A copy: changing it never changes the frame."""
        return self._data.copy(deep=False)

    @property
    def label_keys(self) -> tuple[str, ...]:
        return self._label_keys

    @property
    def key_columns(self) -> tuple[str, ...]:
        """The columns that identify a series: metric, host, resource and the label keys."""
        return (METRIC, HOST, RESOURCE, *self._label_keys)

    @property
    def report(self) -> ValidationReport:
        """What validation found and changed while building this frame."""
        return self._report

    def series(self) -> tuple[SeriesKey, ...]:
        """The series in the frame, in row order."""
        return series_keys(self._data, self._label_keys)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        if not len(self._data):
            return "MetricFrame(0 observations)"
        timestamps = self._data[TIMESTAMP]
        return (
            f"MetricFrame({len(self._data)} observations, {len(self.series())} series, "
            f"{timestamps.min().isoformat()} to {timestamps.max().isoformat()})"
        )


def series_keys(data: pd.DataFrame, label_keys: Sequence[str]) -> tuple[SeriesKey, ...]:
    """One :class:`SeriesKey` per distinct series in ``data``, in order of first appearance."""
    unique = data[[METRIC, HOST, RESOURCE, *label_keys]].drop_duplicates()
    keys = []
    for metric, host, resource, *labels in unique.itertuples(index=False, name=None):
        present = {key: value for key, value in zip(label_keys, labels, strict=True) if value}
        keys.append(SeriesKey(metric, host, resource or None, present))
    return tuple(keys)


def check_canonical(data: pd.DataFrame, label_keys: tuple[str, ...]) -> None:
    """Raise ``ValueError`` unless ``data`` has the layout and invariants of a MetricFrame."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if list(label_keys) != sorted(set(label_keys)):
        raise ValueError(f"label_keys must be sorted and unique, got {list(label_keys)}")
    for key in label_keys:
        if not isinstance(key, str) or label_key_problem(key) is not None:
            raise ValueError(f"invalid label key {key!r}")
    expected = canonical_columns(label_keys)
    if list(data.columns) != expected:
        raise ValueError(f"columns must be {expected}, got {list(data.columns)}")
    if not data.index.equals(pd.RangeIndex(len(data))):
        raise ValueError("the index must be a default RangeIndex")
    if str(data[TIMESTAMP].dtype) != "datetime64[ns, UTC]" or data[TIMESTAMP].isna().any():
        raise ValueError("timestamps must be datetime64[ns, UTC] with no missing values")
    values = data[VALUE]
    if values.dtype != np.float64 or not np.isfinite(values.to_numpy()).all():
        raise ValueError("values must be finite float64")
    for column in (METRIC, HOST, RESOURCE, *label_keys, UNIT):
        texts = data[column]
        if not isinstance(texts.dtype, pd.StringDtype) or texts.isna().any():
            raise ValueError(f"column {column!r} must have the str dtype and no missing values")
        for text in texts.unique():
            if text == "" and column not in (METRIC, HOST):
                continue
            if identifier_problem(text) is not None:
                raise ValueError(f"column {column!r} holds an invalid identifier {text!r}")
    _check_order(data, (METRIC, HOST, RESOURCE, *label_keys))


def _check_order(data: pd.DataFrame, key_columns: tuple[str, ...]) -> None:
    """Rows must be strictly increasing by (series, timestamp); one unit per series. O(n)."""
    if len(data) < 2:
        return
    before = np.zeros(len(data) - 1, dtype=bool)  # row i-1 sorts before row i on the keys so far
    tied = np.ones(len(data) - 1, dtype=bool)
    for column in key_columns:
        previous = data[column].iloc[:-1].reset_index(drop=True)
        current = data[column].iloc[1:].reset_index(drop=True)
        before |= tied & (previous < current).to_numpy(dtype=bool)
        tied &= (previous == current).to_numpy(dtype=bool)
    same_series = tied.copy()
    timestamps = data[TIMESTAMP].astype("int64").to_numpy()
    before |= tied & (timestamps[:-1] < timestamps[1:])
    if not before.all():
        raise ValueError(
            "rows must be sorted by series and then timestamp, with at most one row per "
            "series and timestamp"
        )
    units = data[UNIT].to_numpy(dtype=object)
    if (same_series & (units[:-1] != units[1:])).any():
        raise ValueError("every series must have exactly one unit")
