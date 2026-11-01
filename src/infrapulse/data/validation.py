"""Validate a long table of observations and normalize it into the canonical layout.

Every problem becomes a :class:`ValidationIssue` with a stable ``code``; the codes are documented in
``docs/data-model.md``. Problems are collected before anything is raised, so one run shows
everything that needs fixing. Nothing changes silently: dropped, collapsed, reordered and converted
rows are all reported.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.extensions import ExtensionDtype

from infrapulse.data.model import identifier_problem, label_key_problem, series_text
from infrapulse.data.timestamps import parse_timestamps
from infrapulse.errors import DataValidationError

OnInvalid = Literal["raise", "drop"]
Duplicates = Literal["raise", "first", "last", "mean"]

TIMESTAMP = "timestamp"
METRIC = "metric"
HOST = "host"
RESOURCE = "resource"
UNIT = "unit"
VALUE = "value"

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
ObjectArray = npt.NDArray[np.object_]

_MAX_EXAMPLES = 5
_INT64_MIN = int(np.iinfo(np.int64).min)
_FIX_OR_DROP = "fix the input, or pass on_invalid='drop' to drop these rows"
_INVALID_IDENTIFIER = (
    "with an identifier that is not a string, has leading or trailing whitespace, "
    "or contains control characters"
)


def canonical_columns(label_keys: Sequence[str]) -> list[str]:
    """Column order of a :class:`~infrapulse.data.frame.MetricFrame` with these label keys."""
    return [TIMESTAMP, METRIC, HOST, RESOURCE, *label_keys, UNIT, VALUE]


class Severity(StrEnum):
    """How serious a :class:`ValidationIssue` is."""

    ERROR = "error"  # the data cannot be used as it is, so validation raises
    WARNING = "warning"  # rows were dropped or values were changed
    INFO = "info"  # a lossless normalization: sorting, UTC conversion, exact duplicates


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One kind of problem found during validation, with up to five example rows."""

    code: str
    severity: Severity
    message: str
    count: int = 0
    column: str | None = None
    examples: tuple[str, ...] = ()
    hint: str | None = None

    def __str__(self) -> str:
        where = f" [{self.column}]" if self.column is not None else ""
        text = f"{self.severity.upper()} {self.code}{where}: {self.message}"
        if self.examples:
            text += "\n    e.g. " + "; ".join(self.examples)
        if self.hint is not None:
            text += f"\n    hint: {self.hint}"
        return text


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything validation found and changed, in the order it was checked."""

    issues: tuple[ValidationIssue, ...] = ()
    input_rows: int = 0
    output_rows: int = 0

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return self._with(Severity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return self._with(Severity.WARNING)

    @property
    def infos(self) -> tuple[ValidationIssue, ...]:
        return self._with(Severity.INFO)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def codes(self) -> frozenset[str]:
        return frozenset(issue.code for issue in self.issues)

    def issues_for(self, code: str) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.code == code)

    def summary(self) -> str:
        header = (
            f"{_n(self.input_rows, 'input row')} -> {_n(self.output_rows, 'observation')}: "
            f"{_n(len(self.errors), 'error')}, {_n(len(self.warnings), 'warning')}, "
            f"{_n(len(self.infos), 'note')}"
        )
        return "\n".join([header, *(f"  {issue}" for issue in self.issues)])

    def _with(self, severity: Severity) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity is severity)


class _IssueLog:
    def __init__(self, input_rows: int) -> None:
        self.input_rows = input_rows
        self.issues: list[ValidationIssue] = []

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        count: int = 0,
        column: str | None = None,
        examples: Iterable[str] = (),
        hint: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                code=code,
                severity=severity,
                message=message,
                count=count,
                column=column,
                examples=tuple(islice(examples, _MAX_EXAMPLES)),
                hint=hint,
            )
        )

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)

    def report(self, output_rows: int) -> ValidationReport:
        return ValidationReport(tuple(self.issues), self.input_rows, output_rows)

    def error(self) -> DataValidationError:
        report = self.report(output_rows=0)
        return DataValidationError(
            f"input data failed validation\n{report.summary()}", report=report
        )


@dataclass(frozen=True)
class _Columns:
    timestamp: str
    metric: str
    value: str
    host: str
    resource: str | None
    unit: str | None
    labels: tuple[str, ...]  # sorted


@dataclass(frozen=True)
class _Identifiers:
    text: ObjectArray  # the identifier, or "" where missing or invalid
    missing: BoolArray
    invalid: BoolArray


def normalize_long_table(
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
) -> tuple[pd.DataFrame, tuple[str, ...], ValidationReport]:
    """Validate ``data`` and return ``(canonical table, label keys, report)``.

    Raises :class:`~infrapulse.errors.DataValidationError`, carrying the full report, if any
    problem of severity ERROR remains. The input is never modified.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas DataFrame, got {type(data).__name__}")
    if on_invalid not in ("raise", "drop"):
        raise ValueError(f"on_invalid must be 'raise' or 'drop', got {on_invalid!r}")
    if duplicates not in ("raise", "first", "last", "mean"):
        raise ValueError(
            f"duplicates must be 'raise', 'first', 'last' or 'mean', got {duplicates!r}"
        )
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of column names such as ['job'], not a string")

    log = _IssueLog(len(data))
    columns = _resolve_columns(
        data,
        log,
        timestamp=timestamp,
        metric=metric,
        value=value,
        host=host,
        resource=resource,
        unit=unit,
        labels=labels,
    )
    if columns is None or log.has_errors:
        raise log.error()
    if len(data) == 0:
        log.add("empty_input", Severity.WARNING, "the input has no rows")

    table = _parse_rows(data, columns, log, assume_timezone=assume_timezone, on_invalid=on_invalid)
    if table is None:
        raise log.error()
    table = _normalize_series(table, columns.labels, log, duplicates=duplicates)
    if log.has_errors:
        raise log.error()
    return table, columns.labels, log.report(output_rows=len(table))


def _resolve_columns(
    data: pd.DataFrame,
    log: _IssueLog,
    *,
    timestamp: str,
    metric: str,
    value: str,
    host: str,
    resource: str | None,
    unit: str | None,
    labels: Sequence[str],
) -> _Columns | None:
    present = data.columns
    if present.has_duplicates:
        repeated = present[present.duplicated()].unique()
        log.add(
            "duplicate_columns",
            Severity.ERROR,
            f"{_n(len(repeated), 'column name')} used more than once",
            count=len(repeated),
            examples=(repr(name) for name in repeated),
            hint="give every column a unique name",
        )
        return None

    for role, name in (
        ("timestamp", timestamp),
        ("metric", metric),
        ("value", value),
        ("host", host),
    ):
        if name not in present:
            hint = f"map a differently named column with {role}='<column name>'"
            if role == "timestamp" and isinstance(data.index, pd.DatetimeIndex):
                hint = "the timestamps look like the index; call df.reset_index() first"
            log.add(
                "missing_column",
                Severity.ERROR,
                f"the {role} column {name!r} does not exist",
                column=name,
                hint=hint,
            )

    optional: dict[str, str | None] = {}
    for role, column, default in (("resource", resource, RESOURCE), ("unit", unit, UNIT)):
        optional[role] = column if column is not None and column in present else None
        if column is not None and column not in present and column != default:
            log.add(
                "missing_column",
                Severity.ERROR,
                f"the {role} column {column!r} does not exist",
                column=column,
            )

    label_columns: list[str] = []
    for name in labels:
        if not isinstance(name, str):
            log.add(
                "invalid_label_key",
                Severity.ERROR,
                f"label column names must be strings, got {name!r}",
            )
            continue
        problem = label_key_problem(name)
        if problem is not None:
            log.add(
                "invalid_label_key",
                Severity.ERROR,
                f"the label column name {name!r} {problem}",
                column=name,
                hint="rename the column; label names become column names",
            )
        elif name not in present:
            log.add(
                "missing_column",
                Severity.ERROR,
                f"the label column {name!r} does not exist",
                column=name,
            )
        else:
            label_columns.append(name)

    resource_column, unit_column = optional["resource"], optional["unit"]
    mapped = [timestamp, metric, value, host, *label_columns]
    mapped += [name for name in (resource_column, unit_column) if name is not None]
    repeated_roles = sorted({name for name in mapped if mapped.count(name) > 1})
    if repeated_roles:
        raise ValueError(f"each column can have only one role, but {repeated_roles} have several")
    unexpected = [name for name in present if name not in set(mapped)]
    if unexpected:
        log.add(
            "unexpected_columns",
            Severity.ERROR,
            f"{_n(len(unexpected), 'column')} not mapped to any role, which would be ignored",
            count=len(unexpected),
            examples=(repr(name) for name in unexpected),
            hint="pass labels=[...] for columns that identify series, and drop the others",
        )
    return _Columns(
        timestamp=timestamp,
        metric=metric,
        value=value,
        host=host,
        resource=resource_column,
        unit=unit_column,
        labels=tuple(sorted(label_columns)),
    )


def _parse_rows(
    data: pd.DataFrame,
    columns: _Columns,
    log: _IssueLog,
    *,
    assume_timezone: str | None,
    on_invalid: OnInvalid,
) -> pd.DataFrame | None:
    """Parse every cell, report row-level problems and return the usable rows (input order)."""
    raw_timestamps = data[columns.timestamp]
    if _is_number_dtype(raw_timestamps.dtype) and raw_timestamps.notna().any():
        present = raw_timestamps.notna().to_numpy(dtype=bool)
        log.add(
            "numeric_timestamps",
            Severity.ERROR,
            "timestamps are plain numbers, so their unit (seconds, milliseconds, ...) is ambiguous",
            count=int(present.sum()),
            column=columns.timestamp,
            examples=_examples(present, raw_timestamps),
            hint="convert them first, e.g. pd.to_datetime(df[column], unit='s', utc=True)",
        )
        return None
    timestamps = parse_timestamps(raw_timestamps, assume_timezone=assume_timezone)
    values, value_missing, value_invalid = _parse_values(data[columns.value])
    metric = _parse_identifiers(data[columns.metric])
    host = _parse_identifiers(data[columns.host])
    # Missing cells in optional columns mean "absent" and are not problems.
    resource = None if columns.resource is None else _parse_identifiers(data[columns.resource])
    unit = None if columns.unit is None else _parse_identifiers(data[columns.unit])
    labels = {name: _parse_identifiers(data[name]) for name in columns.labels}

    if timestamps.naive.any():
        count = int(timestamps.naive.sum())
        log.add(
            "naive_timestamps",
            Severity.ERROR,
            f"{_n(count, 'timestamp')} without a time zone, so the instant is ambiguous",
            count=count,
            column=columns.timestamp,
            examples=_examples(timestamps.naive, raw_timestamps),
            hint="add an offset such as 'Z' or '+07:00', or pass assume_timezone='Asia/Jakarta' "
            "(any IANA zone) if they are local times of one zone",
        )

    row_problems: list[tuple[str, str, BoolArray, str]] = [
        ("missing_timestamp", columns.timestamp, timestamps.missing, "without a timestamp"),
        (
            "invalid_timestamp",
            columns.timestamp,
            timestamps.invalid,
            "with a timestamp that cannot be parsed, lies outside 1677-09-21..2262-04-11, or "
            "names a local time that does not exist or is ambiguous in assume_timezone",
        ),
        ("invalid_value", columns.value, value_invalid, "with a value that is not a finite number"),
        ("missing_identifier", columns.metric, metric.missing, "without a metric name"),
        ("invalid_identifier", columns.metric, metric.invalid, _INVALID_IDENTIFIER),
        ("missing_identifier", columns.host, host.missing, "without a host"),
        ("invalid_identifier", columns.host, host.invalid, _INVALID_IDENTIFIER),
    ]
    for column, optional in ((columns.resource, resource), (columns.unit, unit)):
        if column is not None and optional is not None:
            row_problems.append(
                ("invalid_identifier", column, optional.invalid, _INVALID_IDENTIFIER)
            )
    for name, parsed in labels.items():
        row_problems.append(("invalid_label", name, parsed.invalid, _INVALID_IDENTIFIER))

    unusable = timestamps.naive.copy()
    for code, column, mask, message in row_problems:
        count = int(mask.sum())
        if not count:
            continue
        if on_invalid == "raise":
            log.add(
                code,
                Severity.ERROR,
                f"{_n(count, 'row')} {message}",
                count=count,
                column=column,
                examples=_examples(mask, data[column]),
                hint=_FIX_OR_DROP,
            )
        else:
            log.add(
                code,
                Severity.WARNING,
                f"dropped {_n(count, 'row')} {message}",
                count=count,
                column=column,
                examples=_examples(mask, data[column]),
            )
        unusable |= mask

    no_value = value_missing & ~unusable
    if no_value.any():
        count = int(no_value.sum())
        log.add(
            "missing_value",
            Severity.WARNING,
            f"dropped {_n(count, 'row')} without a value; they become gaps, which are never filled",
            count=count,
            column=columns.value,
            examples=_examples(no_value, data[columns.value]),
        )

    keep = ~(unusable | value_missing)
    _report_time_zones(
        log,
        columns.timestamp,
        timestamps.converted & keep,
        timestamps.source_zones,
        timestamps.localized & keep,
        assume_timezone,
    )
    rows = np.flatnonzero(keep)
    table: dict[str, object] = {
        TIMESTAMP: timestamps.utc.iloc[rows].reset_index(drop=True),
        METRIC: pd.Series(metric.text[rows], dtype="str"),
        HOST: pd.Series(host.text[rows], dtype="str"),
        RESOURCE: _absent_or(resource, rows),
    }
    for name in columns.labels:
        table[name] = pd.Series(labels[name].text[rows], dtype="str")
    table[UNIT] = _absent_or(unit, rows)
    table[VALUE] = values[rows]
    return pd.DataFrame(table)


def _report_time_zones(
    log: _IssueLog,
    column: str,
    converted: BoolArray,
    zones: tuple[str, ...],
    localized: BoolArray,
    assume_timezone: str | None,
) -> None:
    if converted.any():
        count = int(converted.sum())
        log.add(
            "timezone_converted",
            Severity.INFO,
            f"converted {_n(count, 'timestamp')} from {', '.join(zones)} to UTC; "
            "the instants are unchanged",
            count=count,
            column=column,
        )
    if localized.any() and assume_timezone is not None:
        count = int(localized.sum())
        log.add(
            "timezone_assumed",
            Severity.INFO,
            f"interpreted {_n(count, 'timestamp')} without a time zone as {assume_timezone} "
            "(assume_timezone) and converted them to UTC",
            count=count,
            column=column,
        )


def _normalize_series(
    table: pd.DataFrame, label_keys: tuple[str, ...], log: _IssueLog, *, duplicates: Duplicates
) -> pd.DataFrame:
    """Sort by series and time, check units, and collapse or resolve repeated timestamps."""
    n = len(table)
    if n == 0:
        return table
    key_columns = [METRIC, HOST, RESOURCE, *label_keys]
    codes = [pd.factorize(table[column], sort=True)[0] for column in key_columns]
    timestamps = np.asarray(table[TIMESTAMP].astype("int64"), dtype=np.int64)
    # np.lexsort sorts by the last key first; the input position breaks every remaining tie.
    order = np.lexsort([np.arange(n), timestamps, *reversed(codes)])
    new_series = np.zeros(n, dtype=bool)
    new_series[0] = True
    for code in codes:
        ordered = code[order]
        new_series[1:] |= ordered[1:] != ordered[:-1]
    series = np.cumsum(new_series, dtype=np.int64) - 1

    in_input_order = np.empty(n, dtype=np.int64)
    in_input_order[order] = series
    latest_so_far = pd.Series(timestamps).groupby(in_input_order).cummax()
    previous_latest = latest_so_far.groupby(in_input_order).shift(1, fill_value=_INT64_MIN)
    late = timestamps < previous_latest.to_numpy()
    if late.any():
        count = int(late.sum())
        log.add(
            "out_of_order",
            Severity.INFO,
            f"sorted {_n(count, 'row')} that arrived after a later row of the same series",
            count=count,
            column=TIMESTAMP,
        )

    table = table.take(order).reset_index(drop=True)
    timestamps = timestamps[order]
    _check_units(table, series, new_series, label_keys, log)
    return _resolve_duplicates(table, series, timestamps, label_keys, log, duplicates=duplicates)


def _check_units(
    table: pd.DataFrame,
    series: npt.NDArray[np.int64],
    new_series: BoolArray,
    label_keys: tuple[str, ...],
    log: _IssueLog,
) -> None:
    units = table[UNIT].to_numpy(dtype=object)
    changes = ~new_series[1:] & (units[1:] != units[:-1])
    conflicted = np.unique(series[1:][changes])
    if conflicted.size:
        first_rows = np.flatnonzero(new_series)
        examples = []
        for series_id in conflicted[:_MAX_EXAMPLES]:
            seen = sorted({str(unit) or "(none)" for unit in units[series == series_id]})
            examples.append(f"{_series_label(table, first_rows[series_id], label_keys)}: {seen}")
        log.add(
            "conflicting_units",
            Severity.ERROR,
            f"{_n(int(conflicted.size), 'series')} with more than one unit",
            count=int(conflicted.size),
            column=UNIT,
            examples=examples,
            hint="convert the values to one unit per series before loading them",
        )

    pairs = table.loc[table[UNIT] != "", [METRIC, UNIT]].drop_duplicates()
    unit_counts = pairs.groupby(METRIC)[UNIT].nunique()
    mixed = sorted(unit_counts[unit_counts > 1].index)
    if mixed:
        log.add(
            "inconsistent_units",
            Severity.WARNING,
            f"{_n(len(mixed), 'metric')} with different units in different series",
            count=len(mixed),
            column=UNIT,
            examples=(
                f"{name}: {sorted(pairs.loc[pairs[METRIC] == name, UNIT])}" for name in mixed
            ),
            hint="series of one metric are only comparable across hosts in the same unit",
        )


def _resolve_duplicates(
    table: pd.DataFrame,
    series: npt.NDArray[np.int64],
    timestamps: npt.NDArray[np.int64],
    label_keys: tuple[str, ...],
    log: _IssueLog,
    *,
    duplicates: Duplicates,
) -> pd.DataFrame:
    """Rows are sorted, so repeats of one (series, timestamp) are adjacent."""
    n = len(table)
    repeat = np.zeros(n, dtype=bool)
    repeat[1:] = (series[1:] == series[:-1]) & (timestamps[1:] == timestamps[:-1])
    if not repeat.any():
        return table
    group = np.cumsum(~repeat) - 1
    starts = np.flatnonzero(~repeat)
    sizes = np.diff(np.append(starts, n))
    values = table[VALUE].to_numpy(dtype=np.float64, copy=True)
    differs = values != values[starts[group]]
    conflicting = np.bincount(group, weights=differs.astype(np.float64), minlength=len(starts)) > 0
    drop = repeat & ~conflicting[group]

    exact = int(drop.sum())
    if exact:
        log.add(
            "exact_duplicates",
            Severity.INFO,
            f"collapsed {_n(exact, 'row')} that repeat the series, timestamp and value "
            "of another row",
            count=exact,
        )

    if conflicting.any():
        groups = np.flatnonzero(conflicting)
        examples = [
            f"{_series_label(table, starts[g], label_keys)} at "
            f"{table[TIMESTAMP].iloc[starts[g]].isoformat()}: "
            f"{values[starts[g] : starts[g] + sizes[g]].tolist()}"
            for g in groups[:_MAX_EXAMPLES]
        ]
        in_conflict = conflicting[group]
        if duplicates == "raise":
            log.add(
                "conflicting_duplicates",
                Severity.ERROR,
                f"{_n(len(groups), 'timestamp')} with different values for the same series",
                count=len(groups),
                examples=examples,
                hint="pass duplicates='first', 'last' or 'mean' to choose how to resolve them",
            )
            return table
        if duplicates == "first":
            drop |= in_conflict & repeat
            how = "keeping the first value in input order"
        elif duplicates == "last":
            last_of_group = np.append(~repeat[1:], True)
            drop |= in_conflict & ~last_of_group
            how = "keeping the last value in input order"
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                means = np.bincount(group, weights=values, minlength=len(starts)) / sizes
            if not np.isfinite(means[groups]).all():
                log.add(
                    "aggregation_overflow",
                    Severity.ERROR,
                    "the mean of duplicate values overflowed float64",
                    examples=examples,
                )
                return table
            values[starts[groups]] = means[groups]
            drop |= in_conflict & repeat
            how = "averaging all of their values"
        log.add(
            "conflicting_duplicates",
            Severity.WARNING,
            f"resolved {_n(len(groups), 'timestamp')} with different values for the same series "
            f"by {how}",
            count=len(groups),
            examples=examples,
        )
    table = table.assign(**{VALUE: values})
    return table.loc[~drop].reset_index(drop=True)


def _series_label(table: pd.DataFrame, row: int, label_keys: tuple[str, ...]) -> str:
    resource = str(table[RESOURCE].iloc[row])
    labels = {key: str(table[key].iloc[row]) for key in label_keys if table[key].iloc[row]}
    return series_text(
        str(table[METRIC].iloc[row]), str(table[HOST].iloc[row]), resource or None, labels
    )


def _parse_values(raw: pd.Series) -> tuple[FloatArray, BoolArray, BoolArray]:
    """Return ``(values, missing, invalid)``. Bools count as 0/1; text is parsed as numbers."""
    if isinstance(raw.dtype, pd.CategoricalDtype):
        raw = raw.astype(object)
    dtype = raw.dtype
    missing = raw.isna().to_numpy(dtype=bool, copy=True)  # Copy-on-Write views are read-only
    if pd.api.types.is_bool_dtype(dtype) or _is_number_dtype(dtype):
        numbers = np.asarray(raw.astype("float64"), dtype=np.float64)
    elif pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        numbers = np.full(len(raw), np.nan, dtype=np.float64)
    else:
        missing |= raw.astype("str").str.strip().eq("").to_numpy(dtype=bool)
        numbers = np.asarray(pd.to_numeric(raw, errors="coerce"), dtype=np.float64)
    invalid = ~missing & ~np.isfinite(numbers)
    return numbers, missing, invalid


def _parse_identifiers(raw: pd.Series) -> _Identifiers:
    """Classify each distinct value once; identifier columns repeat a few values many times."""
    codes, uniques = pd.factorize(raw, use_na_sentinel=True)
    text = np.full(len(uniques) + 1, "", dtype=object)  # the extra last slot serves missing cells
    unique_missing = np.zeros(len(uniques) + 1, dtype=bool)
    unique_missing[-1] = True
    unique_invalid = np.zeros(len(uniques) + 1, dtype=bool)
    for position, item in enumerate(uniques):
        if not isinstance(item, str):
            unique_invalid[position] = True
        elif item == "":
            unique_missing[position] = True
        elif identifier_problem(item) is not None:
            unique_invalid[position] = True
        else:
            text[position] = item
    slots = np.where(codes < 0, len(uniques), codes)
    return _Identifiers(text[slots], unique_missing[slots], unique_invalid[slots])


def _absent_or(parsed: _Identifiers | None, rows: npt.NDArray[np.intp]) -> pd.Series:
    if parsed is None:
        return pd.Series(np.full(len(rows), "", dtype=object), dtype="str")
    return pd.Series(parsed.text[rows], dtype="str")


def _is_number_dtype(dtype: np.dtype[Any] | ExtensionDtype) -> bool:
    return (
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def _examples(mask: BoolArray, raw: pd.Series) -> list[str]:
    return [f"row {row}: {_shown(raw.iloc[row])}" for row in np.flatnonzero(mask)[:_MAX_EXAMPLES]]


def _shown(cell: object) -> str:
    """``repr`` of a cell, with numpy scalars shown as plain Python values (nan, not np.float64)."""
    return repr(cell.item() if isinstance(cell, np.generic) else cell)


def _n(count: int, noun: str) -> str:
    """``1 row``, ``2 rows``; ``series`` is its own plural."""
    return f"{count} {noun}" if count == 1 or noun.endswith("series") else f"{count} {noun}s"
