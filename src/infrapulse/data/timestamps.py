"""Turn a timestamp column into UTC instants at nanosecond resolution.

Policy (see ``docs/data-model.md``):

* timezone-aware values are converted to UTC. The instant never changes, and the source zones or
  offsets are returned so that the conversion can be reported;
* values without a time zone are *naive*. They are only interpreted when the caller names their
  zone with ``assume_timezone``; pandas itself would silently treat them as UTC;
* local times that do not exist or are ambiguous in that zone (DST transitions) are invalid, as are
  unparseable values and instants outside the nanosecond range (1677-09-21 to 2262-04-11).

Numeric columns (epoch seconds, milliseconds, ...) are the caller's concern: their unit is
ambiguous, so they are rejected before they reach this module.
"""

from __future__ import annotations

import re
import zoneinfo
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
import pandas as pd

BoolArray = npt.NDArray[np.bool_]
Rows = npt.NDArray[np.intp]

# A UTC offset ("Z", "+07:00", "+0700", "-05") after a time of day (hh, hh:mm or hh:mm:ss[.f],
# following the "T" or space separator). Requiring the time keeps the "-01" of a date-only string
# such as "2026-01-01" from being read as an offset.
_OFFSET = re.compile(
    r"[Tt\s]\d{2}(?::?\d{2}(?::?\d{2}(?:[.,]\d+)?)?)?\s*([Zz]|[+-]\d{2}(?::?\d{2})?)$"
)
_UTC_NAMES = ("UTC", "Etc/UTC", "Z", "+00:00", "+0000", "+00", "-00:00", "-0000", "-00")
_NS_MIN = pd.Timestamp.min.tz_localize("UTC")
_NS_MAX = pd.Timestamp.max.tz_localize("UTC")


@dataclass(frozen=True)
class ParsedTimestamps:
    """Result of :func:`parse_timestamps`. Every array is aligned with the input rows."""

    utc: pd.Series  # datetime64[ns, UTC] with a RangeIndex; NaT wherever a timestamp is unusable
    missing: BoolArray  # null or blank
    invalid: BoolArray  # unparseable, out of range, or a nonexistent/ambiguous local time
    naive: BoolArray  # no time zone, and no assume_timezone to interpret it
    converted: BoolArray  # timezone-aware in a zone or offset other than UTC
    localized: BoolArray  # naive, interpreted in assume_timezone
    source_zones: tuple[str, ...]  # zones/offsets of the converted rows, sorted


def check_timezone(name: str) -> str:
    """Return ``name`` if it is a known IANA time zone, otherwise raise ``ValueError``."""
    try:
        zoneinfo.ZoneInfo(name)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError, TypeError) as exc:
        raise ValueError(
            f"unknown time zone {name!r}; use an IANA name such as 'UTC' or 'Asia/Jakarta'"
        ) from exc
    return name


def parse_timestamps(values: pd.Series, *, assume_timezone: str | None = None) -> ParsedTimestamps:
    """Parse datetime64 values, ISO 8601 strings or ``datetime`` objects into UTC nanoseconds."""
    if assume_timezone is not None:
        check_timezone(assume_timezone)
    values = values.reset_index(drop=True)
    builder = _Builder(len(values), assume_timezone)
    dtype = values.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        builder.missing[:] = values.isna().to_numpy(dtype=bool)
        rows = np.flatnonzero(~builder.missing)
        builder.store_aware(rows, values.iloc[rows], str(dtype.tz))
    elif pd.api.types.is_datetime64_dtype(dtype):
        builder.missing[:] = values.isna().to_numpy(dtype=bool)
        rows = np.flatnonzero(~builder.missing)
        builder.store_naive(rows, values.iloc[rows])
    else:
        builder.parse_text(values)
    return builder.result()


class _Builder:
    """Collects the outcome of every row; timestamps are held as naive UTC datetime64[ns]."""

    def __init__(self, n: int, assume_timezone: str | None) -> None:
        self.assume_timezone = assume_timezone
        self.utc = np.full(n, np.datetime64("NaT", "ns"))
        self.missing = np.zeros(n, dtype=bool)
        self.invalid = np.zeros(n, dtype=bool)
        self.naive = np.zeros(n, dtype=bool)
        self.converted = np.zeros(n, dtype=bool)
        self.localized = np.zeros(n, dtype=bool)
        self.zones: set[str] = set()

    def parse_text(self, values: pd.Series) -> None:
        stripped = self._as_text(values).str.strip()
        self.missing |= (stripped == "").to_numpy(dtype=bool)
        present = ~self.missing & ~self.invalid
        offsets = stripped.str.extract(_OFFSET, expand=False)
        has_offset = offsets.notna().to_numpy(dtype=bool)

        aware_rows = np.flatnonzero(present & has_offset)
        if len(aware_rows):
            parsed = pd.to_datetime(
                stripped.iloc[aware_rows], format="ISO8601", utc=True, errors="coerce"
            )
            zones = offsets.iloc[aware_rows].str.upper().to_numpy(dtype=object)
            self.store_aware(aware_rows, parsed, zones)

        naive_rows = np.flatnonzero(present & ~has_offset)
        if len(naive_rows):
            try:
                parsed = pd.to_datetime(
                    stripped.iloc[naive_rows], format="ISO8601", errors="coerce"
                )
            except ValueError:  # last resort: an offset spelling that _OFFSET does not know
                self.invalid[naive_rows] = True
                return
            if isinstance(parsed.dtype, pd.DatetimeTZDtype):  # same last resort
                self.invalid[naive_rows] = True
                return
            unparsed = parsed.isna().to_numpy(dtype=bool)
            self.invalid[naive_rows[unparsed]] = True
            self.store_naive(naive_rows[~unparsed], parsed[~unparsed])

    def store_aware(
        self, rows: Rows, values: pd.Series, zones: str | npt.NDArray[np.object_]
    ) -> None:
        ok = self._store(rows, values)
        zone_of_row = np.full(len(rows), zones, dtype=object) if isinstance(zones, str) else zones
        foreign = ok & ~np.isin(zone_of_row, _UTC_NAMES)
        self.converted[rows[foreign]] = True
        self.zones.update(str(zone) for zone in np.unique(zone_of_row[foreign]))

    def store_naive(self, rows: Rows, local: pd.Series) -> None:
        if self.assume_timezone is None:
            self.naive[rows] = True
            return
        aware = local.dt.tz_localize(self.assume_timezone, ambiguous="NaT", nonexistent="NaT")
        ok = self._store(rows, aware)
        self.localized[rows[ok]] = True

    def result(self) -> ParsedTimestamps:
        return ParsedTimestamps(
            utc=pd.Series(pd.DatetimeIndex(self.utc).tz_localize("UTC")),
            missing=self.missing,
            invalid=self.invalid,
            naive=self.naive,
            converted=self.converted,
            localized=self.localized,
            source_zones=tuple(sorted(self.zones)),
        )

    def _store(self, rows: Rows, values: pd.Series) -> BoolArray:
        """Store the in-range values at ``rows``, mark the others invalid; return the mask."""
        values = values.reset_index(drop=True)
        ok = (values.notna() & (values >= _NS_MIN) & (values <= _NS_MAX)).to_numpy(dtype=bool)
        self.invalid[rows[~ok]] = True
        if ok.any():
            utc = values[ok].dt.tz_convert("UTC").dt.tz_localize(None).dt.as_unit("ns")
            self.utc[rows[ok]] = utc.to_numpy(dtype="datetime64[ns]")
        return ok

    def _as_text(self, values: pd.Series) -> pd.Series:
        """Every cell as text (NaN where missing); ``datetime`` objects become ISO 8601 strings."""
        if isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype(object)
        self.missing |= values.isna().to_numpy(dtype=bool)
        if isinstance(values.dtype, pd.StringDtype):
            return values
        if pd.api.types.infer_dtype(values, skipna=True) in ("string", "empty"):
            return values.astype("str")
        cells = values.to_numpy(dtype=object)
        texts = np.full(len(values), None, dtype=object)
        for row in np.flatnonzero(~self.missing):
            cell = cells[row]
            if isinstance(cell, str):
                texts[row] = cell
            elif isinstance(cell, datetime):
                texts[row] = cell.isoformat()
            else:
                self.invalid[row] = True
        return pd.Series(texts, dtype="str")
