# Data model and preprocessing

This page specifies how InfraPulse represents, validates and resamples metrics. It describes the
behaviour of `infrapulse` 0.1.0.dev0 (Milestone 1). Every rule below is covered by a test in
`tests/unit/`.

## Observations and series

An **observation** is one numeric sample: a timestamp, a metric name, a value, a host, and optionally
a resource, labels and a unit. A **series** is every observation that shares the same identity:

```
series identity = (metric, host, resource, labels)
```

- `metric` is what is measured, e.g. `cpu_usage` or `node_network_receive_bytes_rate`.
- `host` is the node that reports it: a server, VM, router, cluster or customer circuit.
- `resource` is optional. It names the part of the host being measured, e.g. `eth0`, `sda`,
  `vm-101` or `customer-123`.
- `labels` are optional extra dimensions, e.g. `{"job": "node", "vlan": "20"}`. Every label you map
  is part of the identity. A label whose value changes over time, such as a software version, starts
  a new series each time it changes. Leave such columns out instead of mapping them as labels.
- `unit` is an attribute of the series, not part of its identity. A series must have exactly one
  unit.

`SeriesKey` represents an identity in code. Its text form is Prometheus-like:
`cpu_usage{host="vps-01",resource="eth0",vlan="20"}`.

There are two ways to build observations:

- `MetricObservation(timestamp, metric, value, host, resource=None, labels={}, unit=None)` is a
  frozen record for programmatic use. It is validated on creation.
- `MetricFrame` is the batch container that everything else works on. It is built with
  `MetricFrame.from_dataframe(...)` or `MetricFrame.from_observations(...)`. Both run the same
  validation.

## The canonical `MetricFrame` layout

A `MetricFrame` holds a long table with one row per observation:

| column | dtype | meaning |
|---|---|---|
| `timestamp` | `datetime64[ns, UTC]` | the instant of the sample |
| `metric` | `str` | metric name, never empty |
| `host` | `str` | host or node identifier, never empty |
| `resource` | `str` | resource identifier; `""` when absent |
| one column per label key, sorted by name | `str` | label value; `""` when the series has no such label |
| `unit` | `str` | unit; `""` when unknown |
| `value` | `float64` | finite value |

These invariants always hold, and the `MetricFrame` constructor checks them:

- Rows are sorted by series and then time, and the index is a default `RangeIndex`.
- A `(series, timestamp)` pair appears at most once.
- Every series has exactly one unit, and every value is finite.

The same rows in any input order produce exactly the same frame.

Absent optional fields are stored as `""`, not as NaN. pandas' `groupby` drops rows whose key is NaN
by default, which would silently delete every series without a resource. Empty strings can never be
real identifiers (see below), so `""` is unambiguous.

`frame.data` returns a copy. Under pandas Copy-on-Write that copy is cheap, and changing it never
changes the frame.

## Identifier rules

Metric names, hosts, resources, units, label keys and label values must be strings that:

- are not empty;
- have no leading or trailing whitespace;
- contain no control or other non-printable characters (`str.isprintable()`), so tab, newline,
  NUL and no-break space are all rejected.

Unicode letters and inner spaces are allowed. Values are **never stripped or coerced**. A stray
space would otherwise make `"vps-01 "` a different host from `"vps-01"`, so it is reported instead.

Label keys become column names, so they may not be `timestamp`, `metric`, `value`, `host`,
`resource`, `unit` or `count`.

Numeric identifiers, such as a Proxmox VMID stored as an integer, are rejected rather than
converted, because `101.0` and `101` would become different strings. Convert them yourself with
`df[col].astype(str)`.

## Timestamps

All timestamps are converted to **UTC instants at nanosecond resolution**. The conversion never
changes the instant.

| input | result |
|---|---|
| `datetime64[*, tz]` column (any zone) | converted to UTC; reported as `timezone_converted` unless already UTC |
| ISO 8601 text with an offset (`Z`, `+07:00`, `+0700`, `+07`) | parsed and converted to UTC; offsets may differ from row to row |
| `datetime` objects with `tzinfo` | same as text with an offset |
| text, `datetime` objects or `datetime64` **without** a time zone | rejected (`naive_timestamps`) unless you pass `assume_timezone` |
| numbers (epoch seconds, milliseconds, ...) | rejected (`numeric_timestamps`), because the unit is ambiguous |

Notes:

- **Naive timestamps are never assumed to be UTC.** pandas would silently do that. Instead, pass
  the zone they were recorded in, e.g. `assume_timezone="Asia/Jakarta"`, which must be an IANA
  name. This is reported as `timezone_assumed`.
- **DST.** With `assume_timezone`, a local time that does not exist (spring forward) or occurs
  twice (fall back) cannot be mapped to one instant, so the row is `invalid_timestamp`. It is
  never guessed. All grid arithmetic happens in UTC: pandas' local-time `dt.floor` raises on the
  repeated hour of a fall-back night.
- **Range.** Nanosecond resolution limits instants to 1677-09-21..2262-04-11. Values outside that
  range, such as a `9999-12-31` placeholder, are `invalid_timestamp`.
- **Formats.** Text is parsed as ISO 8601 only (`format="ISO8601"`). pandas never guesses the
  format from the first row, so `01/02/2026` is invalid rather than silently month-first. Blank
  text and nulls count as missing.
- **Epoch numbers** must be converted explicitly, e.g.
  `pd.to_datetime(df["ts"], unit="s", utc=True)`.

## Values

- Numeric and boolean columns are used as they are; `True` and `False` count as 1 and 0.
- Text is parsed as numbers. Surrounding spaces are ignored, and `"1e3"` is accepted.
- A missing value (null, NaN or blank text) means the sample has no value. The row is dropped and
  reported as `missing_value`. After `regularize`, it is part of a gap.
- A value that cannot be a finite number is invalid: `"N/A"`, `"1,5"`, `"inf"`, the text `"nan"`,
  ±inf, a duration or a datetime. Load CSVs with pandas' `na_values=` if a token such as `"N/A"`
  really means missing.
- Values are treated as **gauges**. Cumulative counters (for example `*_bytes_total`) must be
  converted to rates before they are loaded, because counter resets are not handled yet. Integers
  beyond ±2⁵³ lose precision in `float64`.

## Validation

`MetricFrame.from_dataframe` maps your columns to roles. The defaults are the canonical names:

```python
MetricFrame.from_dataframe(
    df,
    timestamp="timestamp",
    metric="metric",
    value="value",
    host="host",
    resource="resource",  # used if a column with this name exists
    unit="unit",  # used if a column with this name exists
    labels=(),  # the columns that further identify a series
    assume_timezone=None,  # IANA zone for naive timestamps
    on_invalid="raise",  # or "drop"
    duplicates="raise",  # or "first", "last", "mean"
)
```

Prometheus-style data maps like this:
`from_dataframe(df, metric="__name__", host="instance", labels=["job", "device"])`.

A column that has no role is an error (`unexpected_columns`). Otherwise a forgotten label column
would silently merge distinct series. Drop the columns you do not need before loading.

Every problem becomes a `ValidationIssue`:

- `code`: stable, listed below;
- `severity`;
- `message`;
- `count`;
- `column`;
- up to five `examples`, given as `row <position>: <value>`, where the position counts from 0
  whatever the input index;
- `hint`.

All problems are collected before anything is raised. A `DataValidationError` lists every one of
them in its message and in `error.report`. A successful frame keeps its `ValidationReport` in
`frame.report`. The input DataFrame is never modified.

### Severity

- **error**: the data cannot be used as it is, so validation raises.
- **warning**: rows were dropped or values changed, and the frame is still produced.
- **info**: a normalization happened that loses no information.

### Issue codes

| code | severity | when | can be dropped or resolved |
|---|---|---|---|
| `duplicate_columns` | error | a column name appears twice | no |
| `missing_column` | error | a mapped column does not exist | no |
| `invalid_label_key` | error | a label column name breaks the identifier rules or is reserved | no |
| `unexpected_columns` | error | a column has no role | no |
| `numeric_timestamps` | error | the timestamp column holds numbers | no |
| `naive_timestamps` | error | timestamps without a zone and no `assume_timezone` | no |
| `conflicting_units` | error | one series has more than one unit (including unit vs none) | no |
| `missing_timestamp` | error | null or blank timestamp | `on_invalid="drop"` → warning |
| `invalid_timestamp` | error | unparseable, out of range, or nonexistent or ambiguous local time | `on_invalid="drop"` → warning |
| `invalid_value` | error | value is not a finite number | `on_invalid="drop"` → warning |
| `missing_identifier` | error | null or empty metric or host | `on_invalid="drop"` → warning |
| `invalid_identifier` | error | metric, host, resource or unit breaks the identifier rules | `on_invalid="drop"` → warning |
| `invalid_label` | error | label value breaks the identifier rules | `on_invalid="drop"` → warning |
| `conflicting_duplicates` | error | one series has different values at the same instant | `duplicates="first"`, `"last"` or `"mean"` → warning |
| `missing_value` | warning | value is null, NaN or blank; the row is dropped | (always dropped) |
| `inconsistent_units` | warning | one metric has different units in different series | (kept) |
| `empty_input` | warning | the input has no rows | (empty frame) |
| `exact_duplicates` | info | a row repeats the series, instant and value of another; collapsed to one | (always) |
| `out_of_order` | info | rows arrived out of time order within their series; sorted | (always) |
| `timezone_converted` | info | timestamps converted from a non-UTC zone or offset | (always) |
| `timezone_assumed` | info | naive timestamps interpreted in `assume_timezone` | (always) |
| `aggregation_overflow` | error | a mean of duplicates or a bucket aggregate overflowed `float64` | no |
| `insufficient_data` | error | `infer_interval` found no series with two observations | no |

**Duplicates.** Two rows are duplicates when they have the same series and the same *instant*, so
`07:00+07:00` and `00:00Z` collide. If they also have the same value, they are collapsed (info). If
the values differ, the default is to raise. `first` and `last` keep a row by input order. `mean`
averages all rows that share the instant. Each choice is reported as a warning.

## Resampling with `regularize`

```python
grid = regularize(frame, "5min", aggregation="mean", start=None, end=None, max_buckets=10_000_000)
```

- **Interval.** A fixed duration: a string such as `"30s"`, `"5min"` or `"1h"`, or a
  `timedelta`. It must be positive. Calendar units with no fixed length (`"1M"`, `"MS"`) are
  rejected. The interval is always explicit: `regularize` never infers it.
- **Buckets** are `[t, t + interval)` and labelled by their start `t`. They are **anchored at the
  Unix epoch in UTC**, so a slice of the data gets exactly the buckets of the whole frame. That
  matters for train/test splits and incremental runs. For intervals that divide a day, buckets line
  up with UTC clock boundaries. In time zones with a half-hour offset, hourly buckets therefore
  start at :30 local time.
- **A bucket's value is only known at its end** (`t + interval`). Detection delay must be measured
  from there (Milestone 4).
- **Aggregation.** Choose from `mean`, `median`, `min`, `max`, `sum`, `first` and `last`. `first`
  and `last` follow time order. `count` is always the number of samples in the bucket.
- **Gaps.** A bucket without samples has `value` NaN and `count` 0. **Nothing is interpolated,
  forward-filled or zero-filled.** That holds for `sum` too, which would otherwise turn an outage
  into a 0.
- **Extent.** Each series runs from its own first bucket to the **last bucket of the frame**. A host
  that stops reporting therefore gets empty trailing buckets instead of silently ending. A series
  that starts late gets no leading gaps, because it did not exist yet.
- **`start` (inclusive) and `end` (exclusive)** must be timezone-aware. Only samples in
  `[start, end)` are used.
  - With `start`, *every* series of the frame gets every bucket from `start`, including series
    with no samples in the window (all gaps).
  - Without `start`, a series with no sample in the window is left out.
  - The grid stays epoch-aligned, so the first and last buckets can be partial, and their `count`
    shows it.
- **Safety.** A grid larger than `max_buckets` rows raises before anything is allocated. This
  catches a typo such as `"1s"` for `"1min"`, or one stray `1970-01-01` timestamp that would
  stretch every series back to 1970. An aggregate that overflows `float64` raises
  `aggregation_overflow` instead of returning ±inf.

### Sampling phase and bucket boundaries

Buckets are floored, so a sample belongs to the bucket that contains its timestamp. This matters
when a source's timestamps scatter on *both* sides of the bucket boundaries, for example ±1 s
around each full minute. A sample at 00:00:59.5 then lands in the 00:00 bucket and leaves 00:01
empty. The data is complete, but the grid shows gaps.

Measured with 20,000 one-minute samples (`regularize(frame, "1min")`):

| timestamps | empty buckets |
|---|---|
| ±10 s or ±1 s around each full minute | 25% (and as many buckets with 2 samples) |
| 0-10 s *after* each full minute (collector lag) | 0% |
| fixed phase at :17 ±1 s (Prometheus-style scrape offset) | 0% |
| ±10 s around each full minute, regularized at `"2min"` or `"5min"` | 0% (counts vary by ±1) |

**How to recognise it.** Buckets with `count` 0 sit next to buckets with `count` 2 in data that
`series_profile` shows as regular.

**What to do today.** Regularize such sources at two or more times their sampling interval. A grid
phase option, which would shift the bucket boundaries away from the sampling instants, is proposed
for the next milestone.

`RegularFrame` provides:
- `.data`: the columns `timestamp`, the series columns, `unit`, `value` and `count`;
- `.interval`, `.aggregation`, `.label_keys`, `.series()`;
- `.observed`: True where `count > 0`;
- `.gaps()`: one row per run of empty buckets, with `start`, `end` (exclusive) and
  `missing_intervals`.

## Describing sampling: `series_profile` and `infer_interval`

`series_profile(frame, tolerance=0.25)` returns one row per series, with the series columns,
`unit` and:

| column | meaning |
|---|---|
| `n_observations` | number of samples |
| `first`, `last` | first and last timestamps |
| `median_interval`, `min_interval`, `max_interval` | spacing between consecutive samples; NaT for a single sample |
| `irregular_fraction` | share of intervals more than `tolerance` × median away from the median; NaN for a single sample |
| `constant` | at least two samples, all equal |

Use it to spot irregular sampling, constant series and short histories. Each detector will set its
own minimum history in Milestone 2.

`infer_interval(frame)` returns the median of the per-series median intervals, rounded to whole
seconds when it is at least one second. It raises `insufficient_data` when no series has two
samples. It is only a suggestion: when series are sampled at different rates, look at
`series_profile` and regularize them separately.

## Known limitations

- Values are gauges. Counters need converting to rates first, and integers beyond ±2⁵³ lose
  precision.
- Units are free text. There is no registry of metric names, units or valid ranges (e.g. 0-100 for
  a percentage) yet.
- Bucket boundaries are fixed at the epoch. Sources whose timestamps straddle them need a
  coarser interval (see *Sampling phase and bucket boundaries*).
- One `regularize` call uses one interval and one aggregation for every series. To use different
  ones, split the frame first.
- Grid buckets are UTC-aligned. Local-calendar buckets (local days, DST-aware hours) are not
  supported; seasonal baselines will need explicit local-time handling (Milestone 5).
- `from_observations` builds its table record by record, and timestamp columns that hold Python
  objects are converted cell by cell. For large batches, prefer `from_dataframe` with datetime64
  or ISO 8601 text columns. As a measured reference, validating 1,000,000 rows (100 series) took
  1.3 s with datetime64 timestamps and 2.0 s with ISO text, and `regularize` took 0.24 s, on one
  laptop (Intel Core 7 240H, Python 3.12). This is not a benchmark.
