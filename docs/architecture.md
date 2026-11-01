# Architecture

## Purpose and boundaries

InfraPulse is a Python **library** that adds an analysis layer to existing infrastructure
monitoring. It takes telemetry that Prometheus, Beszel, Proxmox or your own exporters already
collect. It will return anomaly findings that carry their evidence: what deviated, from which
baseline, by how much, and for how long.

It deliberately does **not**:
- collect metrics;
- store them;
- draw dashboards;
- send notifications;
- claim root causes.

Those stay with the tools that already do them well. Results will be plain Python objects that any
of those tools, or a Telegram bot, can consume.

Engineering principles:
- CPU-first, with no GPU and no deep-learning framework;
- offline-first, needing no database, broker, server or external AI API;
- few dependencies (numpy and pandas today; scikit-learn when Isolation Forest arrives);
- deterministic results;
- explicit, documented behaviour rather than silent fixes.

## Layers and data flow

```
 your telemetry (DataFrame, CSV, records; Prometheus adapter in M6)
        │
        ▼
 MetricFrame.from_dataframe / from_observations   ── data/          (M1, done)
   validation → ValidationReport                    canonical long table, UTC
        │
        ▼
 regularize · series_profile · infer_interval     ── preprocessing  (M1, done)
   epoch-aligned UTC grid, explicit gaps
        │
        ▼
 detectors: MAD, EWMA → rolling, Isolation Forest  ── detectors/     (M2, M5)
        │
        ▼
 anomaly results: score, severity, evidence        ── scoring/,      (M3)
   explanations built only from detector output       explainability/
        │
        ▼
 evaluation: precision/recall, false alerts/day,   ── evaluation/    (M4)
   detection delay, runtime, memory
```

Each layer depends only on the layers above it. Ingestion adapters will produce `MetricFrame`s and
know nothing about detectors. Detectors will take regularized data and know nothing about where it
came from.

## Package layout

Current, Milestone 1:

```
src/infrapulse/
├── __init__.py        public API (re-exports) and __version__
├── errors.py          InfraPulseError, DataValidationError
├── data/
│   ├── model.py       MetricObservation, SeriesKey, identifier rules
│   ├── timestamps.py  timestamp parsing and UTC normalization (internal)
│   ├── validation.py  ValidationReport/Issue/Severity, table validation (internal)
│   └── frame.py       MetricFrame
└── preprocessing.py   regularize, RegularFrame, series_profile, infer_interval
```

Planned:

- `detectors/` (M2): the detector interface plus MAD and EWMA;
- `scoring/` and `explainability/` (M3);
- `evaluation/` (M4);
- `integrations/` (M6), with optional extras such as `infrapulse[prometheus]`.

The spec's proposed tree is followed with three deliberate deviations:

1. **No `core/` package.** The only cross-cutting module so far is `errors.py`, which sits at the
   top level. A package for one module adds an import level and no value.
2. **`preprocessing` is a module, not a package.** It has one cohesive concern today. When rolling
   statistics and feature extraction arrive with the detectors, it can become a package without
   changing `from infrapulse.preprocessing import ...`.
3. **No empty placeholder packages.** Detector, scoring and evaluation packages will be created
   with their first real code, so the tree never advertises features that do not exist.

## Public API

The public API is exactly the set of names exported from `infrapulse`:

- `MetricObservation`, `SeriesKey`, `MetricFrame`;
- `ValidationReport`, `ValidationIssue`, `Severity`;
- `regularize`, `RegularFrame`, `series_profile`, `infer_interval`;
- `InfraPulseError`, `DataValidationError`;
- `__version__`.

Everything else, including the functions inside `infrapulse.data.timestamps` and
`infrapulse.data.validation`, is internal and may change. Issue `code` strings and the canonical
column layout are part of the public contract.

**Versioning.** Semantic versioning. During `0.x`, a minor release may change the public API, and
the change is listed in `CHANGELOG.md` under *Changed* or *Removed*. Patch releases only fix bugs.
`1.0.0` will be released once the detector and result interfaces have been used on real data.

## Design decisions

Each record follows the same structure: the problem, the decision, the alternatives considered, why
the decision was taken, and its limits.

### D1. A long table with one row per sample

- **Problem.** Infrastructure data arrives as samples from many hosts at different rates, with
  per-series units and arbitrary dimensions (interfaces, disks, VLANs, customers).
- **Decision.** The canonical form is a long table: one row per sample, identified by
  `(metric, host, resource, labels)`. This matches Prometheus and OpenTelemetry data models.
- **Alternatives.**
  - A wide table (one column per metric) cannot hold series with different rates or units, or
    event-like series such as backup durations.
  - One canonical label string per row (`k="v",…`) is compact, but filtering it needs string
    parsing.
  - A separate series table referenced by id is efficient, but awkward to inspect.
- **Why.** It is general, maps 1:1 to the planned adapters, and keeps each label key as a normal
  pandas column (`df[df.vlan == "20"]`).
- **Limits.** The number of label columns grows with the union of label keys. Multivariate detectors
  (M5) will need a wide view built by pivoting. That is safe because each (series, timestamp) pair
  is unique, so `pivot` is enough and the averaging `pivot_table` is never needed.

### D2. Timestamps are UTC instants; naive time is never guessed

- **Problem.** The spec says to "preserve timestamp semantics and avoid silently changing time
  zones". Data mixes offsets and zones, and sometimes has none.
- **Decision.**
  - Store UTC at nanosecond resolution and report every conversion.
  - Reject naive timestamps unless the caller names their zone with `assume_timezone`.
  - Mark DST-nonexistent or ambiguous local times invalid.
  - Reject numeric epochs.
  - Parse text only as ISO 8601.
- **Alternatives.**
  - Keep each frame's input zone end to end: one zone per frame, more DST code paths.
  - Accept naive input as UTC, which is pandas' behaviour with `utc=True`.
- **Why.**
  - UTC instants make ordering, duplicate detection and grid arithmetic unambiguous. In pandas 3,
    local-time `dt.floor` *raises* on the repeated hour of a DST fall-back night.
  - Silently treating naive local times as UTC is exactly the bug the spec warns about.
  - The user confirmed this choice before implementation.
- **Limits.** Outputs are in UTC, so local display needs an explicit `tz_convert`. Seasonal
  baselines (M5) will need explicit local-time features.

### D3. Validation collects every problem; policies are explicit

- **Problem.** Real metrics are messy: missing and invalid values, duplicates, out-of-order rows.
  Silent fixes hide data problems. Failing on the first problem makes cleanup slow.
- **Decision.**
  - Every check produces a `ValidationIssue` with a stable code, a count, examples and a hint, and
    all of them are collected before raising.
  - Corruption is an error by default: unparseable timestamps, invalid values, bad identifiers. It
    can be dropped explicitly with `on_invalid="drop"`, which reports a warning.
  - Missing values are always dropped with a warning, so they become visible gaps.
  - Conflicting duplicates are an error unless `duplicates=` chooses a resolution.
  - Structural problems cannot be dropped.
- **Alternatives.**
  - Pydantic models: an extra dependency, and validation per record is slow for tables.
  - Raising on the first problem.
  - Lenient defaults (drop everything quietly).
- **Why.** Engineers see every problem at once and decide each trade-off themselves. The report
  stays attached to the frame for audit.
- **Limits.** Strict defaults mean raw exports often need one explicit option such as
  `on_invalid="drop"`. That is intended, because it is visible in the calling code.

### D4. `""` represents an absent optional field

- **Problem.** pandas' default `groupby(dropna=True)` silently deletes rows whose key is NaN. A
  NaN resource would make every series without a resource disappear from per-series operations.
- **Decision.** Absent `resource`, labels and `unit` are stored as `""`. Empty identifiers are
  rejected on input, so `""` cannot be confused with a real value.
- **Alternatives.** NaN plus `dropna=False` everywhere, which is easy to forget in future code.
- **Limits.** Users reading `frame.data` see `""` instead of NaN. `SeriesKey` turns it back into
  `None`.

### D5. Epoch-anchored UTC buckets; gaps are explicit and never filled

- **Problem.**
  - Detectors need regular series.
  - Interpolating or forward-filling invents data, can hide outages, and can leak future values
    into the past.
  - Grids anchored at the data start differ between slices.
- **Decision.**
  - Buckets are `[t, t+interval)`, floored with integer nanosecond arithmetic and anchored at the
    epoch.
  - An empty bucket keeps NaN and `count == 0`.
  - Each series extends to the frame's last bucket, so a host that stops reporting shows trailing
    gaps.
  - `max_buckets` guards against huge grids.
- **Alternatives.**
  - `DataFrame.resample`: its default anchors at the start of the first day, and its empty `sum`
    buckets become 0.
  - Rounding to the nearest bucket, which shifts later samples to earlier labels.
  - Ending each series at its own last sample, which hides a stopped host.
  - Opt-in filling with a flag, deferred until a detector needs it.
- **Why.** Identical grids across slices make incremental runs and time-aware evaluation splits
  consistent. Explicit gaps are evidence: "no data for 15 minutes" is often the finding.
- **Limits.**
  - One interval and one aggregation per call.
  - Buckets follow the UTC clock, so in half-hour time zones hourly buckets start at :30 local.
  - A bucket's value is only known at its end, which matters for detection delay in M4.

### D6. Unmapped columns are errors

- **Problem.** A forgotten label column, such as `device`, would merge distinct series and produce
  conflicting duplicates or interleaved values.
- **Decision.** Every column must have a role, and extra columns raise `unexpected_columns`.
- **Alternatives.** Ignoring extra columns with a warning, or making every extra column a label.
- **Why.** No silent loss of identity, and no magic.
- **Limits.** Metadata columns must be dropped before loading.

### D7. Frozen dataclasses and a small, validated container

- **Problem.** Later milestones need invariants they can rely on, such as sorted, unique, finite
  data in UTC, without re-checking.
- **Decision.**
  - `MetricObservation` and `SeriesKey` are frozen, slotted dataclasses.
  - `MetricFrame` checks its invariants in O(n) on construction and only hands out copies of its
    data, which are cheap under pandas Copy-on-Write.
- **Alternatives.**
  - Pydantic: an extra dependency.
  - Returning a bare DataFrame and documenting the invariants: nothing enforces them.
- **Limits.** Python cannot make the constructor private. Direct construction is allowed for data
  that is already canonical and is checked.

### D8. Python 3.11+ and dependency bounds

- **Problem.** The target hosts include Debian 12 and Proxmox VE 8, which ship Python 3.11. NumPy
  2.5 needs Python 3.12 or newer.
- **Decision.**
  - `requires-python = ">=3.11"` and `numpy>=2.0`. The uv lock resolves numpy 2.4.x on 3.11 and
    2.5.x on 3.12+.
  - `pandas>=3.0`, because the code relies on its `str` dtype and Copy-on-Write.
  - Local development uses 3.12.
- **Limits.**
  - Only the declared lower bounds and the locked versions are tested, not the versions in
    between. The CI `lowest` job installs the lower bounds (numpy 2.0.0 and pandas 3.0.0 on
    Python 3.11); all tests passed with them on 2026-11-01.
  - Python 3.11 rules out 3.12-only syntax.

## Quality gate and tooling

| tool | role | configuration |
|---|---|---|
| uv | environment, lockfile, build (`uv_build`) | `pyproject.toml`, `uv.lock` |
| ruff | format and lint (pycodestyle, pyflakes, isort, bugbear, pyupgrade, pandas-vet, NumPy, DTZ, …) | `[tool.ruff]` |
| mypy | strict type checking of `src/` with `pandas-stubs` | `[tool.mypy]` |
| pytest | tests. Warnings are errors, randomness is seeded, nothing uses the network | `[tool.pytest.ini_options]` |

### Continuous integration

`.github/workflows/ci.yml` runs on every pull request, on every push to `master`, and on demand. It
was brought forward from Milestone 6 so that `master` can be protected now.

| job | checks |
|---|---|
| Format, lint and type check | `uv sync --locked` (the lockfile must be current), `ruff format --check`, `ruff check`, `mypy` |
| Tests (Python 3.11, 3.12, 3.13, 3.14) | the full suite with coverage; under 95% fails |
| Tests (lowest supported numpy and pandas) | the suite on Python 3.11 with every direct dependency at its declared lower bound |
| Build and smoke-test the package | `uv build` (sdist, then a wheel from it), then the example runs against the installed wheel |
| All checks passed | succeeds only if every job above succeeded; this is the one required status check for `master` |

Supply chain: the workflow has read-only permissions, and checkout does not keep the token. Every
action is pinned to a full commit SHA, and the uv version is pinned. Updating them is a deliberate
change.

CI does not deliver anything yet: documentation publishing (mkdocs) and PyPI releases remain part of
Milestone 6.
