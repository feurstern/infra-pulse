# InfraPulse

[![CI](https://github.com/feurstern/infra-pulse/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/feurstern/infra-pulse/actions/workflows/ci.yml)

CPU-first, explainable anomaly detection for infrastructure metrics, as a Python library.

InfraPulse is meant to sit *next to* the monitoring you already run (Prometheus, Grafana, Beszel,
Proxmox notifications). It does not collect metrics, draw dashboards or send alerts. It reads the
telemetry you already have and will add an analysis layer:
- baselines that adapt to each series instead of static thresholds;
- detection of unusual behaviour that is aware of workload and seasonality;
- correlation across metrics and hosts;
- findings that show their evidence;
- a way to measure detection quality, false alerts per day and detection delay on your own data.

It targets small VPS, bare-metal and ISP environments: limited history, noisy or incomplete
metrics, and no GPU.

An anomaly is a statistical observation, not a diagnosis. InfraPulse will report *what* deviated
and by how much. It never claims a failure or a root cause.

## Status

**Pre-alpha. Not on PyPI. Do not use it in production yet.**

| Milestone | Scope | State |
|---|---|---|
| 0 | Repository foundation, tooling, architecture, CI | done |
| 1 | Data model, validation, timestamp normalization, resampling | **done** |
| 2 | First detectors (MAD, EWMA) and the detector interface | planned |
| 3 | Normalized anomaly results, severity, evidence-based explanations | planned |
| 4 | Evaluation: precision/recall, false alerts per day, detection delay | planned |
| 5 | Rolling baselines, Isolation Forest, seasonal baselines | planned |
| 6 | Prometheus adapter, documentation site, first release candidate | planned |

There are no anomaly detectors yet. The library currently validates and prepares data for them.

## Install

From a checkout of this repository, with [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # creates .venv with InfraPulse and the development tools
```

or into an existing environment: `pip install .`. InfraPulse needs Python 3.11+, numpy and
pandas 3. It needs no GPU, database or server.

## Quick start

```python
import pandas as pd

import infrapulse as ip

raw = pd.DataFrame(
    {
        "timestamp": [
            "2026-01-05T09:00:00+07:00",  # the same instant as 02:00Z
            "2026-01-05T02:01:00Z",
            "2026-01-05T02:04:00Z",
            "2026-01-05T02:04:30Z",
        ],
        "metric": ["cpu_usage"] * 4,
        "host": ["vps-01"] * 4,
        "value": [12.0, 14.5, 80.0, None],
        "unit": ["percent"] * 4,
    }
)

frame = ip.MetricFrame.from_dataframe(raw)  # bad data raises DataValidationError with a full report
print(frame.report.summary())

grid = ip.regularize(frame, "1min")  # epoch-aligned UTC buckets; empty buckets stay NaN
print(grid.data[["timestamp", "value", "count"]].to_string(index=False))
```

```text
4 input rows -> 3 observations: 0 errors, 1 warning, 1 note
  WARNING missing_value [value]: dropped 1 row without a value; they become gaps, which are never filled
    e.g. row 3: nan
  INFO timezone_converted [timestamp]: converted 1 timestamp from +07:00 to UTC; the instants are unchanged
                timestamp  value  count
2026-01-05 02:00:00+00:00   12.0      1
2026-01-05 02:01:00+00:00   14.5      1
2026-01-05 02:02:00+00:00    NaN      0
2026-01-05 02:03:00+00:00    NaN      0
2026-01-05 02:04:00+00:00   80.0      1
```

[`examples/quickstart.py`](examples/quickstart.py) shows more: several samples in one bucket, a
second host that stops reporting, and `gaps()`.

## What the data layer guarantees

- **Nothing changes silently.** Every dropped, collapsed, reordered or converted row appears in the
  `ValidationReport`. Problems that would make the data wrong raise `DataValidationError`, which
  lists *all* of them at once.
- **Time is unambiguous.** Timestamps are stored as UTC instants with nanosecond precision.
  Timestamps without a time zone are rejected unless you name their zone with
  `assume_timezone="Asia/Jakarta"`. Local times that DST makes nonexistent or ambiguous are
  reported, never guessed.
- **Gaps stay visible.** `regularize` never interpolates or forward-fills. An empty bucket has
  `value` NaN and `count` 0. A host that stops reporting gets empty trailing buckets up to the end
  of the data, so an outage is not hidden by the grid simply ending.
- **Series do not merge by accident.** A series is identified by metric, host, optional resource
  and labels. A column you did not map is an error, not silently dropped.
- **Results are deterministic.** The same rows in any order produce the same frame.

The exact schema, validation rules, issue codes and resampling semantics are in
[docs/data-model.md](docs/data-model.md). The design decisions are in
[docs/architecture.md](docs/architecture.md).

## Development

```bash
uv sync                      # environment with the dev tools
uv run ruff format --check . # formatting
uv run ruff check .          # lint
uv run mypy                  # strict type check of src/
uv run pytest                # tests (warnings are errors)
```

GitHub Actions runs the same checks on every pull request. They also cover Python 3.11-3.14, the
lowest supported numpy and pandas, and the package build, and they must pass before anything is
merged into `master`. See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and branch
protection, and [CHANGELOG.md](CHANGELOG.md) for changes.

## Licence and author

Apache-2.0, see [LICENSE](LICENSE). Developed by [feurstern](https://github.com/feurstern).
