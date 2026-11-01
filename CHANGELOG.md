# Changelog

All notable changes to InfraPulse are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/). While the version is `0.x`, minor releases may change
the public API; any such change is listed under *Changed* or *Removed*.

## [Unreleased]

### Added

- Continuous integration (GitHub Actions), brought forward from Milestone 6. `CONTRIBUTING.md`
  describes how to protect `master` with it. The workflow runs:
  - formatting, lint and type checks;
  - tests on Python 3.11-3.14 with a 95% coverage floor;
  - tests against the lowest supported numpy and pandas;
  - a package build with a smoke test of the installed wheel;
  - a single "All checks passed" status check that `master` requires.
- Project foundation:
  - packaging with uv and `uv_build`, for Python 3.11+;
  - ruff, strict mypy and pytest configuration;
  - Apache-2.0 licence;
  - architecture and data-model documentation.
- `MetricObservation` and `SeriesKey`: frozen records with identifier, timestamp and value
  validation.
- `MetricFrame`: the validated canonical long table.
  - Build it with `from_dataframe`, which maps columns to roles, or `from_observations`.
  - Timestamps are normalized to UTC nanoseconds. Naive timestamps are never guessed, and
    `assume_timezone` lets you name their zone. DST-invalid local times and numeric epochs are
    rejected.
  - Duplicates are detected on the UTC instant. Policies: `raise`, `first`, `last`, `mean`.
  - Rows are sorted deterministically, each series must have one unit, and unmapped columns are
    rejected.
- `ValidationReport`, `ValidationIssue` and `Severity`: every problem has a stable code, a count,
  examples and a hint, and all problems are collected before raising.
- `regularize` and `RegularFrame`:
  - epoch-aligned UTC buckets with seven aggregations and a sample `count`;
  - explicit gaps that are never filled, including trailing gaps for series that stop reporting;
  - `start`/`end` windows, a `max_buckets` guard, an overflow check, and `gaps()`.
- `series_profile` (sampling intervals, irregularity, constant series) and `infer_interval`.
