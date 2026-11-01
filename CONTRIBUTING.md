# Contributing to InfraPulse

Thank you for helping. InfraPulse aims to be a library that infrastructure engineers can trust, so
correctness, explicit behaviour and reproducibility matter more than the number of features.

## Set up

You need [uv](https://docs.astral.sh/uv/). Python is installed by uv if necessary.

```bash
uv sync
```

## Quality gate

Run all four checks before you propose a change; CI will run the same ones.

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

`uv run pytest --cov=infrapulse --cov-report=term-missing` shows test coverage.

## Conventions

- **No silent data changes.** Anything that drops, converts, reorders or fills data must report
  it, normally as a `ValidationIssue` with a new, documented code. Never interpolate or fill gaps
  implicitly.
- **Tests are deterministic.** Use `np.random.default_rng(seed)`. Never use the global NumPy random
  state; ruff's NPY002 rule enforces this. Do not use the network. Warnings are errors.
- **Edge cases are part of the feature.** Cover missing, duplicate, irregular, constant, empty and
  extreme data, and time zone and DST behaviour where timestamps are involved.
- **Public API.** Only names exported from `infrapulse` are public. Changing one needs a CHANGELOG
  entry and an update to the docs.
- **Dependencies.** Do not add a runtime dependency without explaining why in the pull request.
  Deep-learning frameworks are out of scope for now.
- **Decisions.** Record substantial design decisions in `docs/architecture.md`, giving the problem,
  the decision, the alternatives, the reasons and the limits.
- **Claims.** Do not claim performance or detection quality without a reproducible measurement.

## Pull requests

Keep each pull request to one concern. Describe what changed, why, and how it was tested. Update
`CHANGELOG.md` under *Unreleased*.
