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

## Pull requests and continuous integration

Work on a branch and open a pull request into `master`. Keep each pull request to one concern.
Describe what changed, why, and how it was tested, and update `CHANGELOG.md` under *Unreleased*.

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) checks every pull request:
- formatting, lint and types;
- the tests on Python 3.11-3.14 with at least 95% coverage;
- the tests against the lowest supported numpy and pandas;
- the package build.

A pull request can only be merged when the **All checks passed** check is green.

## Protecting `master` (maintainers)

`master` is protected by a repository ruleset, not by anything in the code. To set it up, on GitHub
open **Settings → Rules → Rulesets → New ruleset → New branch ruleset**:

1. **Name** `protect master`. Set **Enforcement status** to *Active*. Leave the **bypass list**
   empty, so the rules also apply to administrators.
2. **Target branches:** *Add target → Include by pattern* `master`. If `master` is the default
   branch, *Include default branch* also works.
3. **Rules:**
   - **Restrict deletions.**
   - **Block force pushes.**
   - **Require a pull request before merging.** With a single maintainer, set *Required approvals*
     to 0, because GitHub does not let you approve your own pull request.
   - **Require status checks to pass.** Add **All checks passed** (source *GitHub Actions*). The
     check can only be selected after the workflow has run once, for example on the first pull
     request. Optionally tick *Require branches to be up to date before merging*.
4. **Create.**

From then on, every change reaches `master` through a pull request whose checks have passed, and no
one can force-push to it or delete it.
