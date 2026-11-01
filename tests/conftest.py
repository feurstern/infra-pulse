"""Shared, deterministic test data."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"

SyntheticTable = Callable[..., pd.DataFrame]


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def synthetic_table() -> SyntheticTable:
    """Factory for long tables of jittered, partly missing samples from several hosts.

    Every call with the same arguments returns the same table: randomness comes only from
    ``np.random.default_rng(seed)``.
    """

    def build(
        seed: int = 0,
        *,
        hosts: int = 3,
        samples: int = 120,
        interval: str = "1min",
        jitter: float = 0.2,
        drop_fraction: float = 0.1,
        start: str = "2026-01-05T00:00:00Z",
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        step = pd.Timedelta(interval)
        parts = []
        for index in range(hosts):
            offsets = np.arange(samples) + rng.uniform(-jitter, jitter, samples)
            keep = rng.random(samples) >= drop_fraction
            timestamps = pd.Timestamp(start) + step * offsets[keep]
            parts.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "metric": "cpu_usage",
                        "host": f"vps-{index:02d}",
                        "value": rng.normal(40.0 + 10 * index, 5.0, int(keep.sum())),
                        "unit": "percent",
                    }
                )
            )
        return pd.concat(parts, ignore_index=True)

    return build
