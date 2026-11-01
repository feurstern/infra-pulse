"""regularize, RegularFrame, series_profile and infer_interval."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from infrapulse import (
    DataValidationError,
    MetricFrame,
    RegularFrame,
    SeriesKey,
    infer_interval,
    regularize,
    series_profile,
)

AGGREGATIONS = ["mean", "median", "min", "max", "sum", "first", "last"]


def frame_of(samples: dict[str, list[tuple[str, float]]], **options: object) -> MetricFrame:
    """One cpu_usage series per host from ``{host: [(timestamp, value), ...]}``."""
    rows = [
        {"timestamp": stamp, "metric": "cpu_usage", "host": host, "value": value}
        for host, points in samples.items()
        for stamp, value in points
    ]
    return MetricFrame.from_dataframe(pd.DataFrame(rows), **options)


def minutes(*offsets: float, value: float = 1.0) -> list[tuple[str, float]]:
    base = pd.Timestamp("2026-01-05T00:00:00Z")
    return [((base + pd.Timedelta(minutes=m)).isoformat(), value) for m in offsets]


def column(grid: RegularFrame, name: str, host: str | None = None) -> list[object]:
    data = grid.data
    if host is not None:
        data = data[data["host"] == host]
    if name == "timestamp":
        return [stamp.strftime("%H:%M") for stamp in data["timestamp"]]
    return data[name].tolist()


def test_regular_series_has_no_gaps() -> None:
    frame = frame_of({"vps-01": [(t, float(i)) for i, (t, _) in enumerate(minutes(*range(10)))]})
    grid = regularize(frame, "1min")
    assert column(grid, "count") == [1] * 10
    assert column(grid, "value") == [float(i) for i in range(10)]
    assert grid.observed.all()
    assert grid.gaps().empty
    assert list(grid.gaps().columns) == [
        "metric",
        "host",
        "resource",
        "start",
        "end",
        "missing_intervals",
    ]


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("mean", 2.0),
        ("median", 2.0),
        ("min", 1.0),
        ("max", 3.0),
        ("sum", 4.0),
        ("first", 1.0),
        ("last", 3.0),
    ],
)
def test_several_samples_in_a_bucket_are_aggregated(aggregation: str, expected: float) -> None:
    frame = frame_of(
        {
            "vps-01": [
                ("2026-01-05T00:00:40Z", 3.0),  # out of order on purpose: first/last use time order
                ("2026-01-05T00:00:10Z", 1.0),
                ("2026-01-05T00:01:30Z", 10.0),
            ]
        }
    )
    grid = regularize(frame, "1min", aggregation=aggregation)
    assert column(grid, "value") == [expected, 10.0]
    assert column(grid, "count") == [2, 1]
    assert grid.aggregation == aggregation


@pytest.mark.parametrize("aggregation", AGGREGATIONS)
def test_gaps_stay_empty_and_are_never_filled(aggregation: str) -> None:
    frame = frame_of({"vps-01": [*minutes(0, value=1.0), *minutes(3, value=4.0)]})
    grid = regularize(frame, "1min", aggregation=aggregation)
    assert column(grid, "timestamp") == ["00:00", "00:01", "00:02", "00:03"]
    values = column(grid, "value")
    assert values[0] == 1.0
    assert np.isnan(values[1])
    assert np.isnan(values[2])  # "sum" must not turn an empty bucket into 0
    assert values[3] == 4.0
    assert column(grid, "count") == [1, 0, 0, 1]
    assert grid.observed.tolist() == [True, False, False, True]


def test_a_host_that_stops_reporting_gets_trailing_gaps() -> None:
    frame = frame_of({"vps-01": minutes(*range(10)), "vps-02": minutes(*range(5))})
    grid = regularize(frame, "1min")
    assert column(grid, "timestamp", "vps-02")[-1] == "00:09"
    assert column(grid, "count", "vps-02") == [1] * 5 + [0] * 5
    gaps = grid.gaps()
    assert gaps["host"].tolist() == ["vps-02"]
    assert gaps["start"].iloc[0] == pd.Timestamp("2026-01-05T00:05:00Z")
    assert gaps["end"].iloc[0] == pd.Timestamp("2026-01-05T00:10:00Z")
    assert gaps["missing_intervals"].tolist() == [5]


def test_a_late_series_starts_at_its_first_sample() -> None:
    frame = frame_of({"vps-01": minutes(*range(10)), "vps-02": minutes(*range(5, 10))})
    grid = regularize(frame, "1min")
    assert column(grid, "timestamp", "vps-02")[0] == "00:05"
    assert grid.gaps().empty


def test_start_gives_every_series_the_whole_window() -> None:
    frame = frame_of({"vps-01": minutes(*range(10)), "vps-02": minutes(*range(5, 10))})
    grid = regularize(frame, "1min", start="2026-01-05T00:00:00Z")
    assert column(grid, "count", "vps-02") == [0] * 5 + [1] * 5


def test_start_and_end_select_a_window() -> None:
    frame = frame_of({"vps-01": [(t, float(i)) for i, (t, _) in enumerate(minutes(*range(10)))]})
    grid = regularize(
        frame,
        "1min",
        start=datetime(2026, 1, 5, 0, 2, 30, tzinfo=UTC),
        end="2026-01-05T00:06:00Z",
    )
    # The 00:02 bucket only covers [00:02:30, 00:03) of the window, which has no sample.
    assert column(grid, "timestamp") == ["00:02", "00:03", "00:04", "00:05"]
    assert column(grid, "count") == [0, 1, 1, 1]
    assert column(grid, "value")[1:] == [3.0, 4.0, 5.0]


def test_a_series_silent_during_the_window_is_all_gaps_with_start() -> None:
    frame = frame_of({"vps-01": minutes(*range(10)), "vps-02": minutes(*range(4))})
    grid = regularize(frame, "1min", start="2026-01-05T00:06:00Z", end="2026-01-05T00:10:00Z")
    assert column(grid, "count", "vps-02") == [0, 0, 0, 0]


def test_without_start_a_series_with_no_sample_in_the_window_is_left_out() -> None:
    frame = frame_of({"vps-01": minutes(*range(10)), "vps-02": minutes(*range(5, 10))})
    grid = regularize(frame, "1min", end="2026-01-05T00:03:00Z")
    assert set(grid.data["host"]) == {"vps-01"}
    assert column(grid, "timestamp") == ["00:00", "00:01", "00:02"]


def test_an_empty_window_without_end_gives_an_empty_grid() -> None:
    frame = frame_of({"vps-01": minutes(*range(3))})
    grid = regularize(frame, "1min", start="2026-01-06T00:00:00Z")
    assert len(grid) == 0
    assert list(grid.data.columns) == [
        "timestamp",
        "metric",
        "host",
        "resource",
        "unit",
        "value",
        "count",
    ]


def test_the_grid_is_anchored_at_the_epoch() -> None:
    frame = frame_of({"vps-01": [("2026-01-05T10:03:17Z", 1.0), ("2026-01-05T10:13:01Z", 2.0)]})
    assert column(regularize(frame, "5min"), "timestamp") == ["10:00", "10:05", "10:10"]
    starts = regularize(frame, "7min").data["timestamp"].astype("int64")
    assert (starts % pd.Timedelta("7min").value == 0).all()


def test_a_slice_gets_the_same_grid_as_the_whole_frame(
    synthetic_table: Callable[..., pd.DataFrame],
) -> None:
    data = synthetic_table(3, samples=600, drop_fraction=0.2)
    step = pd.Timedelta("7min")
    cut = pd.Timestamp("2026-01-05T04:00:00Z").floor(step)  # a bucket boundary
    whole = regularize(MetricFrame.from_dataframe(data), step).data
    part = regularize(MetricFrame.from_dataframe(data[data["timestamp"] >= cut]), step).data
    expected = whole[whole["timestamp"] >= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(part, expected)


def test_bucket_boundaries_are_exact_to_the_nanosecond() -> None:
    frame = frame_of(
        {
            "vps-01": [
                ("2026-01-05T00:04:59.999999999Z", 1.0),
                ("2026-01-05T00:05:00.000000000Z", 2.0),
            ]
        }
    )
    assert column(regularize(frame, "5min"), "value") == [1.0, 2.0]


def test_timestamps_before_1970_are_floored_correctly() -> None:
    frame = frame_of({"vps-01": [("1969-12-31T23:59:30Z", 1.0), ("1970-01-01T00:00:30Z", 2.0)]})
    grid = regularize(frame, "1min")
    assert grid.data["timestamp"].tolist() == [
        pd.Timestamp("1969-12-31T23:59:00Z"),
        pd.Timestamp("1970-01-01T00:00:00Z"),
    ]


def test_the_dst_fall_back_night_regularizes_cleanly() -> None:
    # Berlin repeats 02:00-03:00 on 2026-10-25. Two hosts' rows are interleaved, as in a scrape.
    local = pd.date_range("2026-10-25 00:00", "2026-10-25 05:00", freq="10min", tz="Europe/Berlin")
    rows = [
        {"timestamp": stamp, "metric": "cpu_usage", "host": host, "value": 1.0}
        for stamp in local
        for host in ("vps-01", "vps-02")
    ]
    frame = MetricFrame.from_dataframe(pd.DataFrame(rows))
    grid = regularize(frame, "30min")
    per_host = grid.data[grid.data["host"] == "vps-01"]
    assert per_host["timestamp"].diff().dropna().eq(pd.Timedelta("30min")).all()
    assert per_host["count"].sum() == len(local)
    # 00:00-05:00 local is six real hours (02:00-03:00 happens twice): 22:00-04:00 UTC,
    # so 12 half-hour buckets plus the one holding the last sample.
    assert len(per_host) == 13


def test_labels_and_units_are_kept() -> None:
    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-05T00:00:00Z", "2026-01-05T00:00:00Z", "2026-01-05T00:02:00Z"],
            "metric": ["disk_io_rate"] * 3,
            "host": ["vps-01"] * 3,
            "resource": ["sda", "sdb", "sda"],
            "mode": ["read", "read", "read"],
            "unit": ["MB/s"] * 3,
            "value": [1.0, 2.0, 3.0],
        }
    )
    grid = regularize(MetricFrame.from_dataframe(data, labels=["mode"]), "1min")
    assert grid.label_keys == ("mode",)
    assert grid.series() == (
        SeriesKey("disk_io_rate", "vps-01", "sda", {"mode": "read"}),
        SeriesKey("disk_io_rate", "vps-01", "sdb", {"mode": "read"}),
    )
    assert set(grid.data["unit"]) == {"MB/s"}
    assert column(grid, "count") == [1, 0, 1, 1, 0, 0]


def test_single_observation_and_constant_series() -> None:
    frame = frame_of({"vps-01": minutes(0, 1, 2, value=5.0), "vps-02": minutes(1)})
    grid = regularize(frame, "1min")
    assert column(grid, "value", "vps-01") == [5.0, 5.0, 5.0]
    assert column(grid, "count", "vps-02") == [1, 0]


def test_an_empty_frame_gives_an_empty_grid() -> None:
    frame = MetricFrame.from_dataframe(
        pd.DataFrame(columns=["timestamp", "metric", "host", "value"])
    )
    grid = regularize(frame, "1min", end="2026-01-05T00:00:00Z")
    assert len(grid) == 0
    assert grid.series() == ()
    assert grid.gaps().empty
    assert regularize(frame, "1min").data.dtypes.astype(str).tolist() == [
        "datetime64[ns, UTC]",
        "str",
        "str",
        "str",
        "str",
        "float64",
        "int64",
    ]


@pytest.mark.parametrize(
    ("interval", "error"),
    [
        ("0s", ValueError),
        ("-5min", ValueError),
        ("1M", ValueError),
        ("MS", ValueError),
        ("soon", ValueError),
        ("NaT", ValueError),
        (300, TypeError),
        (None, TypeError),
    ],
)
def test_invalid_intervals(interval: object, error: type[Exception]) -> None:
    frame = frame_of({"vps-01": minutes(0, 1)})
    with pytest.raises(error):
        regularize(frame, interval)


@pytest.mark.parametrize("interval", [timedelta(minutes=5), pd.Timedelta("5min"), "5min", "300s"])
def test_interval_forms(interval: object) -> None:
    frame = frame_of({"vps-01": minutes(0, 7)})
    grid = regularize(frame, interval)
    assert grid.interval == pd.Timedelta("5min")
    assert column(grid, "timestamp") == ["00:00", "00:05"]


@pytest.mark.parametrize(
    "options",
    [
        {"aggregation": "mode"},
        {"max_buckets": 0},
        {"max_buckets": True},
        {"max_buckets": 1.5},
        {"start": "2026-01-05T00:00:00"},
        {"start": datetime(2026, 1, 5)},  # noqa: DTZ001
        {"start": "2026-01-05T00:05:00Z", "end": "2026-01-05T00:05:00Z"},
        {"start": "whenever"},
    ],
)
def test_invalid_options(options: dict[str, object]) -> None:
    frame = frame_of({"vps-01": minutes(0, 1)})
    with pytest.raises(ValueError):  # noqa: PT011 - each case has its own message
        regularize(frame, "1min", **options)


def test_regularize_needs_a_metric_frame() -> None:
    with pytest.raises(TypeError, match="MetricFrame"):
        regularize(pd.DataFrame(), "1min")


def test_a_stray_1970_timestamp_trips_the_size_guard() -> None:
    frame = frame_of({"vps-01": [*minutes(0, 1, 2), ("1970-01-01T00:00:00Z", 1.0)]})
    with pytest.raises(ValueError, match=r"max_buckets.*1970-01-01") as caught:
        regularize(frame, "1min")
    assert "1970-01-01T00:00:00+00:00" in str(caught.value)


def test_max_buckets_is_the_largest_allowed_grid() -> None:
    frame = frame_of({"vps-01": minutes(0, 9)})
    with pytest.raises(ValueError, match="max_buckets=9"):
        regularize(frame, "1min", max_buckets=9)
    assert len(regularize(frame, "1min", max_buckets=10)) == 10


def test_an_overflowing_aggregate_is_an_error_not_inf() -> None:
    frame = frame_of(
        {"vps-01": [("2026-01-05T00:00:00Z", 1.7e308), ("2026-01-05T00:00:30Z", 1.7e308)]}
    )
    with pytest.raises(DataValidationError, match="overflowed") as caught:
        regularize(frame, "1min", aggregation="sum")
    assert caught.value.report is not None
    assert caught.value.report.codes == {"aggregation_overflow"}
    assert regularize(frame, "1min", aggregation="max").data["value"].tolist() == [1.7e308]


def test_regularize_leaves_the_frame_unchanged() -> None:
    frame = frame_of({"vps-01": minutes(0, 3)})
    before = frame.data
    regularize(frame, "1min")
    pd.testing.assert_frame_equal(frame.data, before)


def test_grid_properties_and_repr() -> None:
    frame = frame_of({"vps-01": minutes(0, 3)})
    grid = regularize(frame, "1min", aggregation="max")
    assert (grid.interval, grid.aggregation, len(grid)) == (pd.Timedelta("1min"), "max", 4)
    assert grid.key_columns == ("metric", "host", "resource")
    assert repr(grid) == "RegularFrame(4 buckets of 0 days 00:01:00, aggregation='max', 2 empty)"
    exposed = grid.data
    exposed.loc[0, "value"] = -1.0
    assert grid.data["value"].iloc[0] == 1.0


def test_gaps_lists_every_run_per_series() -> None:
    frame = frame_of(
        {
            "vps-01": minutes(0, 3, 4, 8),
            "vps-02": minutes(0, 1, 2, 3, 4, 5, 6, 7, 8),
            "vps-03": minutes(1, 3),
        }
    )
    gaps = regularize(frame, "1min").gaps()
    summary = [
        (host, start.strftime("%H:%M"), end.strftime("%H:%M"), missing)
        for host, start, end, missing in gaps[
            ["host", "start", "end", "missing_intervals"]
        ].itertuples(index=False)
    ]
    assert summary == [
        ("vps-01", "00:01", "00:03", 2),
        ("vps-01", "00:05", "00:08", 3),
        ("vps-03", "00:02", "00:03", 1),
        ("vps-03", "00:04", "00:09", 5),
    ]


@pytest.mark.parametrize("seed", range(8))
def test_regularize_never_invents_values(
    seed: int, synthetic_table: Callable[..., pd.DataFrame]
) -> None:
    rng = np.random.default_rng(seed)
    interval = pd.Timedelta(str(rng.choice(["30s", "1min", "5min"])))
    aggregation = str(rng.choice(AGGREGATIONS))
    data = synthetic_table(seed, samples=200, jitter=0.6, drop_fraction=0.3)
    frame = MetricFrame.from_dataframe(data, duplicates="first")
    grid = regularize(frame, interval, aggregation=aggregation).data

    assert grid["count"].sum() == len(frame)
    empty = grid["count"] == 0
    assert grid.loc[empty, "value"].isna().all()
    assert grid.loc[~empty, "value"].notna().all()

    samples = frame.data
    samples["timestamp"] = samples["timestamp"].dt.floor(interval)
    expected = samples.groupby(["host", "timestamp"])["value"].agg(aggregation)
    observed = grid.loc[~empty].set_index(["host", "timestamp"])["value"]
    pd.testing.assert_series_equal(observed, expected, check_names=False, rtol=1e-12)

    last_bucket = grid["timestamp"].max()
    for _, series in grid.groupby("host"):
        assert series["timestamp"].iloc[-1] == last_bucket
        assert series["timestamp"].diff().dropna().eq(interval).all()


def test_series_profile(synthetic_table: Callable[..., pd.DataFrame]) -> None:
    jittered = synthetic_table(1, hosts=1, samples=100, jitter=0.05, drop_fraction=0.0)
    extra = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-05T00:00:00Z", "2026-01-05T00:05:00Z", "2026-01-05T00:00:00Z"]
            ),
            "metric": ["cpu_usage"] * 3,
            "host": ["flat", "flat", "single"],
            "value": [7.0, 7.0, 3.0],
            "unit": ["percent"] * 3,
        }
    )
    frame = MetricFrame.from_dataframe(pd.concat([jittered, extra], ignore_index=True))
    profile = series_profile(frame).set_index("host")

    assert profile.loc["vps-00", "n_observations"] == 100
    assert abs(profile.loc["vps-00", "median_interval"] - pd.Timedelta("1min")) < pd.Timedelta("4s")
    assert profile.loc["vps-00", "irregular_fraction"] == 0.0
    assert not profile.loc["vps-00", "constant"]
    assert profile.loc["flat", "constant"]
    assert profile.loc["flat", "median_interval"] == pd.Timedelta("5min")
    assert profile.loc["single", "n_observations"] == 1
    assert pd.isna(profile.loc["single", "median_interval"])
    assert np.isnan(profile.loc["single", "irregular_fraction"])
    assert not profile.loc["single", "constant"]


def test_series_profile_irregular_fraction() -> None:
    frame = frame_of({"vps-01": minutes(0, 1, 2, 3, 4, 10)})
    profile = series_profile(frame, tolerance=0.25)
    assert profile["irregular_fraction"].tolist() == [0.2]
    assert profile["max_interval"].tolist() == [pd.Timedelta("6min")]


def test_series_profile_of_an_empty_frame() -> None:
    frame = MetricFrame.from_observations([])
    profile = series_profile(frame)
    assert profile.empty
    assert "median_interval" in profile.columns


@pytest.mark.parametrize("tolerance", [-0.1, True, "0.2"])
def test_series_profile_rejects_bad_tolerance(tolerance: object) -> None:
    with pytest.raises(ValueError, match="tolerance"):
        series_profile(frame_of({"vps-01": minutes(0, 1)}), tolerance=tolerance)


def test_infer_interval_ignores_clock_jitter(synthetic_table: Callable[..., pd.DataFrame]) -> None:
    frame = MetricFrame.from_dataframe(synthetic_table(2, jitter=0.03, drop_fraction=0.0))
    assert infer_interval(frame) == pd.Timedelta("1min")


def test_infer_interval_keeps_subsecond_intervals() -> None:
    stamps = pd.date_range("2026-01-05", periods=20, freq="250ms", tz="UTC")
    frame = MetricFrame.from_dataframe(
        pd.DataFrame({"timestamp": stamps, "metric": "rtt", "host": "vps-01", "value": 1.0})
    )
    assert infer_interval(frame) == pd.Timedelta("250ms")


def test_infer_interval_uses_the_median_series() -> None:
    frame = frame_of(
        {
            "vps-01": minutes(*range(5)),
            "vps-02": minutes(*range(5)),
            "vps-03": minutes(0, 5, 10),
        }
    )
    assert infer_interval(frame) == pd.Timedelta("1min")


def test_infer_interval_needs_two_observations_in_some_series() -> None:
    frame = frame_of({"vps-01": minutes(0), "vps-02": minutes(3)})
    with pytest.raises(DataValidationError, match="two observations") as caught:
        infer_interval(frame)
    assert caught.value.report is not None
    assert caught.value.report.codes == {"insufficient_data"}


def test_regular_frame_checks_its_columns() -> None:
    grid = regularize(frame_of({"vps-01": minutes(0, 1)}), "1min")
    with pytest.raises(ValueError, match="columns"):
        RegularFrame(
            grid.data.drop(columns="count"),
            label_keys=(),
            interval=grid.interval,
            aggregation="mean",
        )


def test_argument_types_are_checked() -> None:
    frame = frame_of({"vps-01": minutes(0, 1)})
    with pytest.raises(TypeError, match="MetricFrame"):
        series_profile(frame.data)
    with pytest.raises(TypeError, match="start"):
        regularize(frame, "1min", start=1767571200)


def test_samples_straddling_a_boundary_split_between_buckets() -> None:
    # Pinned behaviour (see docs/data-model.md, "Sampling phase and bucket boundaries"): buckets
    # are floored, so a sample just before a boundary belongs to the earlier bucket.
    frame = frame_of(
        {
            "vps-01": [
                ("2026-01-05T00:00:00Z", 1.0),
                ("2026-01-05T00:01:59Z", 2.0),
                ("2026-01-05T00:02:01Z", 3.0),
            ]
        }
    )
    assert column(regularize(frame, "1min"), "count") == [1, 1, 1]
    jittered = frame_of(
        {
            "vps-01": [
                ("2026-01-05T00:00:01Z", 1.0),
                ("2026-01-05T00:00:59Z", 2.0),
                ("2026-01-05T00:02:01Z", 3.0),
            ]
        }
    )
    assert column(regularize(jittered, "1min"), "count") == [2, 0, 1]
    assert column(regularize(jittered, "2min"), "count") == [2, 1]
