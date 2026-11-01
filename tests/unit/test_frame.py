"""MetricFrame: canonical layout, determinism, immutability and invariants."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from infrapulse import MetricFrame, MetricObservation, SeriesKey

T0 = datetime(2026, 1, 5, tzinfo=UTC)


@pytest.fixture
def frame() -> MetricFrame:
    data = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-05T00:01:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-05T00:00:00Z",
                "2026-01-05T00:00:30Z",
                "2026-01-05T00:00:00Z",
            ],
            "metric": ["net_rx_rate", "net_rx_rate", "cpu_usage", "net_rx_rate", "net_rx_rate"],
            "host": ["router-1", "router-1", "vps-01", "router-1", "router-1"],
            "resource": ["ge-0/0/1", "ge-0/0/1", None, "ge-0/0/2", "ge-0/0/2"],
            "unit": ["Mbit/s", "Mbit/s", "percent", "Mbit/s", "Mbit/s"],
            "vlan": ["10", "10", None, "20", "20"],
            "value": [2.0, 1.0, 50.0, 4.0, 3.0],
        }
    )
    return MetricFrame.from_dataframe(data, labels=["vlan"])


def test_canonical_layout(frame: MetricFrame) -> None:
    data = frame.data
    assert list(data.columns) == [
        "timestamp",
        "metric",
        "host",
        "resource",
        "vlan",
        "unit",
        "value",
    ]
    assert str(data["timestamp"].dtype) == "datetime64[ns, UTC]"
    for column in ("metric", "host", "resource", "vlan", "unit"):
        assert isinstance(data[column].dtype, pd.StringDtype)
    assert data["value"].dtype == np.float64
    assert data.index.equals(pd.RangeIndex(5))
    assert frame.label_keys == ("vlan",)
    assert frame.key_columns == ("metric", "host", "resource", "vlan")


def test_rows_are_sorted_by_series_then_time(frame: MetricFrame) -> None:
    data = frame.data
    assert data["metric"].tolist() == ["cpu_usage"] + ["net_rx_rate"] * 4
    assert data["resource"].tolist() == ["", "ge-0/0/1", "ge-0/0/1", "ge-0/0/2", "ge-0/0/2"]
    assert data["value"].tolist() == [50.0, 1.0, 2.0, 3.0, 4.0]


def test_series_keys(frame: MetricFrame) -> None:
    assert frame.series() == (
        SeriesKey("cpu_usage", "vps-01"),
        SeriesKey("net_rx_rate", "router-1", "ge-0/0/1", {"vlan": "10"}),
        SeriesKey("net_rx_rate", "router-1", "ge-0/0/2", {"vlan": "20"}),
    )


@pytest.mark.parametrize("seed", range(5))
def test_shuffled_input_gives_an_identical_frame(
    seed: int, synthetic_table: Callable[..., pd.DataFrame]
) -> None:
    data = synthetic_table(seed)
    shuffled = data.sample(frac=1.0, random_state=np.random.default_rng(seed + 100))
    pd.testing.assert_frame_equal(
        MetricFrame.from_dataframe(data).data, MetricFrame.from_dataframe(shuffled).data
    )


def test_data_is_a_protected_copy(frame: MetricFrame) -> None:
    exposed = frame.data
    exposed.loc[0, "value"] = -1.0
    exposed["metric"] = "changed"
    assert frame.data["value"].iloc[0] == 50.0
    assert frame.data["metric"].iloc[0] == "cpu_usage"


def test_the_constructor_keeps_its_own_copy(frame: MetricFrame) -> None:
    source = frame.data
    copy = MetricFrame(source, label_keys=frame.label_keys)
    source.loc[0, "value"] = -1.0
    assert copy.data["value"].iloc[0] == 50.0


def test_round_trip_through_the_canonical_layout(frame: MetricFrame) -> None:
    direct = MetricFrame(frame.data, label_keys=frame.label_keys)
    pd.testing.assert_frame_equal(direct.data, frame.data)
    again = MetricFrame.from_dataframe(frame.data, labels=frame.label_keys)
    pd.testing.assert_frame_equal(again.data, frame.data)
    assert again.report.issues == ()


def test_from_observations_matches_from_dataframe() -> None:
    jakarta = ZoneInfo("Asia/Jakarta")
    observations = [
        MetricObservation(T0 + timedelta(minutes=1), "cpu_usage", 2.0, "vps-01", unit="percent"),
        MetricObservation(
            (T0 + timedelta(minutes=0)).astimezone(jakarta),
            "cpu_usage",
            1,
            "vps-01",
            unit="percent",
        ),
        MetricObservation(T0, "disk_io", 5.0, "vps-01", resource="sda", labels={"mode": "read"}),
    ]
    from_records = MetricFrame.from_observations(observations)
    from_table = MetricFrame.from_dataframe(
        pd.DataFrame(
            {
                "timestamp": [
                    "2026-01-05T00:01:00Z",
                    "2026-01-05T07:00:00+07:00",
                    "2026-01-05T00:00:00Z",
                ],
                "metric": ["cpu_usage", "cpu_usage", "disk_io"],
                "host": ["vps-01"] * 3,
                "resource": [None, None, "sda"],
                "mode": [None, None, "read"],
                "unit": ["percent", "percent", None],
                "value": [2.0, 1.0, 5.0],
            }
        ),
        labels=["mode"],
    )
    pd.testing.assert_frame_equal(from_records.data, from_table.data)
    assert from_records.report.codes == {"out_of_order", "timezone_converted"}


def test_from_observations_with_duplicates() -> None:
    first = MetricObservation(T0, "cpu_usage", 1.0, "vps-01")
    second = MetricObservation(T0, "cpu_usage", 3.0, "vps-01")
    with pytest.raises(ValueError, match="conflicting_duplicates"):
        MetricFrame.from_observations([first, second])
    frame = MetricFrame.from_observations([first, second], duplicates="last")
    assert frame.data["value"].tolist() == [3.0]


def test_from_observations_accepts_any_iterable_and_nothing() -> None:
    generated = (MetricObservation(T0 + timedelta(minutes=i), "up", 1, "vps-01") for i in range(3))
    assert len(MetricFrame.from_observations(generated)) == 3
    empty = MetricFrame.from_observations([])
    assert len(empty) == 0
    assert empty.report.codes == {"empty_input"}


def test_from_observations_rejects_other_objects() -> None:
    with pytest.raises(TypeError, match="MetricObservation"):
        MetricFrame.from_observations([{"timestamp": T0}])


def test_len_and_repr(frame: MetricFrame) -> None:
    assert len(frame) == 5
    assert repr(frame) == (
        "MetricFrame(5 observations, 3 series, "
        "2026-01-05T00:00:00+00:00 to 2026-01-05T00:01:00+00:00)"
    )
    empty = MetricFrame.from_observations([])
    assert repr(empty) == "MetricFrame(0 observations)"


def _swap_rows(data: pd.DataFrame) -> pd.DataFrame:
    return data.iloc[[1, 0, 2, 3, 4]].reset_index(drop=True)


def _set(column: str, row: int, value: object) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def change(data: pd.DataFrame) -> pd.DataFrame:
        data = data.astype({column: object}) if not isinstance(value, float) else data
        data.loc[row, column] = value
        return data

    return change


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_swap_rows, "sorted"),
        (lambda d: d.assign(timestamp=d["timestamp"].iloc[[0, 1, 1, 3, 4]].to_numpy()), "sorted"),
        (_set("value", 0, float("nan")), "finite"),
        (_set("value", 0, float("inf")), "finite"),
        (lambda d: d.assign(timestamp=d["timestamp"].dt.tz_localize(None)), "datetime64"),
        (lambda d: d.assign(timestamp=d["timestamp"].dt.as_unit("us")), "datetime64"),
        (
            lambda d: d[["metric", "timestamp", "host", "resource", "vlan", "unit", "value"]],
            "columns",
        ),
        (lambda d: d.drop(columns="vlan"), "columns"),
        (lambda d: d.astype({"metric": object}), "str dtype"),
        (lambda d: d.assign(host=d["host"].replace("vps-01", "")), "invalid identifier"),
        (lambda d: d.assign(resource=d["resource"].replace("ge-0/0/1", "ge-0/0/1 ")), "invalid"),
        (lambda d: d.assign(unit=["percent", "Mbit/s", "kbit/s", "Mbit/s", "Mbit/s"]), "one unit"),
        (lambda d: d.set_axis([1, 2, 3, 4, 5]), "RangeIndex"),
    ],
)
def test_the_constructor_checks_the_invariants(
    frame: MetricFrame, change: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MetricFrame(change(frame.data), label_keys=frame.label_keys)


def test_the_constructor_checks_label_keys(frame: MetricFrame) -> None:
    with pytest.raises(ValueError, match="columns"):
        MetricFrame(frame.data)
    with pytest.raises(ValueError, match="sorted and unique"):
        MetricFrame(frame.data, label_keys=["vlan", "vlan"])
    with pytest.raises(TypeError, match="DataFrame"):
        MetricFrame([1, 2, 3])


def test_the_constructor_rejects_invalid_label_keys(frame: MetricFrame) -> None:
    data = frame.data.rename(columns={"vlan": "vlan "})
    with pytest.raises(ValueError, match="invalid label key"):
        MetricFrame(data, label_keys=["vlan "])
