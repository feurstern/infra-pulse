"""MetricObservation and SeriesKey: identifier and value rules, immutability and identity."""

import dataclasses
from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from infrapulse import DataValidationError, MetricObservation, SeriesKey

T0 = datetime(2026, 1, 5, 12, 0, tzinfo=UTC)


def observation(**changes: object) -> MetricObservation:
    fields: dict[str, object] = {
        "timestamp": T0,
        "metric": "cpu_usage",
        "value": 42.0,
        "host": "vps-01",
    }
    return MetricObservation(**(fields | changes))


def test_valid_observation_keeps_its_fields() -> None:
    obs = observation(resource="eth0", labels={"az": "a"}, unit="percent")
    assert obs.timestamp == T0
    assert (obs.metric, obs.value, obs.host) == ("cpu_usage", 42.0, "vps-01")
    assert (obs.resource, dict(obs.labels), obs.unit) == ("eth0", {"az": "a"}, "percent")


def test_timestamp_without_time_zone_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="no time zone"):
        observation(timestamp=datetime(2026, 1, 5, 12, 0))  # noqa: DTZ001


@pytest.mark.parametrize("timestamp", ["2026-01-05T12:00:00Z", np.datetime64("2026-01-05"), 0])
def test_timestamp_must_be_a_datetime(timestamp: object) -> None:
    with pytest.raises(TypeError, match="timestamp must be a datetime"):
        observation(timestamp=timestamp)


def test_missing_timestamp_is_rejected() -> None:
    with pytest.raises(DataValidationError, match="missing"):
        observation(timestamp=pd.NaT)


def test_pandas_timestamp_with_time_zone_is_accepted() -> None:
    stamp = pd.Timestamp("2026-01-05T19:00:00.000000001+07:00")
    assert observation(timestamp=stamp).timestamp == stamp


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), np.float64("nan")])
def test_value_must_be_finite(value: float) -> None:
    with pytest.raises(DataValidationError, match="finite"):
        observation(value=value)


@pytest.mark.parametrize("value", ["1.5", None, Decimal("1.5"), [1.0]])
def test_value_must_be_a_real_number(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        observation(value=value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3, 3.0), (True, 1.0), (np.int64(5), 5.0), (np.bool_(False), 0.0), (np.float32(0.5), 0.5)],
)
def test_numbers_and_bools_are_stored_as_float(value: object, expected: float) -> None:
    stored = observation(value=value).value
    assert stored == expected
    assert type(stored) is float


@pytest.mark.parametrize("field", ["metric", "host", "resource", "unit"])
@pytest.mark.parametrize(
    "bad", ["", " vps-01", "vps-01 ", "vps\t01", "vps-01\n", "vps\x0001", "\u00a0vps"]
)
def test_identifiers_are_rejected_not_stripped(field: str, bad: str) -> None:
    with pytest.raises(DataValidationError, match=field):
        observation(**{field: bad})


@pytest.mark.parametrize("field", ["metric", "host", "resource", "unit"])
def test_identifiers_must_be_strings(field: str) -> None:
    with pytest.raises(TypeError, match=f"{field} must be a str"):
        observation(**{field: 101})


def test_unicode_and_inner_spaces_are_allowed() -> None:
    obs = observation(host="höst 01", metric="disk used", labels={"zone": "jakarta-ä"})
    assert obs.host == "höst 01"


def test_labels_are_a_sorted_read_only_copy() -> None:
    labels = {"b": "2", "a": "1"}
    obs = observation(labels=labels)
    labels["c"] = "3"
    assert list(obs.labels) == ["a", "b"]
    with pytest.raises(TypeError):
        obs.labels["c"] = "3"


@pytest.mark.parametrize(
    "key", ["timestamp", "metric", "value", "host", "resource", "unit", "count"]
)
def test_reserved_label_keys_are_rejected(key: str) -> None:
    with pytest.raises(DataValidationError, match="reserved"):
        observation(labels={key: "x"})


@pytest.mark.parametrize(
    ("labels", "error"),
    [
        ({"az": ""}, DataValidationError),
        ({" az": "a"}, DataValidationError),
        ({"az": "a "}, DataValidationError),
        ({"az": 1}, TypeError),
        ({1: "a"}, TypeError),
        ([("az", "a")], TypeError),
    ],
)
def test_invalid_labels_are_rejected(labels: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        observation(labels=labels)


def test_equality_and_hash_ignore_label_order() -> None:
    first = observation(labels={"a": "1", "b": "2"})
    second = observation(labels={"b": "2", "a": "1"})
    assert first == second
    assert hash(first) == hash(second)
    assert first != observation(labels={"a": "1", "b": "3"})


def test_the_same_instant_in_another_zone_is_equal() -> None:
    jakarta = observation(timestamp=T0.astimezone(ZoneInfo("Asia/Jakarta")))
    assert jakarta == observation()
    assert hash(jakarta) == hash(observation())


def test_observation_is_immutable() -> None:
    obs = observation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.value = 1.0


def test_key_identifies_the_series() -> None:
    obs = observation(resource="eth0", labels={"az": "a"}, unit="percent", value=1.0)
    assert obs.key == SeriesKey("cpu_usage", "vps-01", "eth0", {"az": "a"})
    assert obs.key in {SeriesKey("cpu_usage", "vps-01", "eth0", {"az": "a"})}


def test_series_key_text() -> None:
    key = SeriesKey("cpu_usage", "vps-01", "eth0", {"zone": "b", "az": "a"})
    assert str(key) == 'cpu_usage{host="vps-01",resource="eth0",az="a",zone="b"}'
    assert str(SeriesKey("up", "vps-01")) == 'up{host="vps-01"}'
    assert str(SeriesKey("up", 'a"b\\c')) == 'up{host="a\\"b\\\\c"}'


def test_series_key_validates_its_parts() -> None:
    with pytest.raises(DataValidationError):
        SeriesKey("cpu", "")
    with pytest.raises(DataValidationError, match="reserved"):
        SeriesKey("cpu", "vps-01", labels={"unit": "percent"})
