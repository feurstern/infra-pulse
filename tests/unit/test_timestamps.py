"""Timestamp normalization: UTC instants, explicit time zones, and no silent assumptions."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from infrapulse.data.timestamps import check_timezone, parse_timestamps


def utc(*texts: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(text).tz_convert("UTC") for text in texts]


def test_aware_values_become_the_same_instants_in_utc() -> None:
    local = pd.Series(pd.date_range("2026-01-05 07:00", periods=3, freq="h", tz="Asia/Jakarta"))
    parsed = parse_timestamps(local)
    assert parsed.utc.tolist() == local.dt.tz_convert("UTC").tolist()
    assert parsed.utc.tolist()[0] == pd.Timestamp("2026-01-05T00:00:00Z")
    assert parsed.converted.all()
    assert parsed.source_zones == ("Asia/Jakarta",)


def test_utc_input_is_not_reported_as_converted() -> None:
    parsed = parse_timestamps(pd.Series(pd.date_range("2026-01-05", periods=2, freq="h", tz="UTC")))
    assert not parsed.converted.any()
    assert parsed.source_zones == ()


@pytest.mark.parametrize(
    "values",
    [
        pd.Series(["2026-01-05T00:00:00Z"]),
        pd.Series(pd.date_range("2026-01-05", periods=1, tz="UTC").as_unit("s")),
        pd.Series([datetime(2026, 1, 5, tzinfo=UTC)], dtype=object),
    ],
)
def test_result_is_always_utc_nanoseconds(values: pd.Series) -> None:
    assert str(parse_timestamps(values).utc.dtype) == "datetime64[ns, UTC]"


def test_iso_strings_with_offsets() -> None:
    parsed = parse_timestamps(
        pd.Series(
            [
                "2026-01-05T10:00:00+07:00",
                "2026-01-05T03:00:00Z",
                "2026-01-05T01:30:00-01:30",
                "2026-01-05T10:00:00+0700",
                "2026-01-05T10:00:00+07",
                "20260105T030000Z",
            ]
        )
    )
    assert parsed.utc.tolist() == utc(*["2026-01-05T03:00:00Z"] * 6)
    assert parsed.converted.tolist() == [True, False, True, True, True, False]
    assert parsed.source_zones == ("+07", "+0700", "+07:00", "-01:30")


def test_fractional_seconds_of_any_precision_keep_nanoseconds() -> None:
    parsed = parse_timestamps(
        pd.Series(
            ["2026-01-05T00:00:00Z", "2026-01-05T00:00:00.5Z", "2026-01-05T00:00:00.123456789Z"]
        )
    )
    assert parsed.utc.iloc[1].microsecond == 500_000
    assert parsed.utc.iloc[2].nanosecond == 789
    assert not parsed.invalid.any()


def test_strings_without_offset_are_naive_not_assumed_utc() -> None:
    parsed = parse_timestamps(pd.Series(["2026-01-05T10:00:00Z", "2026-01-05T10:00:00"]))
    assert parsed.naive.tolist() == [False, True]
    assert parsed.utc.iloc[0] == pd.Timestamp("2026-01-05T10:00:00Z")
    assert pd.isna(parsed.utc.iloc[1])
    assert not parsed.invalid.any()


def test_date_only_strings_are_naive_not_negative_offsets() -> None:
    parsed = parse_timestamps(pd.Series(["2026-01-01"]))
    assert parsed.naive.tolist() == [True]


def test_assume_timezone_interprets_naive_values() -> None:
    parsed = parse_timestamps(
        pd.Series(["2026-01-05T10:00:00", "2026-01-05T03:00:00Z"]), assume_timezone="Asia/Jakarta"
    )
    assert parsed.utc.tolist() == utc("2026-01-05T03:00:00Z", "2026-01-05T03:00:00Z")
    assert parsed.localized.tolist() == [True, False]
    assert not parsed.naive.any()


def test_naive_datetime64_values() -> None:
    naive = pd.Series(pd.to_datetime(["2026-01-05 10:00", "2026-01-05 11:00"]))
    assert parse_timestamps(naive).naive.all()
    parsed = parse_timestamps(naive, assume_timezone="Asia/Jakarta")
    assert parsed.utc.tolist() == utc("2026-01-05T03:00:00Z", "2026-01-05T04:00:00Z")


def test_nonexistent_and_ambiguous_local_times_are_invalid() -> None:
    # In Europe/Berlin, 02:30 does not exist on 2026-03-29 and happens twice on 2026-10-25.
    parsed = parse_timestamps(
        pd.Series(["2026-03-29T02:30:00", "2026-10-25T02:30:00", "2026-06-01T12:00:00"]),
        assume_timezone="Europe/Berlin",
    )
    assert parsed.invalid.tolist() == [True, True, False]
    assert parsed.utc.iloc[2] == pd.Timestamp("2026-06-01T10:00:00Z")


def test_missing_and_blank_values() -> None:
    parsed = parse_timestamps(pd.Series([None, np.nan, "", "   ", pd.NaT], dtype=object))
    assert parsed.missing.all()
    assert not parsed.invalid.any()
    assert parsed.utc.isna().all()


@pytest.mark.parametrize(
    "text",
    [
        "yesterday",
        "12:00",
        "2026-13-45T00:00:00Z",
        "2026-01-05T10:00:00 UTC",
        "2026-01-05T10:00:00+07:00:00",
        "9999-12-31T00:00:00Z",  # beyond the nanosecond range (2262-04-11)
        "1500-01-01T00:00:00Z",
    ],
)
def test_unusable_text_is_invalid(text: str) -> None:
    parsed = parse_timestamps(pd.Series([text, "2026-01-05T00:00:00Z"]))
    assert parsed.invalid.tolist() == [True, False]
    assert not parsed.naive.any()


def test_datetime_objects_in_an_object_column() -> None:
    parsed = parse_timestamps(
        pd.Series(
            [
                datetime(2026, 1, 5, 10, tzinfo=ZoneInfo("Asia/Jakarta")),
                pd.Timestamp("2026-01-05T03:00:00Z"),
                datetime(2026, 1, 5, 3),  # noqa: DTZ001
            ],
            dtype=object,
        )
    )
    assert parsed.utc.tolist()[:2] == utc("2026-01-05T03:00:00Z", "2026-01-05T03:00:00Z")
    assert parsed.naive.tolist() == [False, False, True]


def test_objects_that_are_not_timestamps_are_invalid() -> None:
    parsed = parse_timestamps(
        pd.Series([1767571200, b"2026", "2026-01-05T00:00:00Z"], dtype=object)
    )
    assert parsed.invalid.tolist() == [True, True, False]


def test_categorical_text_is_parsed() -> None:
    values = pd.Series(["2026-01-05T00:00:00Z", "2026-01-05T00:01:00Z"], dtype="category")
    assert parse_timestamps(values).utc.tolist() == utc("2026-01-05T00:00Z", "2026-01-05T00:01Z")


def test_input_index_is_ignored() -> None:
    values = pd.Series(["2026-01-05T00:00:00Z", "nope"], index=[10, 20])
    parsed = parse_timestamps(values)
    assert parsed.utc.index.tolist() == [0, 1]
    assert parsed.invalid.tolist() == [False, True]


@pytest.mark.parametrize("name", ["Mars/Olympus", "", "../etc/passwd", "+07:00"])
def test_unknown_time_zone_names_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match="unknown time zone"):
        check_timezone(name)
    with pytest.raises(ValueError, match="unknown time zone"):
        parse_timestamps(pd.Series(["2026-01-05T00:00:00"]), assume_timezone=name)


def test_an_hour_only_time_with_an_offset_is_aware() -> None:
    # Regression: "T10Z" was once read as naive, and pandas then rejected the whole naive batch.
    parsed = parse_timestamps(
        pd.Series(["2026-01-05T10Z", "2026-01-05T10:00:00", "2026-01-05 17+07"]),
        assume_timezone="Asia/Jakarta",
    )
    assert parsed.utc.tolist() == utc(
        "2026-01-05T10:00:00Z", "2026-01-05T03:00:00Z", "2026-01-05T10:00:00Z"
    )
    assert parsed.localized.tolist() == [False, True, False]
    assert not parsed.invalid.any()
