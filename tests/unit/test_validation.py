"""Validation of long tables: every rule, its severity, and what the report says."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from infrapulse import DataValidationError, MetricFrame, Severity, ValidationReport


def table(**columns: object) -> pd.DataFrame:
    """Two clean UTC samples of one series, with any column replaced or added."""
    base: dict[str, object] = {
        "timestamp": ["2026-01-05T00:00:00Z", "2026-01-05T00:01:00Z"],
        "metric": ["cpu_usage", "cpu_usage"],
        "host": ["vps-01", "vps-01"],
        "value": [1.0, 2.0],
    }
    return pd.DataFrame(base | columns)


def rejected(data: pd.DataFrame, **options: object) -> ValidationReport:
    with pytest.raises(DataValidationError) as caught:
        MetricFrame.from_dataframe(data, **options)
    report = caught.value.report
    assert report is not None
    assert report.output_rows == 0
    return report


def test_clean_input_produces_no_issues() -> None:
    frame = MetricFrame.from_dataframe(table())
    assert frame.report.issues == ()
    assert (frame.report.input_rows, frame.report.output_rows) == (2, 2)


def test_raise_reports_every_problem_at_once() -> None:
    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-05T00:00:00Z", "not a time", "2026-01-05T00:02:00Z", None],
            "metric": ["cpu_usage", "cpu_usage", " cpu_usage", "cpu_usage"],
            "host": ["vps-01", None, "vps-01", "vps-01"],
            "value": [1.0, 2.0, "N/A", 4.0],
        }
    )
    report = rejected(data)
    assert {issue.code for issue in report.errors} == {
        "invalid_timestamp",
        "missing_timestamp",
        "invalid_value",
        "invalid_identifier",
        "missing_identifier",
    }
    invalid_value = report.issues_for("invalid_value")[0]
    assert (invalid_value.count, invalid_value.column) == (1, "value")
    assert invalid_value.examples == ("row 2: 'N/A'",)
    assert invalid_value.hint is not None
    assert "on_invalid='drop'" in invalid_value.hint
    assert "5 errors" in report.summary()


def test_drop_removes_invalid_rows_and_downgrades_them_to_warnings() -> None:
    data = pd.DataFrame(
        {
            "timestamp": ["2026-01-05T00:00:00Z", "not a time", "2026-01-05T00:02:00Z"],
            "metric": ["cpu_usage", "cpu_usage", "cpu_usage"],
            "host": ["vps-01", "vps-01", "vps-01"],
            "value": [1.0, 2.0, "N/A"],
        }
    )
    frame = MetricFrame.from_dataframe(data, on_invalid="drop")
    assert frame.data["value"].tolist() == [1.0]
    assert {issue.code for issue in frame.report.warnings} == {"invalid_timestamp", "invalid_value"}
    assert not frame.report.has_errors
    assert frame.report.issues_for("invalid_value")[0].message.startswith("dropped 1 row ")


def test_drop_can_leave_an_empty_frame() -> None:
    frame = MetricFrame.from_dataframe(table(value=["x", "y"]), on_invalid="drop")
    assert len(frame) == 0
    assert list(frame.data.columns) == ["timestamp", "metric", "host", "resource", "unit", "value"]


def test_missing_values_are_dropped_with_a_warning_by_default() -> None:
    data = table(
        timestamp=[
            "2026-01-05T00:00:00Z",
            "2026-01-05T00:01:00Z",
            "2026-01-05T00:02:00Z",
            "2026-01-05T00:03:00Z",
        ],
        metric=["cpu_usage"] * 4,
        host=["vps-01"] * 4,
        value=[1.0, None, np.nan, ""],
    )
    frame = MetricFrame.from_dataframe(data)
    assert frame.data["value"].tolist() == [1.0]
    issue = frame.report.issues_for("missing_value")[0]
    assert (issue.severity, issue.count) == (Severity.WARNING, 3)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["1.5", " 42 "], [1.5, 42.0]),
        (["1e3", "-0.25"], [1000.0, -0.25]),
        ([True, False], [1.0, 0.0]),
        (pd.array([1, 2], dtype="Int64"), [1.0, 2.0]),
        (pd.Series(["7", "8"], dtype="category"), [7.0, 8.0]),
    ],
)
def test_values_are_parsed_as_numbers(raw: object, expected: list[float]) -> None:
    frame = MetricFrame.from_dataframe(table(value=raw))
    assert frame.data["value"].tolist() == expected
    assert frame.data["value"].dtype == np.float64


@pytest.mark.parametrize("bad", ["N/A", "1,5", "inf", "-inf", "nan", "error", float("inf")])
def test_values_that_are_not_finite_numbers_are_invalid(bad: object) -> None:
    report = rejected(table(value=[1.0, bad]))
    assert report.codes == {"invalid_value"}


def test_durations_are_not_silently_converted_to_numbers() -> None:
    report = rejected(table(value=[timedelta(seconds=5), timedelta(seconds=6)]))
    assert report.issues_for("invalid_value")[0].count == 2


def test_numeric_timestamps_are_rejected_with_a_conversion_hint() -> None:
    report = rejected(table(timestamp=[1767571200, 1767571260]), on_invalid="drop")
    issue = report.issues_for("numeric_timestamps")[0]
    assert issue.hint is not None
    assert "unit='s'" in issue.hint


def test_an_all_missing_timestamp_column_is_missing_not_numeric() -> None:
    report = rejected(table(timestamp=[np.nan, np.nan]))
    assert report.issues_for("missing_timestamp")[0].count == 2


def test_naive_timestamps_are_an_error_even_with_drop() -> None:
    data = table(timestamp=["2026-01-05T00:00:00", "2026-01-05T00:01:00Z"])
    report = rejected(data, on_invalid="drop")
    issue = report.issues_for("naive_timestamps")[0]
    assert (issue.count, issue.examples) == (1, ("row 0: '2026-01-05T00:00:00'",))
    assert issue.hint is not None
    assert "assume_timezone" in issue.hint


def test_assume_timezone_is_reported() -> None:
    frame = MetricFrame.from_dataframe(
        table(timestamp=["2026-01-05T07:00:00", "2026-01-05T07:01:00"]),
        assume_timezone="Asia/Jakarta",
    )
    assert frame.data["timestamp"].iloc[0] == pd.Timestamp("2026-01-05T00:00:00Z")
    issue = frame.report.issues_for("timezone_assumed")[0]
    assert (issue.severity, issue.count) == (Severity.INFO, 2)
    assert "Asia/Jakarta" in issue.message


def test_time_zone_conversion_is_reported() -> None:
    frame = MetricFrame.from_dataframe(
        table(timestamp=["2026-01-05T07:00:00+07:00", "2026-01-05T00:01:00Z"])
    )
    issue = frame.report.issues_for("timezone_converted")[0]
    assert (issue.severity, issue.count) == (Severity.INFO, 1)
    assert "+07:00" in issue.message


def test_missing_required_column() -> None:
    report = rejected(table().drop(columns="host"))
    assert report.issues_for("missing_column")[0].column == "host"


def test_timestamps_in_the_index_get_a_hint() -> None:
    data = table()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.pop("timestamp"), utc=True))
    issue = rejected(data).issues_for("missing_column")[0]
    assert issue.hint is not None
    assert "reset_index" in issue.hint


def test_unmapped_columns_are_an_error_not_silently_ignored() -> None:
    data = table(job=["node", "node"])
    report = rejected(data)
    assert report.issues_for("unexpected_columns")[0].examples == ("'job'",)
    frame = MetricFrame.from_dataframe(data, labels=["job"])
    assert frame.label_keys == ("job",)


def test_optional_columns() -> None:
    assert len(MetricFrame.from_dataframe(table())) == 2  # no resource or unit column
    report = rejected(table(), resource="interface")  # explicitly named, so it must exist
    assert report.issues_for("missing_column")[0].column == "interface"


def test_prometheus_style_column_mapping() -> None:
    data = pd.DataFrame(
        {
            "ts": ["2026-01-05T00:00:00Z", "2026-01-05T00:00:00Z"],
            "__name__": ["node_network_receive_bytes_rate"] * 2,
            "instance": ["router-1:9100"] * 2,
            "job": ["node"] * 2,
            "device": ["eth0", "eth1"],
            "v": [10.0, 20.0],
        }
    )
    frame = MetricFrame.from_dataframe(
        data,
        timestamp="ts",
        metric="__name__",
        host="instance",
        value="v",
        labels=["job", "device"],
    )
    assert list(frame.data.columns) == [
        "timestamp",
        "metric",
        "host",
        "resource",
        "device",
        "job",
        "unit",
        "value",
    ]
    assert len(frame.series()) == 2


@pytest.mark.parametrize(
    ("labels", "code"),
    [
        (["unit"], "invalid_label_key"),
        (["job "], "invalid_label_key"),
        (["nope"], "missing_column"),
    ],
)
def test_invalid_label_columns(labels: list[str], code: str) -> None:
    data = table(unit=["percent", "percent"], **{"job ": ["a", "a"]})
    report = rejected(data, labels=labels)
    assert code in report.codes


def test_labels_must_be_a_sequence_of_names() -> None:
    with pytest.raises(TypeError, match="sequence of column names"):
        MetricFrame.from_dataframe(table(job=["a", "a"]), labels="job")


def test_a_column_cannot_have_two_roles() -> None:
    with pytest.raises(ValueError, match="only one role"):
        MetricFrame.from_dataframe(table(), metric="host")


def test_duplicate_column_names() -> None:
    data = pd.concat([table(), table()[["host"]]], axis=1)
    assert rejected(data).codes == {"duplicate_columns"}


@pytest.mark.parametrize(("option", "value"), [("on_invalid", "ignore"), ("duplicates", "max")])
def test_unknown_policies_are_argument_errors(option: str, value: str) -> None:
    with pytest.raises(ValueError, match=option):
        MetricFrame.from_dataframe(table(), **{option: value})


def test_data_must_be_a_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        MetricFrame.from_dataframe([{"timestamp": "2026-01-05T00:00:00Z"}])


@pytest.mark.parametrize(
    ("column", "values", "code"),
    [
        ("host", [" vps-01", "vps-01"], "invalid_identifier"),
        ("host", ["vps-01", 101], "invalid_identifier"),
        ("host", pd.Series([101, 102]), "invalid_identifier"),
        ("metric", ["cpu\tusage", "cpu_usage"], "invalid_identifier"),
        ("resource", ["eth0\n", "eth0"], "invalid_identifier"),
        ("unit", ["percent ", "percent"], "invalid_identifier"),
        ("metric", [None, "cpu_usage"], "missing_identifier"),
        ("host", ["", "vps-01"], "missing_identifier"),
    ],
)
def test_identifier_problems(column: str, values: object, code: str) -> None:
    report = rejected(table(**{column: values}))
    assert report.codes == {code}
    assert report.issues_for(code)[0].column == column


def test_invalid_label_values() -> None:
    report = rejected(table(job=["node", "node "]), labels=["job"])
    assert report.issues_for("invalid_label")[0].examples == ("row 1: 'node '",)


def test_missing_optional_cells_mean_absent() -> None:
    data = table(resource=[None, "eth0"], unit=[np.nan, np.nan], job=["", "node"])
    frame = MetricFrame.from_dataframe(data, labels=["job"])
    assert frame.data["resource"].tolist() == ["", "eth0"]
    assert frame.data["unit"].tolist() == ["", ""]
    assert frame.data["job"].tolist() == ["", "node"]
    assert frame.report.issues == ()


def test_an_entirely_empty_optional_column_is_absent() -> None:
    data = table(resource=[np.nan, np.nan])  # how read_csv loads an empty column
    assert data["resource"].dtype == np.float64
    frame = MetricFrame.from_dataframe(data)
    assert frame.data["resource"].tolist() == ["", ""]


def test_exact_duplicates_are_collapsed() -> None:
    data = pd.concat([table(), table().iloc[[0]]], ignore_index=True)
    frame = MetricFrame.from_dataframe(data)
    assert len(frame) == 2
    issue = frame.report.issues_for("exact_duplicates")[0]
    assert (issue.severity, issue.count) == (Severity.INFO, 1)


def test_the_same_instant_written_with_different_offsets_is_a_duplicate() -> None:
    same = table(timestamp=["2026-01-05T07:00:00+07:00", "2026-01-05T00:00:00Z"], value=[1.0, 1.0])
    assert len(MetricFrame.from_dataframe(same)) == 1
    different = table(timestamp=["2026-01-05T07:00:00+07:00", "2026-01-05T00:00:00Z"])
    assert "conflicting_duplicates" in rejected(different).codes


def test_conflicting_duplicates_raise_by_default() -> None:
    report = rejected(table(timestamp=["2026-01-05T00:00:00Z"] * 2, value=[1.0, 3.0]))
    issue = report.issues_for("conflicting_duplicates")[0]
    assert issue.count == 1
    assert issue.examples == ('cpu_usage{host="vps-01"} at 2026-01-05T00:00:00+00:00: [1.0, 3.0]',)


@pytest.mark.parametrize(("policy", "expected"), [("first", 5.0), ("last", 7.0), ("mean", 6.0)])
def test_conflicting_duplicate_policies_follow_input_order(policy: str, expected: float) -> None:
    data = table(
        timestamp=["2026-01-05T00:01:00Z", "2026-01-05T00:00:00Z", "2026-01-05T00:01:00Z"],
        metric=["cpu_usage"] * 3,
        host=["vps-01"] * 3,
        value=[5.0, 1.0, 7.0],
    )
    frame = MetricFrame.from_dataframe(data, duplicates=policy)
    assert frame.data["value"].tolist() == [1.0, expected]
    assert frame.report.issues_for("conflicting_duplicates")[0].severity is Severity.WARNING


def test_mean_of_duplicates_that_overflows_is_an_error() -> None:
    data = table(timestamp=["2026-01-05T00:00:00Z"] * 2, value=[1.7e308, 1.6e308])
    report = rejected(data, duplicates="mean")
    assert "aggregation_overflow" in report.codes


def test_the_same_timestamp_in_different_series_is_not_a_duplicate() -> None:
    data = table(timestamp=["2026-01-05T00:00:00Z"] * 2, host=["vps-01", "vps-02"])
    assert len(MetricFrame.from_dataframe(data)) == 2
    by_label = table(timestamp=["2026-01-05T00:00:00Z"] * 2, device=["sda", "sdb"])
    frame = MetricFrame.from_dataframe(by_label, labels=["device"])
    assert len(frame.series()) == 2
    assert frame.report.issues == ()


def test_out_of_order_rows_are_sorted_and_reported() -> None:
    data = table(
        timestamp=["2026-01-05T00:02:00Z", "2026-01-05T00:00:00Z", "2026-01-05T00:01:00Z"],
        metric=["cpu_usage"] * 3,
        host=["vps-01"] * 3,
        value=[3.0, 1.0, 2.0],
    )
    frame = MetricFrame.from_dataframe(data)
    assert frame.data["value"].tolist() == [1.0, 2.0, 3.0]
    assert frame.report.issues_for("out_of_order")[0].count == 2


@pytest.mark.parametrize("units", [["percent", "ratio"], ["percent", None]])
def test_a_series_must_have_one_unit(units: list[str | None]) -> None:
    report = rejected(table(unit=units), on_invalid="drop")
    issue = report.issues_for("conflicting_units")[0]
    assert issue.count == 1
    assert issue.examples[0].startswith('cpu_usage{host="vps-01"}')


def test_one_metric_in_different_units_on_different_hosts_warns() -> None:
    frame = MetricFrame.from_dataframe(table(host=["vps-01", "vps-02"], unit=["percent", "ratio"]))
    issue = frame.report.issues_for("inconsistent_units")[0]
    assert issue.severity is Severity.WARNING
    assert issue.examples == ("cpu_usage: ['percent', 'ratio']",)


def test_empty_input_warns() -> None:
    frame = MetricFrame.from_dataframe(
        pd.DataFrame(columns=["timestamp", "metric", "host", "value"])
    )
    assert len(frame) == 0
    assert frame.report.codes == {"empty_input"}


def test_input_is_not_modified_and_examples_use_positions() -> None:
    data = table(value=[1.0, "N/A"])
    data.index = pd.Index([10, 20])
    before = data.copy(deep=True)
    report = rejected(data)
    pd.testing.assert_frame_equal(data, before)
    assert report.issues_for("invalid_value")[0].examples == ("row 1: 'N/A'",)


def test_messy_csv_fixture(fixtures_dir: Path) -> None:
    raw = pd.read_csv(fixtures_dir / "messy_metrics.csv")
    report = rejected(raw, labels=["interface"])
    assert {issue.code for issue in report.errors} == {
        "invalid_timestamp",
        "invalid_value",
        "conflicting_duplicates",
    }

    frame = MetricFrame.from_dataframe(
        raw, labels=["interface"], on_invalid="drop", duplicates="last"
    )
    counts = {issue.code: issue.count for issue in frame.report.issues}
    assert counts == {
        "invalid_timestamp": 1,
        "invalid_value": 1,
        "missing_value": 1,
        "timezone_converted": 1,
        "out_of_order": 2,
        "exact_duplicates": 1,
        "conflicting_duplicates": 1,
    }
    assert len(frame) == 6
    assert len(frame.series()) == 3
    resolved = frame.data.query("interface == 'ge-0/0/2'")["value"].tolist()
    assert resolved == [16.0]


def test_label_column_names_must_be_strings() -> None:
    data = table()
    data[1] = ["a", "b"]
    report = rejected(data, labels=[1])
    assert "invalid_label_key" in report.codes
