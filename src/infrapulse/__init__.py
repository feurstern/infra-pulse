"""InfraPulse: CPU-first, explainable anomaly detection for infrastructure metrics.

This release contains the data model and preprocessing; detectors come in later milestones. Every
name exported here is public API. Anything else (including submodule internals) may change without
notice.
"""

from importlib.metadata import PackageNotFoundError, version

from infrapulse.data import (
    MetricFrame,
    MetricObservation,
    SeriesKey,
    Severity,
    ValidationIssue,
    ValidationReport,
)
from infrapulse.errors import DataValidationError, InfraPulseError
from infrapulse.preprocessing import RegularFrame, infer_interval, regularize, series_profile

try:
    __version__ = version("infrapulse")
except PackageNotFoundError:  # imported from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "DataValidationError",
    "InfraPulseError",
    "MetricFrame",
    "MetricObservation",
    "RegularFrame",
    "SeriesKey",
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "infer_interval",
    "regularize",
    "series_profile",
]
