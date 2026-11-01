"""Exceptions raised by InfraPulse.

A problem with the *data* raises :class:`DataValidationError`. When it comes from validating a table
it carries the full :class:`~infrapulse.data.validation.ValidationReport`, so every problem can be
fixed in one pass instead of one error at a time. A problem with an *argument* (a negative interval,
an unknown policy name) raises the built-in ``ValueError`` or ``TypeError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infrapulse.data.validation import ValidationReport


class InfraPulseError(Exception):
    """Base class for errors raised by InfraPulse."""


class DataValidationError(InfraPulseError, ValueError):
    """Input data violates the documented data model (see ``docs/data-model.md``).

    ``report`` lists every problem that was found; it is ``None`` when a single record or value was
    rejected (for example by :class:`~infrapulse.data.model.MetricObservation`).
    """

    def __init__(self, message: str, *, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report
