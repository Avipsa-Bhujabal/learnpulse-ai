"""Future-inactivity outcome labels kept separate from behavioral features."""

from __future__ import annotations

import csv
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from learnpulse.features import DailyStudentActivity

CourseKey = tuple[str, str]
StudentKey = tuple[str, str, int]


@dataclass(frozen=True)
class FutureInactivityOutcome:
    """Observed activity outcome in a strictly future course-day window."""

    module: str
    presentation: str
    student_id: int
    observation_day: int
    future_window_start_day: int
    future_window_end_day: int
    future_window_days: int
    future_activity_days: int | None
    future_clicks: int | None
    future_inactivity: bool | None
    official_withdrawal_day: int | None
    officially_withdrawn_by_observation_day: bool
    officially_withdrawn_by_window_end: bool
    outcome_eligible: bool
    ineligibility_reason: str | None
    prediction_eligible: bool
    prediction_ineligibility_reason: str | None

    def as_message(self) -> dict[str, str | int | bool | None]:
        """Return a JSON-serializable outcome row."""
        return asdict(self)


def load_course_lengths(path: Path) -> dict[CourseKey, int]:
    """Load each OULAD presentation length without changing the source CSV."""
    lengths: dict[CourseKey, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            lengths[(row["code_module"], row["code_presentation"])] = int(
                row["module_presentation_length"]
            )
    return lengths


def load_withdrawal_days(path: Path) -> dict[StudentKey, int | None]:
    """Load official unregistration days, interpreting ``?`` as missing."""
    withdrawals: dict[StudentKey, int | None] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            raw_day = row["date_unregistration"].strip()
            withdrawals[
                (
                    row["code_module"],
                    row["code_presentation"],
                    int(row["id_student"]),
                )
            ] = None if raw_day in {"", "?"} else int(raw_day)
    return withdrawals


def final_course_day(module_presentation_length: int) -> int:
    """Return the last valid day for an OULAD presentation length.

    OULAD documents the value as a duration in days and defines the starting
    day as 0. Therefore a length of 268 contains Days 0 through 267.
    """
    if module_presentation_length <= 0:
        raise ValueError("module_presentation_length must be greater than zero")
    return module_presentation_length - 1


def calculate_future_inactivity_outcomes(
    daily_summaries: Iterable[DailyStudentActivity],
    observation_days: Iterable[tuple[str, str, int, int]],
    course_lengths: dict[CourseKey, int],
    withdrawal_days: dict[StudentKey, int | None] | None = None,
    future_window_days: int = 14,
) -> list[FutureInactivityOutcome]:
    """Label exact future activity windows for requested observations.

    ``future_inactivity`` means no recorded OULAD VLE activity during days
    ``t + 1`` through ``t + N`` inclusive. Presentation length is treated as
    a duration starting at Day 0, so its final day is length minus one.
    Ineligible windows receive null activity outcomes rather than a shortened
    future window.
    """
    if future_window_days <= 0:
        raise ValueError("future_window_days must be greater than zero")

    grouped: dict[StudentKey, list[DailyStudentActivity]] = defaultdict(list)
    for summary in daily_summaries:
        grouped[(summary.module, summary.presentation, summary.student_id)].append(summary)

    indexed: dict[StudentKey, tuple[list[int], list[int]]] = {}
    for key, summaries in grouped.items():
        ordered = sorted(summaries, key=lambda item: item.day)
        days = [item.day for item in ordered]
        prefix_clicks = [0]
        for item in ordered:
            prefix_clicks.append(prefix_clicks[-1] + item.total_clicks)
        indexed[key] = (days, prefix_clicks)

    withdrawals = withdrawal_days or {}
    outcomes: list[FutureInactivityOutcome] = []
    for module, presentation, student_id, observation_day in observation_days:
        key = (module, presentation, student_id)
        window_start = observation_day + 1
        window_end = observation_day + future_window_days
        withdrawal_day = withdrawals.get(key)
        withdrawn_by_observation = (
            withdrawal_day is not None and withdrawal_day <= observation_day
        )
        withdrawn_by_end = withdrawal_day is not None and withdrawal_day <= window_end
        course_length = course_lengths.get((module, presentation))

        reason: str | None = None
        if course_length is None:
            reason = "missing_course_metadata"
        elif window_end > final_course_day(course_length):
            reason = "future_window_exceeds_course_end"

        prediction_reason = reason
        if prediction_reason is None and withdrawn_by_observation:
            prediction_reason = "already_withdrawn_by_observation_day"

        if reason is not None:
            activity_days: int | None = None
            clicks: int | None = None
            inactive: bool | None = None
        else:
            days, prefix_clicks = indexed.get(key, ([], [0]))
            first = bisect_left(days, window_start)
            after_last = bisect_right(days, window_end)
            activity_days = after_last - first
            clicks = prefix_clicks[after_last] - prefix_clicks[first]
            inactive = activity_days == 0

        outcomes.append(
            FutureInactivityOutcome(
                module=module,
                presentation=presentation,
                student_id=student_id,
                observation_day=observation_day,
                future_window_start_day=window_start,
                future_window_end_day=window_end,
                future_window_days=future_window_days,
                future_activity_days=activity_days,
                future_clicks=clicks,
                future_inactivity=inactive,
                official_withdrawal_day=withdrawal_day,
                officially_withdrawn_by_observation_day=withdrawn_by_observation,
                officially_withdrawn_by_window_end=withdrawn_by_end,
                outcome_eligible=reason is None,
                ineligibility_reason=reason,
                prediction_eligible=prediction_reason is None,
                prediction_ineligibility_reason=prediction_reason,
            )
        )

    return sorted(
        outcomes,
        key=lambda item: (item.module, item.presentation, item.student_id, item.observation_day),
    )
