"""Leakage-safe personal baselines derived from rolling activity features."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from learnpulse.rolling_features import RollingStudentActivity


@dataclass(frozen=True)
class PersonalBaseline:
    """Current behavior compared with the same student's earlier windows."""

    module: str
    presentation: str
    student_id: int
    day: int
    window_complete: bool
    clicks_last_7_days: int
    active_days_last_7_days: int
    days_since_last_activity: int | None
    historical_windows_available: int
    baseline_ready: bool
    historical_average_clicks: float | None
    historical_standard_deviation_clicks: float | None
    click_difference_from_personal_average: float | None
    click_percentage_change_from_personal_average: float | None
    click_z_score: float | None
    historical_average_active_days: float | None
    active_days_difference_from_personal_average: float | None
    current_inactivity_gap: int | None
    longest_previous_inactivity_gap: int | None

    def as_message(self) -> dict[str, str | int | float | bool | None]:
        """Return a JSON-serializable baseline row at full precision."""
        return asdict(self)


def calculate_personal_baselines(
    rows: Iterable[RollingStudentActivity], minimum_history: int = 7
) -> list[PersonalBaseline]:
    """Compare each rolling row only with earlier rows from the same student.

    Means and sample variance are updated online after emitting each result, so
    neither the current row nor future rows can enter the row's baseline. Only
    earlier rows marked as complete seven-day windows update that history.
    """
    if minimum_history < 0:
        raise ValueError("minimum_history cannot be negative")

    groups: dict[tuple[str, str, int], list[RollingStudentActivity]] = defaultdict(list)
    for row in rows:
        groups[(row.module, row.presentation, row.student_id)].append(row)

    results: list[PersonalBaseline] = []
    for key, group in groups.items():
        count = 0
        mean_clicks = 0.0
        clicks_m2 = 0.0
        mean_active_days = 0.0
        longest_inactivity: int | None = None

        for row in sorted(group, key=lambda item: item.day):
            historical_mean = mean_clicks if count else None
            historical_active_mean = mean_active_days if count else None
            standard_deviation = (
                math.sqrt(clicks_m2 / (count - 1)) if count >= 2 else None
            )
            click_difference = (
                row.clicks_last_7_days - historical_mean
                if historical_mean is not None
                else None
            )
            percentage_change = (
                click_difference / historical_mean * 100
                if click_difference is not None and historical_mean != 0
                else None
            )
            z_score = (
                click_difference / standard_deviation
                if click_difference is not None
                and standard_deviation is not None
                and standard_deviation != 0
                else None
            )
            active_difference = (
                row.active_days_last_7_days - historical_active_mean
                if historical_active_mean is not None
                else None
            )

            results.append(
                PersonalBaseline(
                    module=key[0],
                    presentation=key[1],
                    student_id=key[2],
                    day=row.day,
                    window_complete=row.window_complete,
                    clicks_last_7_days=row.clicks_last_7_days,
                    active_days_last_7_days=row.active_days_last_7_days,
                    days_since_last_activity=row.days_since_last_activity,
                    historical_windows_available=count,
                    baseline_ready=count >= minimum_history,
                    historical_average_clicks=historical_mean,
                    historical_standard_deviation_clicks=standard_deviation,
                    click_difference_from_personal_average=click_difference,
                    click_percentage_change_from_personal_average=percentage_change,
                    click_z_score=z_score,
                    historical_average_active_days=historical_active_mean,
                    active_days_difference_from_personal_average=active_difference,
                    current_inactivity_gap=row.days_since_last_activity,
                    longest_previous_inactivity_gap=longest_inactivity,
                )
            )

            # Update statistical history only after a complete current window
            # has been emitted. Partial windows never enter the baseline.
            if row.window_complete:
                count += 1
                click_delta = row.clicks_last_7_days - mean_clicks
                mean_clicks += click_delta / count
                clicks_m2 += click_delta * (row.clicks_last_7_days - mean_clicks)
                mean_active_days += (
                    row.active_days_last_7_days - mean_active_days
                ) / count
            if row.days_since_last_activity is not None:
                longest_inactivity = (
                    row.days_since_last_activity
                    if longest_inactivity is None
                    else max(longest_inactivity, row.days_since_last_activity)
                )

    return sorted(
        results,
        key=lambda item: (item.module, item.presentation, item.student_id, item.day),
    )
