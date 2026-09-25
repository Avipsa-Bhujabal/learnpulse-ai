"""Build a leakage-auditable modeling table without training a model."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from learnpulse.outcomes import FutureInactivityOutcome
from learnpulse.personal_baseline import PersonalBaseline
from learnpulse.rolling_features import RollingStudentActivity

Row = dict[str, object]
RowKey = tuple[str, str, int, int]
StudentKey = tuple[str, str, int]

IDENTIFIER_COLUMNS = ["module", "presentation", "student_id", "observation_day"]

MODEL_FEATURE_COLUMNS = [
    "clicks_last_7_days",
    "active_days_last_7_days",
    "resources_last_7_days",
    "activity_types_last_7_days",
    "homepage_clicks_last_7_days",
    "forum_clicks_last_7_days",
    "content_clicks_last_7_days",
    "quiz_clicks_last_7_days",
    "days_since_last_activity",
    "historical_windows_available",
    "historical_average_clicks",
    "historical_standard_deviation_clicks",
    "click_difference_from_personal_average",
    "click_percentage_change_from_personal_average",
    "click_z_score",
    "historical_average_active_days",
    "active_days_difference_from_personal_average",
    "current_inactivity_gap",
    "longest_previous_inactivity_gap",
]

TARGET_COLUMN = "future_inactivity"

AUDIT_ONLY_COLUMNS = [
    "future_window_start_day",
    "future_window_end_day",
    "future_activity_days",
    "future_clicks",
    "official_withdrawal_day",
    "officially_withdrawn_by_observation_day",
    "officially_withdrawn_by_window_end",
    "final_result",
    "prediction_eligible",
    "prediction_ineligibility_reason",
    "window_complete",
    "baseline_ready",
    "outcome_eligible",
    "ineligibility_reason",
    "exclusion_reasons",
]

OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN] + AUDIT_ONLY_COLUMNS

FORBIDDEN_MODEL_FEATURE_COLUMNS = set(AUDIT_ONLY_COLUMNS) | {
    TARGET_COLUMN,
    "future_window_days",
}

NONNEGATIVE_COLUMNS = [
    "clicks_last_7_days",
    "active_days_last_7_days",
    "resources_last_7_days",
    "activity_types_last_7_days",
    "homepage_clicks_last_7_days",
    "forum_clicks_last_7_days",
    "content_clicks_last_7_days",
    "quiz_clicks_last_7_days",
    "historical_windows_available",
    "current_inactivity_gap",
    "longest_previous_inactivity_gap",
    "future_activity_days",
    "future_clicks",
]


@dataclass(frozen=True)
class ModelingTableResult:
    """Included and excluded rows plus validation diagnostics."""

    model_ready_rows: list[Row]
    excluded_rows: list[Row]
    duplicate_key_count: int

    @property
    def all_rows(self) -> list[Row]:
        """Return every candidate row in deterministic key order."""
        return sorted(
            self.model_ready_rows + self.excluded_rows,
            key=_row_key,
        )


def load_student_results(path: Path) -> dict[StudentKey, str]:
    """Load final results for auditing only; never for feature construction."""
    results: dict[StudentKey, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            results[
                (
                    row["code_module"],
                    row["code_presentation"],
                    int(row["id_student"]),
                )
            ] = row["final_result"]
    return results


def validate_model_feature_columns(columns: Sequence[str]) -> None:
    """Reject target, future-derived, withdrawal, and audit fields as inputs."""
    forbidden = sorted(set(columns) & FORBIDDEN_MODEL_FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"Forbidden model feature columns: {', '.join(forbidden)}")


def duplicate_key_count(rows: Iterable[Row]) -> int:
    """Count candidate rows beyond the first occurrence of each unique key."""
    counts = Counter(_row_key(row) for row in rows)
    return sum(count - 1 for count in counts.values() if count > 1)


def missing_values_by_feature(rows: Iterable[Row]) -> dict[str, int]:
    """Count meaningful null values for every model feature."""
    materialized = list(rows)
    return {
        column: sum(row.get(column) is None for row in materialized)
        for column in MODEL_FEATURE_COLUMNS
    }


def build_modeling_table(
    rolling_rows: Iterable[RollingStudentActivity],
    baseline_rows: Iterable[PersonalBaseline],
    outcome_rows: Iterable[FutureInactivityOutcome],
    final_results: dict[StudentKey, str],
    observation_days: Sequence[int],
) -> ModelingTableResult:
    """Join one candidate per outcome key and separate eligible/excluded rows."""
    validate_model_feature_columns(MODEL_FEATURE_COLUMNS)
    requested_days = set(observation_days)
    if len(requested_days) != len(observation_days):
        raise ValueError("observation_days must not contain duplicates")

    rolling_index = {
        (row.module, row.presentation, row.student_id, row.day): row
        for row in rolling_rows
        if row.day in requested_days
    }
    baseline_index = {
        (row.module, row.presentation, row.student_id, row.day): row
        for row in baseline_rows
        if row.day in requested_days
    }
    outcomes = list(outcome_rows)
    duplicate_count = duplicate_key_count(
        {
            "module": row.module,
            "presentation": row.presentation,
            "student_id": row.student_id,
            "observation_day": row.observation_day,
        }
        for row in outcomes
    )
    if duplicate_count:
        raise ValueError(f"Duplicate modeling keys detected: {duplicate_count}")

    included: list[Row] = []
    excluded: list[Row] = []
    for outcome in outcomes:
        if outcome.observation_day not in requested_days:
            raise ValueError(f"Unexpected observation day: {outcome.observation_day}")
        key: RowKey = (
            outcome.module,
            outcome.presentation,
            outcome.student_id,
            outcome.observation_day,
        )
        rolling = rolling_index.get(key)
        baseline = baseline_index.get(key)
        reasons: list[str] = []
        if rolling is None:
            reasons.append("missing_rolling_features")
        elif not rolling.window_complete:
            reasons.append("partial_rolling_window")
        if baseline is None:
            reasons.append("missing_personal_baseline")
        elif not baseline.baseline_ready:
            reasons.append("baseline_not_ready")
        if not outcome.outcome_eligible:
            reasons.append(outcome.ineligibility_reason or "outcome_ineligible")
        if not outcome.prediction_eligible:
            reasons.append(
                outcome.prediction_ineligibility_reason or "prediction_ineligible"
            )
        if outcome.future_inactivity is None:
            reasons.append("missing_target")
        if outcome.officially_withdrawn_by_observation_day:
            reasons.append("withdrawn_by_observation_day")

        row: Row = {
            "module": outcome.module,
            "presentation": outcome.presentation,
            "student_id": outcome.student_id,
            "observation_day": outcome.observation_day,
            **_rolling_features(rolling),
            **_baseline_features(baseline),
            TARGET_COLUMN: outcome.future_inactivity,
            "future_window_start_day": outcome.future_window_start_day,
            "future_window_end_day": outcome.future_window_end_day,
            "future_activity_days": outcome.future_activity_days,
            "future_clicks": outcome.future_clicks,
            "official_withdrawal_day": outcome.official_withdrawal_day,
            "officially_withdrawn_by_observation_day": outcome.officially_withdrawn_by_observation_day,
            "officially_withdrawn_by_window_end": outcome.officially_withdrawn_by_window_end,
            "final_result": final_results.get(key[:3]),
            "prediction_eligible": outcome.prediction_eligible,
            "prediction_ineligibility_reason": outcome.prediction_ineligibility_reason,
            "window_complete": rolling.window_complete if rolling else None,
            "baseline_ready": baseline.baseline_ready if baseline else None,
            "outcome_eligible": outcome.outcome_eligible,
            "ineligibility_reason": outcome.ineligibility_reason,
            "exclusion_reasons": "|".join(dict.fromkeys(reasons)) or None,
        }
        _validate_row_values(row)
        (excluded if reasons else included).append(row)

    included.sort(key=_row_key)
    excluded.sort(key=_row_key)
    return ModelingTableResult(included, excluded, duplicate_count)


def write_modeling_csv(path: Path, rows: Iterable[Row]) -> None:
    """Write a modeling or exclusion table without touching source data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _row_key(row: Row) -> RowKey:
    return (
        str(row["module"]),
        str(row["presentation"]),
        int(row["student_id"]),
        int(row["observation_day"]),
    )


def _rolling_features(row: RollingStudentActivity | None) -> Row:
    names = MODEL_FEATURE_COLUMNS[:9]
    return {name: getattr(row, name) if row else None for name in names}


def _baseline_features(row: PersonalBaseline | None) -> Row:
    names = MODEL_FEATURE_COLUMNS[9:]
    return {name: getattr(row, name) if row else None for name in names}


def _validate_row_values(row: Row) -> None:
    if row[TARGET_COLUMN] not in {True, False, None}:
        raise ValueError(f"Invalid target value: {row[TARGET_COLUMN]!r}")
    for column in NONNEGATIVE_COLUMNS:
        value = row.get(column)
        if value is not None and value < 0:  # type: ignore[operator]
            raise ValueError(f"Negative value for {column}: {value}")
