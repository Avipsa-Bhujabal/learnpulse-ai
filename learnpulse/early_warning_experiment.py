"""Observation-day-specific grouped evaluation of personal-change features."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.baseline_models import (
    ModelRow,
    build_logistic_pipeline,
    calculate_metrics,
)
from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)

DAY_SUMMARY_METRICS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
    "predicted_positive_rows",
]


@dataclass(frozen=True)
class DayFold:
    """Training and validation indices for one observation-day fold."""

    observation_day: int
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_students: frozenset[int]
    validation_students: frozenset[int]
    uses_common_assignment: bool


@dataclass(frozen=True)
class EarlyWarningResult:
    """Day-level folds, metrics, OOF predictions, and trajectories."""

    folds: list[DayFold]
    fold_results: list[dict[str, object]]
    predictions: list[dict[str, object]]
    trajectories: list[dict[str, object]]
    summaries: dict[int, dict[str, dict[str, float]]]
    pooled_results: dict[int, dict[str, object]]
    fitted_pipelines: list[Pipeline]


def validate_early_warning_features() -> None:
    """Require the exact approved Step 7 personal-change feature set."""
    validate_model_feature_columns(PERSONAL_CHANGE_FEATURE_COLUMNS)
    if not set(PERSONAL_CHANGE_FEATURE_COLUMNS) <= set(MODEL_FEATURE_COLUMNS):
        raise ValueError("Early-warning features must be approved model features")
    if "observation_day" in PERSONAL_CHANGE_FEATURE_COLUMNS:
        raise ValueError("observation_day must not be a model feature")


def create_day_folds(
    rows: Sequence[ModelRow],
    observation_days: Sequence[int],
    preferred_folds: int = 5,
    random_seed: int = 42,
) -> list[DayFold]:
    """Reuse common group assignments, reducing only class-invalid days."""
    if preferred_folds < 2:
        raise ValueError("folds must be at least two")
    requested = set(observation_days)
    indexed_days = {
        day: [index for index, row in enumerate(rows) if int(row["observation_day"]) == day]
        for day in observation_days
    }
    if any(not indices for indices in indexed_days.values()):
        missing = [day for day, indices in indexed_days.items() if not indices]
        raise ValueError(f"No eligible rows for observation days: {missing}")

    # One common assignment is preferred and reused for every day where it
    # preserves both classes in all training and validation partitions.
    all_indices = [i for i, row in enumerate(rows) if int(row["observation_day"]) in requested]
    common_splitter = StratifiedGroupKFold(
        n_splits=preferred_folds, shuffle=True, random_state=random_seed
    )
    common_assignment: dict[int, int] = {}
    common_targets = [int(rows[i]["future_inactivity"]) for i in all_indices]
    common_groups = [int(rows[i]["student_id"]) for i in all_indices]
    for fold, (_, validation_local) in enumerate(
        common_splitter.split([[0.0]] * len(all_indices), common_targets, common_groups),
        start=1,
    ):
        for local_index in validation_local:
            common_assignment[common_groups[int(local_index)]] = fold

    definitions: list[DayFold] = []
    for day in observation_days:
        day_indices = indexed_days[day]
        common = _folds_from_assignment(
            rows, day, day_indices, common_assignment, preferred_folds, True
        )
        if _folds_have_both_classes(rows, common):
            definitions.extend(common)
            continue

        maximum_possible = min(
            preferred_folds,
            sum(int(rows[index]["future_inactivity"]) for index in day_indices),
        )
        selected: list[DayFold] | None = None
        for fold_count in range(maximum_possible, 1, -1):
            splitter = StratifiedGroupKFold(
                n_splits=fold_count, shuffle=True, random_state=random_seed
            )
            targets = [int(rows[index]["future_inactivity"]) for index in day_indices]
            groups = [int(rows[index]["student_id"]) for index in day_indices]
            candidate: list[DayFold] = []
            for fold, (train_local, validation_local) in enumerate(
                splitter.split([[0.0]] * len(day_indices), targets, groups), start=1
            ):
                train = tuple(day_indices[int(i)] for i in train_local)
                validation = tuple(day_indices[int(i)] for i in validation_local)
                candidate.append(_day_fold(rows, day, fold, train, validation, False))
            if _folds_have_both_classes(rows, candidate):
                selected = candidate
                break
        if selected is None:
            raise ValueError(f"Could not create class-valid grouped folds for Day {day}")
        definitions.extend(selected)
    return definitions


def run_early_warning_experiment(
    rows: Sequence[ModelRow],
    observation_days: Sequence[int],
    folds: int = 5,
    random_seed: int = 42,
) -> EarlyWarningResult:
    """Fit a fresh personal-change pipeline for every day and grouped fold."""
    validate_early_warning_features()
    definitions = create_day_folds(rows, observation_days, folds, random_seed)
    prediction_by_index: dict[int, tuple[int, float, int]] = {}
    fold_results: list[dict[str, object]] = []
    fitted: list[Pipeline] = []

    for definition in definitions:
        pipeline = build_logistic_pipeline(random_seed)
        fitted.append(pipeline)
        x_train = _matrix(rows, definition.train_indices)
        y_train = [int(rows[i]["future_inactivity"]) for i in definition.train_indices]
        x_validation = _matrix(rows, definition.validation_indices)
        actual = [int(rows[i]["future_inactivity"]) for i in definition.validation_indices]
        pipeline.fit(x_train, y_train)
        probabilities = [float(value) for value in pipeline.predict_proba(x_validation)[:, 1]]
        predictions = [int(value >= 0.50) for value in probabilities]
        for index, probability, prediction in zip(
            definition.validation_indices, probabilities, predictions
        ):
            if index in prediction_by_index:
                raise RuntimeError("A row received more than one OOF prediction")
            prediction_by_index[index] = (definition.fold, probability, prediction)
        classifier = pipeline.named_steps["classifier"]
        iterations = max(int(value) for value in classifier.n_iter_)
        fold_results.append(
            {
                "observation_day": definition.observation_day,
                "fold": definition.fold,
                "uses_common_assignment": definition.uses_common_assignment,
                "training_students": len(definition.train_students),
                "validation_students": len(definition.validation_students),
                "training_rows": len(definition.train_indices),
                "validation_rows": len(definition.validation_indices),
                "validation_positive_labels": sum(actual),
                "converged": iterations < classifier.max_iter,
                "iterations": iterations,
                **calculate_metrics(actual, predictions, probabilities),
            }
        )

    eligible_indices = [
        index for index, row in enumerate(rows) if int(row["observation_day"]) in observation_days
    ]
    if set(prediction_by_index) != set(eligible_indices):
        raise RuntimeError("Every eligible row must receive exactly one OOF prediction")
    predictions = [
        {
            "student_id": int(rows[index]["student_id"]),
            "observation_day": int(rows[index]["observation_day"]),
            "fold": prediction_by_index[index][0],
            "actual_future_inactivity": int(rows[index]["future_inactivity"]),
            "personal_change_probability": prediction_by_index[index][1],
            "personal_change_prediction": prediction_by_index[index][2],
            "prediction_source": "out_of_fold",
        }
        for index in eligible_indices
    ]
    summaries = _summaries(fold_results, observation_days)
    pooled = _pooled(predictions, observation_days)
    trajectories = build_trajectories(predictions)
    return EarlyWarningResult(
        definitions, fold_results, predictions, trajectories, summaries, pooled, fitted
    )


def build_trajectories(predictions: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Pivot only OOF probabilities into student warning trajectories."""
    by_student: dict[int, list[dict[str, object]]] = {}
    for row in predictions:
        if row.get("prediction_source") != "out_of_fold":
            raise ValueError("Trajectories require out-of-fold predictions")
        by_student.setdefault(int(row["student_id"]), []).append(row)
    trajectories: list[dict[str, object]] = []
    for student, student_rows in sorted(by_student.items()):
        ordered = sorted(student_rows, key=lambda row: int(row["observation_day"]))
        if len(ordered) < 2:
            continue
        binary = [int(row["personal_change_prediction"]) for row in ordered]
        if all(value == 1 for value in binary):
            pattern = "consistently_high"
        elif all(value == 0 for value in binary):
            pattern = "consistently_low"
        elif binary[0] == 0 and binary[-1] == 1:
            pattern = "low_to_high"
        elif binary[0] == 1 and binary[-1] == 0:
            pattern = "high_to_low"
        else:
            pattern = "mixed"
        trajectory: dict[str, object] = {
            "student_id": student,
            "trajectory_pattern": pattern,
            "observation_count": len(ordered),
        }
        for row in ordered:
            day = int(row["observation_day"])
            trajectory[f"day_{day}_probability"] = row["personal_change_probability"]
            trajectory[f"day_{day}_prediction"] = row["personal_change_prediction"]
        trajectories.append(trajectory)
    return trajectories


def _folds_from_assignment(rows, day, indices, assignment, fold_count, common):
    result = []
    for fold in range(1, fold_count + 1):
        validation = tuple(i for i in indices if assignment.get(int(rows[i]["student_id"])) == fold)
        train = tuple(i for i in indices if i not in set(validation))
        result.append(_day_fold(rows, day, fold, train, validation, common))
    return result


def _day_fold(rows, day, fold, train, validation, common):
    train_students = frozenset(int(rows[i]["student_id"]) for i in train)
    validation_students = frozenset(int(rows[i]["student_id"]) for i in validation)
    if train_students & validation_students:
        raise ValueError(f"Student overlap for Day {day}, fold {fold}")
    return DayFold(day, fold, train, validation, train_students, validation_students, common)


def _folds_have_both_classes(rows, folds):
    return all(
        {int(rows[i]["future_inactivity"]) for i in fold.train_indices} == {0, 1}
        and {int(rows[i]["future_inactivity"]) for i in fold.validation_indices} == {0, 1}
        for fold in folds
    )


def _matrix(rows, indices):
    return [
        [float("nan") if rows[i].get(column) is None else float(rows[i][column]) for column in PERSONAL_CHANGE_FEATURE_COLUMNS]
        for i in indices
    ]


def _summaries(fold_results, days):
    output = {}
    for day in days:
        selected = [row for row in fold_results if row["observation_day"] == day]
        output[day] = {}
        for metric in DAY_SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected]
            output[day][metric] = {
                "mean": statistics.fmean(values),
                "standard_deviation": statistics.stdev(values),
            }
    return output


def _pooled(predictions, days):
    output = {}
    for day in days:
        selected = [row for row in predictions if row["observation_day"] == day]
        output[day] = {
            "eligible_rows": len(selected),
            "unique_students": len({row["student_id"] for row in selected}),
            "future_inactive_rows": sum(int(row["actual_future_inactivity"]) for row in selected),
            "future_inactivity_rate": sum(int(row["actual_future_inactivity"]) for row in selected) / len(selected),
            **calculate_metrics(
                [int(row["actual_future_inactivity"]) for row in selected],
                [int(row["personal_change_prediction"]) for row in selected],
                [float(row["personal_change_probability"]) for row in selected],
            ),
        }
    return output
