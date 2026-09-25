"""Grouped cross-validation ablation of absolute and personal-change features."""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from learnpulse.baseline_models import (
    ModelRow,
    build_logistic_pipeline,
    calculate_metrics,
    target_vector,
)
from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)

ABSOLUTE_FEATURE_COLUMNS = [
    "clicks_last_7_days",
    "active_days_last_7_days",
    "resources_last_7_days",
    "activity_types_last_7_days",
    "homepage_clicks_last_7_days",
    "forum_clicks_last_7_days",
    "content_clicks_last_7_days",
    "quiz_clicks_last_7_days",
    "days_since_last_activity",
]

PERSONAL_CHANGE_FEATURE_COLUMNS = [
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

COMBINED_FEATURE_COLUMNS = list(
    dict.fromkeys(ABSOLUTE_FEATURE_COLUMNS + PERSONAL_CHANGE_FEATURE_COLUMNS)
)

MODEL_FEATURE_GROUPS = {
    "absolute": ABSOLUTE_FEATURE_COLUMNS,
    "personal_change": PERSONAL_CHANGE_FEATURE_COLUMNS,
    "combined": COMBINED_FEATURE_COLUMNS,
}

SUMMARY_METRICS = [
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "brier_score",
    "predicted_positive_rows",
]

PAIRED_COMPARISONS = {
    "personal_change_minus_absolute": ("personal_change", "absolute"),
    "combined_minus_absolute": ("combined", "absolute"),
    "combined_minus_personal_change": ("combined", "personal_change"),
}


@dataclass(frozen=True)
class FoldDefinition:
    """One validation fold with mutually exclusive student groups."""

    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_students: frozenset[int]
    validation_students: frozenset[int]


@dataclass(frozen=True)
class AblationResult:
    """All folds, per-fold metrics, OOF predictions, and comparisons."""

    folds: list[FoldDefinition]
    fold_results: list[dict[str, object]]
    predictions: list[dict[str, object]]
    summaries: dict[str, dict[str, dict[str, float]]]
    pooled_results: dict[str, dict[str, object]]
    paired_differences: dict[str, dict[str, dict[str, object]]]
    fitted_pipelines: list[Pipeline]


def validate_feature_groups() -> None:
    """Ensure all ablation features are approved, unique, and leakage-safe."""
    approved = set(MODEL_FEATURE_COLUMNS)
    for name, columns in MODEL_FEATURE_GROUPS.items():
        validate_model_feature_columns(columns)
        if not set(columns) <= approved:
            raise ValueError(f"{name} contains columns outside MODEL_FEATURE_COLUMNS")
        if len(columns) != len(set(columns)):
            raise ValueError(f"{name} contains duplicate features")
    if set(COMBINED_FEATURE_COLUMNS) != approved:
        raise ValueError("Combined features must equal the approved feature schema")


def create_grouped_folds(
    rows: Sequence[ModelRow], folds: int = 5, random_seed: int = 42
) -> list[FoldDefinition]:
    """Create reproducible stratified folds grouped strictly by student."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    targets = target_vector(rows)
    groups = [int(row["student_id"]) for row in rows]
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=random_seed
    )
    definitions: list[FoldDefinition] = []
    placeholder = [[0.0] for _ in rows]
    for fold_number, (train, validation) in enumerate(
        splitter.split(placeholder, targets, groups), start=1
    ):
        train_indices = tuple(int(index) for index in train)
        validation_indices = tuple(int(index) for index in validation)
        train_students = frozenset(groups[index] for index in train_indices)
        validation_students = frozenset(groups[index] for index in validation_indices)
        if train_students & validation_students:
            raise ValueError(f"Student overlap in fold {fold_number}")
        validation_targets = {targets[index] for index in validation_indices}
        if validation_targets != {0, 1}:
            raise ValueError(f"Both classes are required in validation fold {fold_number}")
        if {targets[index] for index in train_indices} != {0, 1}:
            raise ValueError(f"Both classes are required in training fold {fold_number}")
        definitions.append(
            FoldDefinition(
                fold_number,
                train_indices,
                validation_indices,
                train_students,
                validation_students,
            )
        )
    return definitions


def run_ablation_experiment(
    rows: Sequence[ModelRow], folds: int = 5, random_seed: int = 42
) -> AblationResult:
    """Generate fixed-threshold out-of-fold results for all three models."""
    validate_feature_groups()
    definitions = create_grouped_folds(rows, folds, random_seed)
    targets = target_vector(rows)
    probabilities: dict[str, list[float | None]] = {
        model: [None] * len(rows) for model in MODEL_FEATURE_GROUPS
    }
    predictions: dict[str, list[int | None]] = {
        model: [None] * len(rows) for model in MODEL_FEATURE_GROUPS
    }
    fold_results: list[dict[str, object]] = []
    fitted_pipelines: list[Pipeline] = []

    for definition in definitions:
        actual = [targets[index] for index in definition.validation_indices]
        for model_name, columns in MODEL_FEATURE_GROUPS.items():
            pipeline = build_logistic_pipeline(random_seed)
            fitted_pipelines.append(pipeline)
            x_train = _matrix(rows, definition.train_indices, columns)
            y_train = [targets[index] for index in definition.train_indices]
            x_validation = _matrix(rows, definition.validation_indices, columns)
            pipeline.fit(x_train, y_train)
            classifier = pipeline.named_steps["classifier"]
            iterations = max(int(value) for value in classifier.n_iter_)
            converged = iterations < classifier.max_iter
            fold_probabilities = [
                float(value) for value in pipeline.predict_proba(x_validation)[:, 1]
            ]
            fold_predictions = [int(value >= 0.50) for value in fold_probabilities]
            for row_index, probability, prediction in zip(
                definition.validation_indices, fold_probabilities, fold_predictions
            ):
                probabilities[model_name][row_index] = probability
                predictions[model_name][row_index] = prediction
            metrics = calculate_metrics(actual, fold_predictions, fold_probabilities)
            fold_results.append(
                {
                    "fold": definition.fold,
                    "model": model_name,
                    "training_students": len(definition.train_students),
                    "validation_students": len(definition.validation_students),
                    "training_rows": len(definition.train_indices),
                    "validation_rows": len(definition.validation_indices),
                    "validation_positive_labels": sum(actual),
                    "converged": converged,
                    "iterations": iterations,
                    **metrics,
                }
            )

    if any(value is None for values in probabilities.values() for value in values):
        raise RuntimeError("Every row must receive one probability from its validation fold")

    oof_rows: list[dict[str, object]] = []
    fold_by_index = {
        index: definition.fold
        for definition in definitions
        for index in definition.validation_indices
    }
    for index, row in enumerate(rows):
        oof_rows.append(
            {
                "student_id": int(row["student_id"]),
                "observation_day": int(row["observation_day"]),
                "fold": fold_by_index[index],
                "actual_future_inactivity": targets[index],
                "absolute_probability": probabilities["absolute"][index],
                "personal_change_probability": probabilities["personal_change"][index],
                "combined_probability": probabilities["combined"][index],
                "absolute_prediction": predictions["absolute"][index],
                "personal_change_prediction": predictions["personal_change"][index],
                "combined_prediction": predictions["combined"][index],
            }
        )

    summaries = _summarize_folds(fold_results)
    pooled = {
        model: calculate_metrics(
            targets,
            [int(value) for value in predictions[model]],
            [float(value) for value in probabilities[model]],
        )
        for model in MODEL_FEATURE_GROUPS
    }
    paired = _paired_differences(fold_results)
    return AblationResult(
        definitions, fold_results, oof_rows, summaries, pooled, paired, fitted_pipelines
    )


def _matrix(
    rows: Sequence[ModelRow], indices: Sequence[int], columns: Sequence[str]
) -> list[list[float]]:
    return [
        [float("nan") if rows[index].get(column) is None else float(rows[index][column]) for column in columns]
        for index in indices
    ]


def _summarize_folds(
    fold_results: Sequence[dict[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODEL_FEATURE_GROUPS:
        model_rows = [row for row in fold_results if row["model"] == model]
        summaries[model] = {}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in model_rows]
            summaries[model][metric] = {
                "mean": statistics.fmean(values),
                "standard_deviation": statistics.stdev(values),
            }
    return summaries


def _paired_differences(
    fold_results: Sequence[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    indexed = {
        (int(row["fold"]), str(row["model"])): row for row in fold_results
    }
    paired: dict[str, dict[str, dict[str, object]]] = {}
    metrics = ["average_precision", "recall", "precision", "f1", "brier_score"]
    fold_numbers = sorted({int(row["fold"]) for row in fold_results})
    for comparison, (left, right) in PAIRED_COMPARISONS.items():
        paired[comparison] = {}
        for metric in metrics:
            differences = [
                float(indexed[(fold, left)][metric])
                - float(indexed[(fold, right)][metric])
                for fold in fold_numbers
            ]
            paired[comparison][metric] = {
                "fold_differences": differences,
                "mean": statistics.fmean(differences),
                "standard_deviation": statistics.stdev(differences),
            }
    return paired
