"""Transparent, out-of-fold explanations for personal-change logistic models."""

from __future__ import annotations

import csv
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.baseline_models import ModelRow, build_logistic_pipeline
from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)

EXPLANATION_DAYS = (28, 56)


@dataclass(frozen=True)
class ExplanationResult:
    """Fold explanations, summaries, individual cases, and stability results."""

    coefficient_rows: list[dict[str, object]]
    coefficient_summaries: list[dict[str, object]]
    permutation_rows: list[dict[str, object]]
    permutation_summaries: list[dict[str, object]]
    predictions: list[dict[str, object]]
    individual_explanations: list[dict[str, object]]
    stability: dict[str, object]
    fitted_pipelines: dict[tuple[int, int], Pipeline]


def load_fold_assignments(path: Path) -> dict[int, int]:
    """Load a unique student-to-validation-fold assignment."""
    assignments: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for raw in csv.DictReader(source):
            student = int(raw["student_id"])
            fold = int(raw["fold"])
            if student in assignments:
                raise ValueError(f"Duplicate fold assignment for student {student}")
            assignments[student] = fold
    if not assignments:
        raise ValueError("Fold-assignment file is empty")
    return assignments


def validate_explanation_features() -> None:
    """Ensure explanation inputs are approved, behavioral model features."""
    validate_model_feature_columns(PERSONAL_CHANGE_FEATURE_COLUMNS)
    if not set(PERSONAL_CHANGE_FEATURE_COLUMNS) <= set(MODEL_FEATURE_COLUMNS):
        raise ValueError("Explanation features must be approved model features")
    forbidden = {"student_id", "observation_day", "future_inactivity", "final_result"}
    if forbidden & set(PERSONAL_CHANGE_FEATURE_COLUMNS):
        raise ValueError("Identifiers, targets, and audit fields cannot be explanation features")


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Recover original and separately labelled missing-indicator names."""
    imputer = pipeline.named_steps["imputer"]
    raw_names = [str(value) for value in imputer.get_feature_names_out(PERSONAL_CHANGE_FEATURE_COLUMNS)]
    names: list[str] = []
    for name in raw_names:
        prefix = "missingindicator_"
        names.append(f"missing_indicator__{name[len(prefix):]}" if name.startswith(prefix) else name)
    coefficients = pipeline.named_steps["classifier"].coef_[0]
    if len(names) != len(coefficients):
        raise RuntimeError("Transformed feature names do not match coefficient count")
    return names


def run_explainability_experiment(
    rows: Sequence[ModelRow],
    fold_assignments: dict[int, int],
    observation_days: Sequence[int] = EXPLANATION_DAYS,
    random_seed: int = 42,
    permutation_repeats: int = 30,
) -> ExplanationResult:
    """Fit fold models and explain validation predictions for Days 28 and 56."""
    validate_explanation_features()
    if tuple(sorted(observation_days)) != EXPLANATION_DAYS:
        raise ValueError("Step 9 analyzes only observation Days 28 and 56")
    if permutation_repeats < 1:
        raise ValueError("permutation_repeats must be positive")
    selected = [row for row in rows if int(row["observation_day"]) in EXPLANATION_DAYS]
    selected = [row for row in selected if int(row["student_id"]) in fold_assignments]
    expected_students = set(fold_assignments)
    for day in EXPLANATION_DAYS:
        day_students = {int(row["student_id"]) for row in selected if int(row["observation_day"]) == day}
        if day_students != expected_students:
            raise ValueError(f"Fold-assigned common cohort is incomplete at Day {day}")

    coefficient_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    fitted: dict[tuple[int, int], Pipeline] = {}
    folds = sorted(set(fold_assignments.values()))
    for day in EXPLANATION_DAYS:
        day_rows = [row for row in selected if int(row["observation_day"]) == day]
        for fold in folds:
            train = [row for row in day_rows if fold_assignments[int(row["student_id"])] != fold]
            validation = [row for row in day_rows if fold_assignments[int(row["student_id"])] == fold]
            train_students = {int(row["student_id"]) for row in train}
            validation_students = {int(row["student_id"]) for row in validation}
            if train_students & validation_students:
                raise RuntimeError("A validation student appeared in model training")
            y_train = [int(row["future_inactivity"]) for row in train]
            y_validation = [int(row["future_inactivity"]) for row in validation]
            if set(y_train) != {0, 1} or set(y_validation) != {0, 1}:
                raise ValueError(f"Both classes required for Day {day}, fold {fold}")
            x_train = feature_matrix(train)
            x_validation = feature_matrix(validation)
            pipeline = build_logistic_pipeline(random_seed)
            pipeline.fit(x_train, y_train)
            fitted[(day, fold)] = pipeline

            names = transformed_feature_names(pipeline)
            coefficients = pipeline.named_steps["classifier"].coef_[0]
            for name, coefficient in zip(names, coefficients):
                coefficient_rows.append(
                    {
                        "observation_day": day,
                        "fold": fold,
                        "feature": name,
                        "is_missing_indicator": name.startswith("missing_indicator__"),
                        "standardized_coefficient": float(coefficient),
                    }
                )

            importance = permutation_importance(
                pipeline,
                x_validation,
                y_validation,
                scoring="average_precision",
                n_repeats=permutation_repeats,
                random_state=random_seed,
            )
            for index, feature in enumerate(PERSONAL_CHANGE_FEATURE_COLUMNS):
                permutation_rows.append(
                    {
                        "observation_day": day,
                        "fold": fold,
                        "feature": feature,
                        "scoring": "average_precision",
                        "data_partition": "validation",
                        "importance_mean": float(importance.importances_mean[index]),
                        "importance_standard_deviation": float(importance.importances_std[index]),
                    }
                )

            probabilities = pipeline.predict_proba(x_validation)[:, 1]
            binary = (probabilities >= 0.50).astype(int)
            for row, probability, predicted in zip(validation, probabilities, binary):
                predictions.append(
                    {
                        "student_id": int(row["student_id"]),
                        "observation_day": day,
                        "fold": fold,
                        "actual_future_inactivity": int(row["future_inactivity"]),
                        "predicted_future_inactivity": int(predicted),
                        "model_output": float(probability),
                        "prediction_source": "out_of_fold",
                        "student_was_in_training": int(row["student_id"]) in train_students,
                    }
                )

    coefficient_summaries = summarize_coefficients(coefficient_rows)
    permutation_summaries = summarize_permutation_importance(permutation_rows)
    individual = build_individual_explanations(selected, predictions, fitted)
    stability = compare_stability(coefficient_summaries)
    return ExplanationResult(
        coefficient_rows,
        coefficient_summaries,
        permutation_rows,
        permutation_summaries,
        sorted(predictions, key=lambda row: (int(row["observation_day"]), int(row["student_id"]))),
        individual,
        stability,
        fitted,
    )


def feature_matrix(rows: Sequence[ModelRow]) -> list[list[float]]:
    """Represent personal-change features numerically without filling nulls."""
    return [
        [math.nan if row.get(feature) is None else float(row[feature]) for feature in PERSONAL_CHANGE_FEATURE_COLUMNS]
        for row in rows
    ]


def summarize_coefficients(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize standardized coefficients and direction agreement across folds."""
    grouped: dict[tuple[int, str], list[float]] = {}
    fold_counts: dict[int, int] = {}
    for row in rows:
        day = int(row["observation_day"])
        fold_counts[day] = max(fold_counts.get(day, 0), int(row["fold"]))
        grouped.setdefault((int(row["observation_day"]), str(row["feature"])), []).append(
            float(row["standardized_coefficient"])
        )
    summaries: list[dict[str, object]] = []
    for (day, feature), values in grouped.items():
        signs = {1 if value > 0 else -1 if value < 0 else 0 for value in values}
        mean = statistics.fmean(values)
        summaries.append(
            {
                "observation_day": day,
                "feature": feature,
                "is_missing_indicator": feature.startswith("missing_indicator__"),
                "folds_present": len(values),
                "mean_standardized_coefficient": mean,
                "standard_deviation_across_folds": statistics.stdev(values) if len(values) > 1 else 0.0,
                "sign_agreement": len(values) == fold_counts[day] and len(signs) == 1 and 0 not in signs,
                "direction": "positive" if mean > 0 else "negative" if mean < 0 else "zero",
                "absolute_mean_coefficient": abs(mean),
            }
        )
    for day in EXPLANATION_DAYS:
        day_rows = sorted(
            (row for row in summaries if int(row["observation_day"]) == day),
            key=lambda row: float(row["absolute_mean_coefficient"]), reverse=True,
        )
        for rank, row in enumerate(day_rows, start=1):
            row["absolute_coefficient_rank"] = rank
    return sorted(summaries, key=lambda row: (int(row["observation_day"]), int(row["absolute_coefficient_rank"])))


def summarize_permutation_importance(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize fold-level validation PR-AUC decreases."""
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((int(row["observation_day"]), str(row["feature"])), []).append(
            float(row["importance_mean"])
        )
    output = []
    for (day, feature), values in grouped.items():
        output.append(
            {
                "observation_day": day,
                "feature": feature,
                "scoring": "average_precision",
                "data_partition": "validation",
                "folds_present": len(values),
                "mean_importance": statistics.fmean(values),
                "standard_deviation_across_folds": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return sorted(output, key=lambda row: (int(row["observation_day"]), -float(row["mean_importance"])))


def build_individual_explanations(rows, predictions, pipelines) -> list[dict[str, object]]:
    """Choose representative confusion cases and compute exact logit contributions."""
    row_lookup = {(int(row["student_id"]), int(row["observation_day"])): row for row in rows}
    output: list[dict[str, object]] = []
    cases = {"true_positive": (1, 1), "false_positive": (0, 1), "false_negative": (1, 0), "true_negative": (0, 0)}
    for day in EXPLANATION_DAYS:
        day_predictions = [row for row in predictions if int(row["observation_day"]) == day]
        for case_type, (actual, predicted) in cases.items():
            chosen = select_representative_case(day_predictions, actual, predicted)
            if chosen is None:
                output.append({"observation_day": day, "case_type": case_type, "available": False})
                continue
            model_row = row_lookup[(int(chosen["student_id"]), day)]
            pipeline = pipelines[(day, int(chosen["fold"]))]
            details = explain_one(model_row, pipeline)
            output.append({"case_type": case_type, "available": True, **chosen, **details})
    return output


def select_representative_case(
    predictions: Sequence[dict[str, object]], actual: int, predicted: int
) -> dict[str, object] | None:
    """Select the case nearest its confusion-category median model output."""
    candidates = [
        row for row in predictions
        if int(row["actual_future_inactivity"]) == actual
        and int(row["predicted_future_inactivity"]) == predicted
    ]
    if not candidates:
        return None
    median_probability = statistics.median(float(row["model_output"]) for row in candidates)
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["model_output"]) - median_probability), int(row["student_id"])
        ),
    )


def explain_one(row: ModelRow, pipeline: Pipeline, tolerance: float = 1e-9) -> dict[str, object]:
    """Return exact standardized-feature contributions to one model logit."""
    matrix = feature_matrix([row])
    transformed = pipeline[:-1].transform(matrix)[0]
    classifier = pipeline.named_steps["classifier"]
    coefficients = classifier.coef_[0]
    names = transformed_feature_names(pipeline)
    contributions = [
        {"feature": name, "standardized_value": float(value), "coefficient": float(coefficient), "contribution": float(value * coefficient)}
        for name, value, coefficient in zip(names, transformed, coefficients)
    ]
    intercept = float(classifier.intercept_[0])
    contribution_sum = intercept + sum(float(item["contribution"]) for item in contributions)
    decision = float(pipeline.decision_function(matrix)[0])
    difference = contribution_sum - decision
    if abs(difference) > tolerance:
        raise RuntimeError("Feature contributions do not reproduce the model decision function")
    increasing = sorted((item for item in contributions if float(item["contribution"]) > 0), key=lambda item: -float(item["contribution"]))[:5]
    decreasing = sorted((item for item in contributions if float(item["contribution"]) < 0), key=lambda item: float(item["contribution"]))[:5]
    return {
        "intercept": intercept,
        "decision_function": decision,
        "intercept_plus_contributions": contribution_sum,
        "contribution_sum_difference": difference,
        "contribution_sum_verified": abs(difference) <= tolerance,
        "top_increasing_contributions": increasing,
        "top_decreasing_contributions": decreasing,
    }


def compare_stability(summaries: Sequence[dict[str, object]]) -> dict[str, object]:
    """Compare coefficient direction and absolute rankings within and across days."""
    by_day = {day: [row for row in summaries if int(row["observation_day"]) == day] for day in EXPLANATION_DAYS}
    original = set(PERSONAL_CHANGE_FEATURE_COLUMNS)
    stable = {
        day: [str(row["feature"]) for row in by_day[day] if row["sign_agreement"]]
        for day in EXPLANATION_DAYS
    }
    unstable = {
        day: [str(row["feature"]) for row in by_day[day] if not row["sign_agreement"]]
        for day in EXPLANATION_DAYS
    }
    top = {day: [str(row["feature"]) for row in by_day[day][:5]] for day in EXPLANATION_DAYS}
    ranks = {}
    for day in EXPLANATION_DAYS:
        original_rows = sorted(
            (row for row in by_day[day] if str(row["feature"]) in original),
            key=lambda row: (-float(row["absolute_mean_coefficient"]), str(row["feature"])),
        )
        ranks[day] = {str(row["feature"]): rank for rank, row in enumerate(original_rows, start=1)}
    common = sorted(set(ranks[28]) & set(ranks[56]))
    rank_correlation = _pearson([ranks[28][feature] for feature in common], [ranks[56][feature] for feature in common]) if len(common) > 1 else None
    return {
        "stable_direction_features": stable,
        "unstable_direction_features": unstable,
        "top_five_by_absolute_coefficient": top,
        "top_five_overlap": sorted(set(top[28]) & set(top[56])),
        "spearman_rank_correlation_original_features": rank_correlation,
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Pearson correlation of rank values, equivalent to Spearman correlation."""
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right))
    return None if denominator == 0 else numerator / denominator
