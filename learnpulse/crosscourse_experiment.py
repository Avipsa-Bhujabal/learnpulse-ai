"""Cross-course generalization experiments for future OULAD VLE inactivity."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from learnpulse.ablation_experiment import (
    ABSOLUTE_FEATURE_COLUMNS,
    COMBINED_FEATURE_COLUMNS,
    PERSONAL_CHANGE_FEATURE_COLUMNS,
)
from learnpulse.baseline_models import build_logistic_pipeline
from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)

MODEL_FEATURE_GROUPS = {
    "A": list(ABSOLUTE_FEATURE_COLUMNS),
    "B": list(PERSONAL_CHANGE_FEATURE_COLUMNS),
    "C": list(COMBINED_FEATURE_COLUMNS),
}
MODEL_NAMES = {"A": "absolute", "B": "personal_change", "C": "combined"}
FORBIDDEN_COLUMNS = {
    "module", "presentation", "student_id", "observation_day", "future_inactivity",
    "future_window_start_day", "future_window_end_day", "future_activity_days",
    "future_clicks", "official_withdrawal_day",
    "officially_withdrawn_by_observation_day", "officially_withdrawn_by_window_end",
    "final_result", "outcome_eligible", "prediction_eligible", "exclusion_reason",
    "source_partition", "dataset_schema_version",
}
PAIRINGS = {"B_minus_A": ("B", "A"), "C_minus_A": ("C", "A"), "C_minus_B": ("C", "B")}
PAIRED_METRICS = ["average_precision", "pr_auc_lift", "recall", "precision", "f1", "brier_score"]
SUMMARY_METRICS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
    "roc_auc", "average_precision", "no_skill_pr_auc", "pr_auc_lift", "brier_score",
    "predicted_positive_rows", "true_negatives", "false_positives",
    "false_negatives", "true_positives",
]
PREDICTION_COLUMNS = [
    "evaluation_design", "overlap_policy", "held_out_module",
    "held_out_presentation", "observation_day", "model_name", "module",
    "presentation", "student_id", "actual_label", "predicted_label",
    "model_output", "fold_or_holdout_id",
]


@dataclass(frozen=True)
class SplitDefinition:
    """A reusable train/test definition shared by Models A, B, and C."""

    evaluation_design: str
    overlap_policy: str
    holdout_id: str
    held_out_module: str | None
    held_out_presentation: str | None
    observation_day: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    overlapping_students_before_removal: int
    training_students_removed: int
    training_rows_removed: int


@dataclass(frozen=True)
class CrossCourseResult:
    """Complete experiment results before atomic serialization."""

    fold_results: list[dict[str, object]]
    predictions: list[dict[str, object]]
    summary_rows: list[dict[str, object]]
    feature_results: list[dict[str, object]]
    report: dict[str, object]
    fitted_models: int


def validate_feature_schemas(groups: dict[str, Sequence[str]] = MODEL_FEATURE_GROUPS) -> None:
    """Require the exact established ablation schemas and reject leakage fields."""
    expected = {
        "A": list(ABSOLUTE_FEATURE_COLUMNS),
        "B": list(PERSONAL_CHANGE_FEATURE_COLUMNS),
        "C": list(COMBINED_FEATURE_COLUMNS),
    }
    if {name: list(values) for name, values in groups.items()} != expected:
        raise ValueError("Cross-course feature groups must match the Step 7 schemas exactly")
    for name, columns in groups.items():
        validate_model_feature_columns(columns)
        forbidden = sorted(set(columns) & FORBIDDEN_COLUMNS)
        if forbidden:
            raise ValueError(f"Model {name} contains forbidden columns: {forbidden}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Model {name} contains duplicate features")
        if not set(columns).issubset(MODEL_FEATURE_COLUMNS):
            raise ValueError(f"Model {name} contains unapproved features")


def load_parquet_rows(path: Path, observation_days: Sequence[int]) -> list[dict[str, object]]:
    """Read only identifiers, target, and approved features through DuckDB."""
    if not path.is_file():
        raise FileNotFoundError(f"Modeling Parquet does not exist: {path}")
    days = tuple(sorted(set(observation_days)))
    selected = ["module", "presentation", "student_id", "observation_day", "future_inactivity", *MODEL_FEATURE_COLUMNS]
    with duckdb.connect() as connection:
        available = {row[0] for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path.resolve())]).fetchall()}
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(f"Modeling Parquet is missing columns: {missing}")
        query = f"SELECT {','.join(selected)} FROM read_parquet(?) WHERE observation_day IN (SELECT * FROM unnest(?)) ORDER BY module,presentation,student_id,observation_day"
        values = connection.execute(query, [str(path.resolve()), list(days)]).fetchall()
    rows = [dict(zip(selected, values_)) for values_ in values]
    if not rows:
        raise ValueError("No modeling rows matched the requested observation days")
    return rows


def create_grouped_reference_splits(
    rows: Sequence[dict[str, object]], day: int, folds: int, random_seed: int
) -> tuple[list[SplitDefinition], str | None]:
    """Find the largest feasible stratified student-grouped fold count."""
    indices = [i for i, row in enumerate(rows) if int(row["observation_day"]) == day]
    for count in range(min(folds, len({int(rows[i]["student_id"]) for i in indices})), 1, -1):
        targets = [int(rows[i]["future_inactivity"]) for i in indices]
        groups = [int(rows[i]["student_id"]) for i in indices]
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=random_seed)
        definitions: list[SplitDefinition] = []
        valid = True
        for fold, (train_local, test_local) in enumerate(splitter.split(np.zeros(len(indices)), targets, groups), 1):
            train = tuple(indices[int(i)] for i in train_local)
            test = tuple(indices[int(i)] for i in test_local)
            if _classes(rows, train) != {0, 1} or _classes(rows, test) != {0, 1}:
                valid = False; break
            definitions.append(SplitDefinition("grouped", "student-disjoint", f"fold_{fold}", None, None, day, train, test, 0, 0, 0))
        if valid:
            reason = None if count == folds else f"Reduced from {folds} to {count}: higher fold counts lacked both classes"
            return definitions, reason
    raise ValueError(f"Could not form class-valid grouped folds for Day {day}")


def create_course_holdout_splits(
    rows: Sequence[dict[str, object]], day: int, design: str,
    overlap_policies: Sequence[str], only_module: str | None = None,
    only_presentation: str | None = None,
) -> list[SplitDefinition]:
    """Create presentation or module holdouts, including strict student removal."""
    if design not in {"presentation", "module"}:
        raise ValueError("Course holdout design must be presentation or module")
    day_indices = [i for i, row in enumerate(rows) if int(row["observation_day"]) == day]
    units = sorted({
        (str(rows[i]["module"]), str(rows[i]["presentation"]) if design == "presentation" else None)
        for i in day_indices
        if (only_module is None or str(rows[i]["module"]) == only_module)
        and (only_presentation is None or str(rows[i]["presentation"]) == only_presentation)
    })
    definitions = []
    for module, presentation in units:
        test = tuple(i for i in day_indices if str(rows[i]["module"]) == module and (design == "module" or str(rows[i]["presentation"]) == presentation))
        base_train = tuple(i for i in day_indices if i not in set(test))
        test_students = {int(rows[i]["student_id"]) for i in test}
        train_students = {int(rows[i]["student_id"]) for i in base_train}
        overlap = test_students & train_students
        for policy in overlap_policies:
            train = base_train if policy == "course-only" else tuple(i for i in base_train if int(rows[i]["student_id"]) not in test_students)
            removed_rows = len(base_train) - len(train)
            removed_students = (
                len({int(rows[i]["student_id"]) for i in base_train if int(rows[i]["student_id"]) in test_students})
                if policy == "student-disjoint" else 0
            )
            holdout = module if design == "module" else f"{module}_{presentation}"
            definitions.append(SplitDefinition(design, policy, holdout, module, presentation, day, train, test, len(overlap), removed_students, removed_rows))
    return definitions


def calculate_metrics_safe(actual: Sequence[int], predicted: Sequence[int], probabilities: Sequence[float]) -> dict[str, object]:
    """Calculate binary metrics without failing on a one-class test holdout."""
    if not actual or len(actual) != len(predicted) or len(actual) != len(probabilities):
        raise ValueError("Metric inputs must be non-empty and have equal lengths")
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    prevalence = sum(actual) / len(actual)
    both = set(actual) == {0, 1}
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "test_rows": len(actual), "positive_rows": int(sum(actual)),
        "negative_rows": len(actual) - int(sum(actual)), "target_prevalence": prevalence,
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)) if both else None,
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)) if sum(actual) else None,
        "specificity": float(specificity) if specificity is not None else None,
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, probabilities)) if both else None,
        "average_precision": float(average_precision_score(actual, probabilities)) if sum(actual) else None,
        "no_skill_pr_auc": prevalence,
        "pr_auc_lift": float(average_precision_score(actual, probabilities) / prevalence) if prevalence > 0 else None,
        "brier_score": float(brier_score_loss(actual, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "predicted_positive_rows": int(sum(predicted)),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
    }


def run_crosscourse_experiment(
    rows: Sequence[dict[str, object]], observation_days: Sequence[int] = (14, 28, 42, 56),
    models: Sequence[str] = ("A", "B", "C"),
    evaluation_designs: Sequence[str] = ("grouped", "presentation", "module"),
    overlap_policies: Sequence[str] = ("course-only", "student-disjoint"),
    folds: int = 5, threshold: float = 0.5, random_seed: int = 42,
    only_module: str | None = None, only_presentation: str | None = None,
) -> CrossCourseResult:
    """Fit day-specific logistic models under all requested generalization designs."""
    validate_feature_schemas()
    _validate_experiment_args(observation_days, models, evaluation_designs, overlap_policies, folds, threshold)
    started = time.perf_counter()
    fold_results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    feature_results: list[dict[str, object]] = []
    fold_reductions: dict[str, str] = {}
    split_count = 0
    for day in observation_days:
        definitions: list[SplitDefinition] = []
        if "grouped" in evaluation_designs:
            grouped, reason = create_grouped_reference_splits(rows, day, folds, random_seed)
            definitions.extend(grouped)
            if reason: fold_reductions[str(day)] = reason
        for design in ("presentation", "module"):
            if design in evaluation_designs:
                definitions.extend(create_course_holdout_splits(rows, day, design, overlap_policies, only_module, only_presentation))
        for split in definitions:
            split_count += 1
            if _classes(rows, split.train_indices) != {0, 1}:
                raise ValueError(f"Training partition {split.holdout_id} Day {day} lacks both classes")
            if split.evaluation_design != "grouped":
                feature_results.extend(dataset_shift_rows(rows, split))
            actual = [int(rows[i]["future_inactivity"]) for i in split.test_indices]
            test_students = len({int(rows[i]["student_id"]) for i in split.test_indices})
            train_students = len({int(rows[i]["student_id"]) for i in split.train_indices})
            warning = None if set(actual) == {0, 1} else "Test holdout contains only one target class; class-dependent metrics are null"
            for model in models:
                columns = MODEL_FEATURE_GROUPS[model]
                pipeline = build_logistic_pipeline(random_seed)
                pipeline.fit(_matrix(rows, split.train_indices, columns), [int(rows[i]["future_inactivity"]) for i in split.train_indices])
                probabilities = [float(value) for value in pipeline.predict_proba(_matrix(rows, split.test_indices, columns))[:, 1]]
                predicted = [int(value >= threshold) for value in probabilities]
                classifier = pipeline.named_steps["classifier"]
                iterations = max(int(value) for value in classifier.n_iter_)
                metrics = calculate_metrics_safe(actual, predicted, probabilities)
                fold_results.append({
                    "evaluation_design": split.evaluation_design, "overlap_policy": split.overlap_policy,
                    "held_out_module": split.held_out_module, "held_out_presentation": split.held_out_presentation,
                    "fold_or_holdout_id": split.holdout_id, "observation_day": day,
                    "model": model, "model_name": MODEL_NAMES[model],
                    "training_rows": len(split.train_indices), "test_rows": len(split.test_indices),
                    "training_students": train_students, "test_students": test_students,
                    "overlapping_students_before_removal": split.overlapping_students_before_removal,
                    "training_students_removed": split.training_students_removed,
                    "training_rows_removed": split.training_rows_removed,
                    "converged": iterations < classifier.max_iter, "iterations": iterations,
                    "warning": warning, **metrics,
                })
                for index, probability, prediction in zip(split.test_indices, probabilities, predicted):
                    row = rows[index]
                    predictions.append({
                        "evaluation_design": split.evaluation_design, "overlap_policy": split.overlap_policy,
                        "held_out_module": split.held_out_module, "held_out_presentation": split.held_out_presentation,
                        "observation_day": day, "model_name": model, "module": row["module"],
                        "presentation": row["presentation"], "student_id": int(row["student_id"]),
                        "actual_label": int(row["future_inactivity"]), "predicted_label": prediction,
                        "model_output": probability, "fold_or_holdout_id": split.holdout_id,
                    })
    summary_rows = summarize_results(fold_results, predictions)
    paired = paired_model_differences(fold_results)
    worst = worst_group_results(fold_results)
    report = {
        "runtime_seconds": time.perf_counter() - started,
        "fitted_models": len(fold_results), "split_definitions": split_count,
        "fold_reductions": fold_reductions, "paired_model_comparisons": paired,
        "worst_group_results": worst,
        "overlap_removal_summary": overlap_removal_summary(fold_results),
        "dataset_shift_summary": dataset_shift_summary(feature_results, fold_results),
    }
    return CrossCourseResult(fold_results, predictions, summary_rows, feature_results, report, len(fold_results))


def _matrix(rows: Sequence[dict[str, object]], indices: Sequence[int], columns: Sequence[str]) -> list[list[float]]:
    return [[math.nan if rows[i].get(column) is None else float(rows[i][column]) for column in columns] for i in indices]


def _classes(rows: Sequence[dict[str, object]], indices: Sequence[int]) -> set[int]:
    return {int(rows[i]["future_inactivity"]) for i in indices}


def dataset_shift_rows(rows: Sequence[dict[str, object]], split: SplitDefinition) -> list[dict[str, object]]:
    """Describe target, missingness, medians, IQRs, and standardized shifts."""
    train_target = [int(rows[i]["future_inactivity"]) for i in split.train_indices]
    test_target = [int(rows[i]["future_inactivity"]) for i in split.test_indices]
    output = []
    for feature in MODEL_FEATURE_COLUMNS:
        train_raw = [rows[i].get(feature) for i in split.train_indices]
        test_raw = [rows[i].get(feature) for i in split.test_indices]
        train = [float(value) for value in train_raw if value is not None]
        test = [float(value) for value in test_raw if value is not None]
        train_median, train_iqr = _median_iqr(train)
        test_median, test_iqr = _median_iqr(test)
        standardized = (
            (test_median - train_median) / train_iqr
            if train_median is not None and test_median is not None and train_iqr not in {None, 0.0}
            else None
        )
        output.append({
            "evaluation_design": split.evaluation_design, "overlap_policy": split.overlap_policy,
            "held_out_module": split.held_out_module, "held_out_presentation": split.held_out_presentation,
            "fold_or_holdout_id": split.holdout_id, "observation_day": split.observation_day,
            "feature": feature,
            "training_target_prevalence": sum(train_target) / len(train_target),
            "test_target_prevalence": sum(test_target) / len(test_target),
            "target_prevalence_difference": sum(test_target) / len(test_target) - sum(train_target) / len(train_target),
            "training_missing_rate": sum(value is None for value in train_raw) / len(train_raw),
            "test_missing_rate": sum(value is None for value in test_raw) / len(test_raw),
            "training_median": train_median, "test_median": test_median,
            "training_iqr": train_iqr, "test_iqr": test_iqr,
            "standardized_median_difference": standardized,
            "zero_training_iqr": train_iqr == 0.0,
        })
    return output


def _median_iqr(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=float)
    return float(np.median(array)), float(np.percentile(array, 75) - np.percentile(array, 25))


def summarize_results(
    fold_results: Sequence[dict[str, object]], predictions: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    """Create explicitly macro and pooled summaries for every comparable group."""
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in fold_results:
        key = (row["evaluation_design"], row["overlap_policy"], row["observation_day"], row["model"])
        grouped[key].append(row)
    output: list[dict[str, object]] = []
    for key, selected in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        design, policy, day, model = key
        macro: dict[str, object] = {
            "evaluation_design": design, "overlap_policy": policy,
            "observation_day": day, "model": model, "summary_type": "macro",
            "holdout_count": len(selected),
        }
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            macro[f"{metric}_mean"] = statistics.fmean(values) if values else None
            macro[f"{metric}_standard_deviation"] = statistics.stdev(values) if len(values) >= 2 else None
        output.append(macro)
        prediction_selected = [
            row for row in predictions
            if row["evaluation_design"] == design and row["overlap_policy"] == policy
            and row["observation_day"] == day and row["model_name"] == model
        ]
        metrics = calculate_metrics_safe(
            [int(row["actual_label"]) for row in prediction_selected],
            [int(row["predicted_label"]) for row in prediction_selected],
            [float(row["model_output"]) for row in prediction_selected],
        )
        output.append({
            "evaluation_design": design, "overlap_policy": policy,
            "observation_day": day, "model": model, "summary_type": "pooled",
            "holdout_count": len(selected), **metrics,
        })
    return output


def worst_group_results(fold_results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Identify worst holdout metric values rather than reporting averages alone."""
    output = []
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in fold_results:
        if row["evaluation_design"] == "grouped":
            continue
        groups[(row["evaluation_design"], row["overlap_policy"], row["observation_day"], row["model"])].append(row)
    specs = {
        "lowest_pr_auc": ("average_precision", min), "lowest_recall": ("recall", min),
        "lowest_precision": ("precision", min), "highest_false_negative_rate": ("false_negative_rate", max),
        "highest_false_positive_rate": ("false_positive_rate", max),
    }
    for key, selected in groups.items():
        for label, (metric, operation) in specs.items():
            valid = [row for row in selected if row.get(metric) is not None]
            if not valid: continue
            chosen = operation(valid, key=lambda row: float(row[metric]))
            output.append({"evaluation_design": key[0], "overlap_policy": key[1], "observation_day": key[2], "model": key[3], "worst_measure": label, "value": chosen[metric], "holdout": chosen["fold_or_holdout_id"]})
    return output


def paired_model_differences(fold_results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize paired holdout differences without significance claims."""
    indexed = {(row["evaluation_design"], row["overlap_policy"], row["observation_day"], row["fold_or_holdout_id"], row["model"]): row for row in fold_results}
    contexts = sorted({key[:3] for key in indexed}, key=lambda value: tuple(str(v) for v in value))
    output = []
    for context in contexts:
        holdouts = sorted({key[3] for key in indexed if key[:3] == context})
        for comparison, (left, right) in PAIRINGS.items():
            for metric in PAIRED_METRICS:
                differences = []
                for holdout in holdouts:
                    a = indexed.get((*context, holdout, left), {}).get(metric)
                    b = indexed.get((*context, holdout, right), {}).get(metric)
                    if a is not None and b is not None:
                        differences.append(float(a) - float(b))
                output.append({
                    "evaluation_design": context[0], "overlap_policy": context[1],
                    "observation_day": context[2], "comparison": comparison, "metric": metric,
                    "valid_holdouts": len(differences),
                    "mean_difference": statistics.fmean(differences) if differences else None,
                    "standard_deviation": statistics.stdev(differences) if len(differences) >= 2 else None,
                    "median_difference": statistics.median(differences) if differences else None,
                    "left_favored": sum(value > 0 for value in differences),
                    "right_favored": sum(value < 0 for value in differences),
                    "ties": sum(value == 0 for value in differences),
                })
    return output


def overlap_removal_summary(fold_results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize unique split overlap diagnostics without triple-counting models."""
    unique = {}
    for row in fold_results:
        key = (row["evaluation_design"], row["overlap_policy"], row["observation_day"], row["fold_or_holdout_id"])
        unique[key] = row
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for key, row in unique.items(): grouped[key[:3]].append(row)
    return [{
        "evaluation_design": key[0], "overlap_policy": key[1], "observation_day": key[2],
        "holdouts": len(values),
        "overlapping_student_ids_before_removal": sum(int(row["overlapping_students_before_removal"]) for row in values),
        "training_students_removed": sum(int(row["training_students_removed"]) for row in values),
        "training_rows_removed": sum(int(row["training_rows_removed"]) for row in values),
    } for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0]))]


def dataset_shift_summary(
    feature_results: Sequence[dict[str, object]], fold_results: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    """Summarize shift magnitude and its descriptive association with Model B PR-AUC."""
    by_holdout: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in feature_results:
        by_holdout[(row["evaluation_design"], row["overlap_policy"], row["observation_day"], row["fold_or_holdout_id"])].append(row)
    metrics = {(row["evaluation_design"], row["overlap_policy"], row["observation_day"], row["fold_or_holdout_id"]): row for row in fold_results if row["model"] == "B"}
    grouped: dict[tuple[object, ...], list[dict[str, float]]] = defaultdict(list)
    for key, values in by_holdout.items():
        standardized = [abs(float(row["standardized_median_difference"])) for row in values if row["standardized_median_difference"] is not None]
        metric = metrics.get(key)
        grouped[key[:3]].append({
            "mean_absolute_standardized_median_shift": statistics.fmean(standardized) if standardized else math.nan,
            "absolute_prevalence_difference": abs(float(values[0]["target_prevalence_difference"])),
            "positive_rows": float(metric["positive_rows"]) if metric else math.nan,
            "average_precision": float(metric["average_precision"]) if metric and metric["average_precision"] is not None else math.nan,
            "zero_iqr_features": float(sum(bool(row["zero_training_iqr"]) for row in values)),
        })
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        output.append({
            "evaluation_design": key[0], "overlap_policy": key[1], "observation_day": key[2],
            "holdouts": len(values),
            "mean_absolute_standardized_median_shift": _finite_mean(values, "mean_absolute_standardized_median_shift"),
            "mean_absolute_prevalence_difference": _finite_mean(values, "absolute_prevalence_difference"),
            "zero_training_iqr_feature_instances": int(sum(row["zero_iqr_features"] for row in values)),
            "correlation_pr_auc_with_feature_shift": _correlation(values, "average_precision", "mean_absolute_standardized_median_shift"),
            "correlation_pr_auc_with_prevalence_shift": _correlation(values, "average_precision", "absolute_prevalence_difference"),
            "correlation_pr_auc_with_positive_count": _correlation(values, "average_precision", "positive_rows"),
            "interpretation": "Descriptive association only; correlations do not establish causation.",
        })
    return output


def _finite_mean(rows: Sequence[dict[str, float]], key: str) -> float | None:
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return statistics.fmean(values) if values else None


def _correlation(rows: Sequence[dict[str, float]], left: str, right: str) -> float | None:
    pairs = [(row[left], row[right]) for row in rows if math.isfinite(row[left]) and math.isfinite(row[right])]
    if len(pairs) < 2 or len({a for a, _ in pairs}) < 2 or len({b for _, b in pairs}) < 2: return None
    return float(np.corrcoef([a for a, _ in pairs], [b for _, b in pairs])[0, 1])


def save_crosscourse_outputs(
    result: CrossCourseResult, source_path: Path, results_path: Path,
    fold_results_path: Path, predictions_path: Path, summary_path: Path,
    feature_results_path: Path, configuration: dict[str, object],
) -> dict[str, object]:
    """Stage every artifact, then publish the set with rollback protection."""
    targets = [results_path, fold_results_path, predictions_path, summary_path, feature_results_path]
    if len({path.resolve() for path in targets}) != len(targets):
        raise ValueError("Experiment output paths must be distinct")
    source_before = _signature(source_path)
    parent = results_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".crosscourse_build_", dir=parent))
    try:
        staged = [temporary_root / path.name for path in targets]
        _write_csv(staged[1], result.fold_results)
        _write_predictions_parquet(staged[2], result.predictions)
        _write_csv(staged[3], result.summary_rows)
        _write_csv(staged[4], result.feature_results)
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration, "runtime": result.report,
            "summaries": result.summary_rows,
            "package_versions": _package_versions(),
            "feature_groups": MODEL_FEATURE_GROUPS,
            "leakage_validation_passed": True,
            "source_integrity": {"before": source_before, "unchanged": source_before == _signature(source_path)},
            "artifact_row_counts": {
                "fold_results": len(result.fold_results), "predictions": len(result.predictions),
                "summary": len(result.summary_rows), "feature_results": len(result.feature_results),
            },
        }
        staged[0].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _publish(list(zip(staged, targets)))
        return report
    finally:
        if temporary_root.exists(): shutil.rmtree(temporary_root)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        if columns: writer.writeheader(); writer.writerows(rows)


def _write_predictions_parquet(path: Path, rows: Sequence[dict[str, object]]) -> None:
    csv_path = path.with_suffix(".staging.csv")
    _write_csv(csv_path, rows)
    target = path.resolve().as_posix().replace("'", "''")
    with duckdb.connect() as connection:
        connection.execute(f"COPY (SELECT * FROM read_csv_auto(?, header=true)) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)", [str(csv_path.resolve())])
    csv_path.unlink()


def _publish(pairs: Sequence[tuple[Path, Path]]) -> None:
    backups, published = [], []
    try:
        for _, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.previous")
                if backup.exists(): backup.unlink()
                os.replace(target, backup); backups.append((backup, target))
        for source, target in pairs:
            os.replace(source, target); published.append(target)
    except Exception:
        for target in published:
            if target.exists(): target.unlink()
        for backup, target in backups:
            if backup.exists(): os.replace(backup, target)
        raise
    for backup, _ in backups:
        if backup.exists(): backup.unlink()


def _signature(path: Path) -> dict[str, object]:
    stat = path.stat(); return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _package_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "duckdb": duckdb.__version__, "numpy": np.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")}


def _validate_experiment_args(days, models, designs, policies, folds, threshold) -> None:
    if not days or len(days) != len(set(days)): raise ValueError("Observation days must be unique and non-empty")
    if not models or not set(models) <= set(MODEL_FEATURE_GROUPS): raise ValueError("Models must be selected from A, B, C")
    if not designs or not set(designs) <= {"grouped", "presentation", "module"}: raise ValueError("Invalid evaluation design")
    if not policies or not set(policies) <= {"course-only", "student-disjoint"}: raise ValueError("Invalid overlap policy")
    if folds < 2: raise ValueError("folds must be at least two")
    if not 0 <= threshold <= 1: raise ValueError("threshold must be between zero and one")
