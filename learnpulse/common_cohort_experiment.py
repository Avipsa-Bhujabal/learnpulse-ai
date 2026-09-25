"""Common-cohort sensitivity analysis for early-warning timing."""

from __future__ import annotations

import random
import statistics
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.baseline_models import (
    ModelRow,
    build_logistic_pipeline,
    calculate_metrics,
)
from learnpulse.early_warning_experiment import DAY_SUMMARY_METRICS


@dataclass(frozen=True)
class CommonCohort:
    """Students having exactly one model-ready row on every requested day."""

    rows: list[ModelRow]
    students_before: int
    students_after: int
    students_removed: int


@dataclass(frozen=True)
class CommonFold:
    """One reusable student validation fold."""

    fold: int
    train_students: frozenset[int]
    validation_students: frozenset[int]


@dataclass(frozen=True)
class CommonCohortResult:
    """Cohort, shared folds, OOF metrics, uncertainty, and comparisons."""

    cohort: CommonCohort
    folds: list[CommonFold]
    fold_results: list[dict[str, object]]
    predictions: list[dict[str, object]]
    pooled_results: dict[int, dict[str, object]]
    summaries: dict[int, dict[str, dict[str, float]]]
    confidence_intervals: dict[int, dict[str, dict[str, float | int | None]]]
    paired_day_differences: dict[str, dict[str, float]]
    fitted_pipelines: list[Pipeline]


def create_common_cohort(
    rows: Sequence[ModelRow], observation_days: Sequence[int]
) -> CommonCohort:
    """Keep only students with exactly one row at every requested day."""
    required = set(observation_days)
    by_student: dict[int, list[ModelRow]] = {}
    for row in rows:
        by_student.setdefault(int(row["student_id"]), []).append(row)
    included: list[ModelRow] = []
    for student_rows in by_student.values():
        relevant = [row for row in student_rows if int(row["observation_day"]) in required]
        days = [int(row["observation_day"]) for row in relevant]
        if set(days) == required and len(days) == len(required):
            included.extend(relevant)
    included.sort(key=lambda row: (int(row["student_id"]), int(row["observation_day"])))
    after = len({int(row["student_id"]) for row in included})
    return CommonCohort(included, len(by_student), after, len(by_student) - after)


def create_shared_student_folds(
    cohort: CommonCohort,
    observation_days: Sequence[int],
    max_folds: int = 4,
    random_seed: int = 42,
) -> list[CommonFold]:
    """Find the largest deterministic multi-day class-valid student partition.

    Candidate assignments are seeded random balanced partitions. The selected
    assignment minimizes squared deviation from equal positive counts across
    every observation day, then fold-size imbalance.
    """
    students = sorted({int(row["student_id"]) for row in cohort.rows})
    labels = {
        (int(row["student_id"]), int(row["observation_day"])): int(row["future_inactivity"])
        for row in cohort.rows
    }
    for fold_count in range(min(max_folds, len(students)), 1, -1):
        rng = random.Random(random_seed)
        best: tuple[float, list[list[int]]] | None = None
        for _ in range(10_000):
            shuffled = students.copy()
            rng.shuffle(shuffled)
            groups = [shuffled[index::fold_count] for index in range(fold_count)]
            if not _assignment_has_both_classes(groups, students, labels, observation_days):
                continue
            score = _balance_score(groups, labels, observation_days)
            if best is None or score < best[0]:
                best = (score, groups)
                if score == 0:
                    break
        if best is not None:
            all_students = frozenset(students)
            return [
                CommonFold(
                    fold=index + 1,
                    train_students=all_students - frozenset(group),
                    validation_students=frozenset(group),
                )
                for index, group in enumerate(best[1])
            ]
    raise ValueError("Could not create class-valid shared student folds")


def run_common_cohort_experiment(
    rows: Sequence[ModelRow],
    observation_days: Sequence[int],
    max_folds: int = 4,
    bootstrap_samples: int = 1000,
    random_seed: int = 42,
) -> CommonCohortResult:
    """Evaluate day-specific models on one cohort and shared student folds."""
    cohort = create_common_cohort(rows, observation_days)
    folds = create_shared_student_folds(cohort, observation_days, max_folds, random_seed)
    prediction_rows: list[dict[str, object]] = []
    fold_results: list[dict[str, object]] = []
    fitted: list[Pipeline] = []

    for day in observation_days:
        day_rows = [row for row in cohort.rows if int(row["observation_day"]) == day]
        for fold in folds:
            train = [row for row in day_rows if int(row["student_id"]) in fold.train_students]
            validation = [row for row in day_rows if int(row["student_id"]) in fold.validation_students]
            train_targets = [int(row["future_inactivity"]) for row in train]
            actual = [int(row["future_inactivity"]) for row in validation]
            if set(train_targets) != {0, 1} or set(actual) != {0, 1}:
                raise ValueError(f"Both classes required for Day {day}, fold {fold.fold}")
            pipeline = build_logistic_pipeline(random_seed)
            fitted.append(pipeline)
            pipeline.fit(_matrix(train), train_targets)
            probabilities = [float(value) for value in pipeline.predict_proba(_matrix(validation))[:, 1]]
            predictions = [int(value >= 0.50) for value in probabilities]
            classifier = pipeline.named_steps["classifier"]
            iterations = max(int(value) for value in classifier.n_iter_)
            fold_results.append(
                {
                    "observation_day": day,
                    "fold": fold.fold,
                    "training_students": len(train),
                    "validation_students": len(validation),
                    "training_rows": len(train),
                    "validation_rows": len(validation),
                    "validation_positive_labels": sum(actual),
                    "converged": iterations < classifier.max_iter,
                    "iterations": iterations,
                    **calculate_metrics(actual, predictions, probabilities),
                }
            )
            for row, probability, prediction in zip(validation, probabilities, predictions):
                prediction_rows.append(
                    {
                        "student_id": int(row["student_id"]),
                        "observation_day": day,
                        "fold": fold.fold,
                        "actual_future_inactivity": int(row["future_inactivity"]),
                        "personal_change_probability": probability,
                        "personal_change_prediction": prediction,
                        "prediction_source": "out_of_fold",
                    }
                )

    expected = cohort.students_after * len(observation_days)
    if len(prediction_rows) != expected:
        raise RuntimeError("Every common-cohort row must receive one OOF prediction")
    prediction_rows.sort(key=lambda row: (int(row["student_id"]), int(row["observation_day"])))
    pooled = _pooled(prediction_rows, observation_days)
    summaries = _summaries(fold_results, observation_days)
    intervals = bootstrap_confidence_intervals(
        prediction_rows, observation_days, bootstrap_samples, random_seed
    )
    paired = _paired_day_differences(pooled)
    return CommonCohortResult(
        cohort, folds, fold_results, prediction_rows, pooled, summaries, intervals, paired, fitted
    )


def student_bootstrap_samples(
    student_ids: Sequence[int], samples: int, random_seed: int
) -> Iterator[list[int]]:
    """Yield bootstrap draws of students; callers retain all rows per draw."""
    rng = random.Random(random_seed)
    population = list(student_ids)
    for _ in range(samples):
        yield [rng.choice(population) for _ in population]


def bootstrap_confidence_intervals(
    predictions: Sequence[dict[str, object]],
    observation_days: Sequence[int],
    samples: int = 1000,
    random_seed: int = 42,
) -> dict[int, dict[str, dict[str, float | int | None]]]:
    """Calculate percentile intervals by resampling whole student trajectories."""
    by_student: dict[int, list[dict[str, object]]] = {}
    for row in predictions:
        by_student.setdefault(int(row["student_id"]), []).append(row)
    required = set(observation_days)
    if any({int(row["observation_day"]) for row in values} != required for values in by_student.values()):
        raise ValueError("Bootstrap students must retain all requested observation days")
    collected = {
        day: {metric: [] for metric in ["average_precision", "roc_auc", "recall", "precision", "f1"]}
        for day in observation_days
    }
    for draw in student_bootstrap_samples(sorted(by_student), samples, random_seed):
        sampled = [row for student in draw for row in by_student[student]]
        for day in observation_days:
            selected = [row for row in sampled if int(row["observation_day"]) == day]
            actual = [int(row["actual_future_inactivity"]) for row in selected]
            if set(actual) != {0, 1}:
                continue
            metrics = calculate_metrics(
                actual,
                [int(row["personal_change_prediction"]) for row in selected],
                [float(row["personal_change_probability"]) for row in selected],
            )
            for metric in collected[day]:
                collected[day][metric].append(float(metrics[metric]))
    return {
        day: {
            metric: {
                "lower_95": float(np.percentile(values, 2.5)) if values else None,
                "upper_95": float(np.percentile(values, 97.5)) if values else None,
                "valid_samples": len(values),
                "skipped_samples": samples - len(values),
            }
            for metric, values in day_metrics.items()
        }
        for day, day_metrics in collected.items()
    }


def _matrix(rows: Sequence[ModelRow]) -> list[list[float]]:
    return [
        [float("nan") if row.get(column) is None else float(row[column]) for column in PERSONAL_CHANGE_FEATURE_COLUMNS]
        for row in rows
    ]


def _assignment_has_both_classes(groups, students, labels, days):
    all_students = set(students)
    return all(
        {labels[(student, day)] for student in group} == {0, 1}
        and {labels[(student, day)] for student in all_students - set(group)} == {0, 1}
        for group in groups
        for day in days
    )


def _balance_score(groups, labels, days):
    score = 0.0
    for day in days:
        counts = [sum(labels[(student, day)] for student in group) for group in groups]
        target = sum(counts) / len(groups)
        score += sum((count - target) ** 2 for count in counts)
    sizes = [len(group) for group in groups]
    target_size = sum(sizes) / len(groups)
    return score * 1000 + sum((size - target_size) ** 2 for size in sizes)


def _pooled(predictions, days):
    output = {}
    for day in days:
        selected = [row for row in predictions if int(row["observation_day"]) == day]
        actual = [int(row["actual_future_inactivity"]) for row in selected]
        prevalence = sum(actual) / len(actual)
        metrics = calculate_metrics(
            actual,
            [int(row["personal_change_prediction"]) for row in selected],
            [float(row["personal_change_probability"]) for row in selected],
        )
        output[day] = {
            "eligible_rows": len(selected),
            "unique_students": len({row["student_id"] for row in selected}),
            "future_inactive_rows": sum(actual),
            "positive_prevalence": prevalence,
            "pr_auc_lift": float(metrics["average_precision"]) / prevalence,
            **metrics,
        }
    return output


def _summaries(fold_results, days):
    output = {}
    for day in days:
        selected = [row for row in fold_results if int(row["observation_day"]) == day]
        output[day] = {}
        for metric in DAY_SUMMARY_METRICS:
            values = [float(row[metric]) for row in selected]
            output[day][metric] = {
                "mean": statistics.fmean(values),
                "standard_deviation": statistics.stdev(values),
            }
    return output


def _paired_day_differences(pooled):
    comparisons = [(28, 14), (42, 28), (56, 42), (56, 14)]
    metrics = ["average_precision", "pr_auc_lift", "recall", "precision", "f1", "brier_score"]
    return {
        f"day_{left}_minus_day_{right}": {
            metric: float(pooled[left][metric]) - float(pooled[right][metric])
            for metric in metrics
        }
        for left, right in comparisons
    }
