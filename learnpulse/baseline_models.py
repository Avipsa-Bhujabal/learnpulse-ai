"""Simple, leakage-safe baseline models for future OULAD inactivity."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    validate_model_feature_columns,
)

ModelRow = dict[str, object]
RULE_DESCRIPTION = (
    "Warn when current_inactivity_gap >= 7, or when click percentage change "
    "is <= -50% and active_days_last_7_days <= 2."
)


@dataclass(frozen=True)
class StudentSplit:
    """Rows split into mutually exclusive training and test student groups."""

    train_rows: list[ModelRow]
    test_rows: list[ModelRow]
    train_student_ids: frozenset[int]
    test_student_ids: frozenset[int]


@dataclass(frozen=True)
class ExperimentResult:
    """Fitted estimators, predictions, probabilities, and metrics."""

    split: StudentSplit
    dummy_model: DummyClassifier
    logistic_pipeline: Pipeline
    dummy_predictions: list[int]
    rule_predictions: list[int]
    logistic_predictions: list[int]
    dummy_probabilities: list[float]
    logistic_probabilities: list[float]
    metrics: dict[str, dict[str, object]]


def load_modeling_rows(path: Path) -> list[ModelRow]:
    """Read model-ready CSV rows while preserving missing numeric values."""
    rows: list[ModelRow] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        for raw in csv.DictReader(source):
            row: ModelRow = {
                "module": raw["module"],
                "presentation": raw["presentation"],
                "student_id": int(raw["student_id"]),
                "observation_day": int(raw["observation_day"]),
                "future_inactivity": _parse_target(raw["future_inactivity"]),
            }
            for column in MODEL_FEATURE_COLUMNS:
                row[column] = None if raw[column] == "" else float(raw[column])
            rows.append(row)
    return rows


def split_rows_by_student(
    rows: Sequence[ModelRow], test_size: float = 0.20, random_seed: int = 42
) -> StudentSplit:
    """Create a reproducible student-level split stratified by any-positive status."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")
    by_student: dict[int, list[ModelRow]] = {}
    for row in rows:
        by_student.setdefault(int(row["student_id"]), []).append(row)
    students = sorted(by_student)
    profiles = [int(any(int(row["future_inactivity"]) == 1 for row in by_student[s])) for s in students]
    train_ids, test_ids = train_test_split(
        students,
        test_size=test_size,
        random_state=random_seed,
        stratify=profiles,
    )
    train_set = frozenset(train_ids)
    test_set = frozenset(test_ids)
    split = StudentSplit(
        train_rows=[row for row in rows if int(row["student_id"]) in train_set],
        test_rows=[row for row in rows if int(row["student_id"]) in test_set],
        train_student_ids=train_set,
        test_student_ids=test_set,
    )
    validate_student_split(split)
    return split


def validate_student_split(split: StudentSplit) -> None:
    """Reject student overlap, divided student histories, or single-class splits."""
    overlap = split.train_student_ids & split.test_student_ids
    if overlap:
        raise ValueError(f"Student overlap detected: {sorted(overlap)}")
    for rows, expected_ids, name in (
        (split.train_rows, split.train_student_ids, "training"),
        (split.test_rows, split.test_student_ids, "test"),
    ):
        actual_ids = {int(row["student_id"]) for row in rows}
        if actual_ids != set(expected_ids):
            raise ValueError(f"Not all {name} student rows stayed together")
        if {int(row["future_inactivity"]) for row in rows} != {0, 1}:
            raise ValueError(f"Both target classes are required in the {name} split")


def build_logistic_pipeline(random_seed: int = 42) -> Pipeline:
    """Create training-only median imputation, scaling, and logistic regression."""
    validate_training_feature_schema(MODEL_FEATURE_COLUMNS)
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", random_state=random_seed, max_iter=1000
                ),
            ),
        ]
    )


def validate_training_feature_schema(columns: Sequence[str]) -> None:
    """Assert that the experiment uses exactly the approved behavioral schema."""
    validate_model_feature_columns(columns)
    if list(columns) != MODEL_FEATURE_COLUMNS:
        raise ValueError("Training columns must exactly match MODEL_FEATURE_COLUMNS")


def rule_based_predictions(rows: Iterable[ModelRow]) -> list[int]:
    """Apply a fixed, interpretable rule selected without test-set tuning."""
    predictions: list[int] = []
    for row in rows:
        gap = row.get("current_inactivity_gap")
        percentage = row.get("click_percentage_change_from_personal_average")
        active_days = row.get("active_days_last_7_days")
        long_gap = gap is not None and float(gap) >= 7
        sharp_drop = (
            percentage is not None
            and active_days is not None
            and float(percentage) <= -50
            and float(active_days) <= 2
        )
        predictions.append(int(long_gap or sharp_drop))
    return predictions


def run_baseline_experiment(
    rows: Sequence[ModelRow], test_size: float = 0.20, random_seed: int = 42
) -> ExperimentResult:
    """Fit and evaluate dummy, fixed-rule, and logistic models on a student split."""
    validate_training_feature_schema(MODEL_FEATURE_COLUMNS)
    split = split_rows_by_student(rows, test_size, random_seed)
    x_train = feature_matrix(split.train_rows)
    x_test = feature_matrix(split.test_rows)
    y_train = target_vector(split.train_rows)
    y_test = target_vector(split.test_rows)

    dummy = DummyClassifier(strategy="prior", random_state=random_seed)
    dummy.fit(x_train, y_train)
    dummy_predictions = [int(value) for value in dummy.predict(x_test)]
    dummy_probabilities = _positive_probabilities(dummy, x_test)

    rule_predictions = rule_based_predictions(split.test_rows)

    logistic = build_logistic_pipeline(random_seed)
    logistic.fit(x_train, y_train)
    logistic_probabilities = [float(value) for value in logistic.predict_proba(x_test)[:, 1]]
    logistic_predictions = [int(value >= 0.50) for value in logistic_probabilities]

    metrics = {
        "dummy": calculate_metrics(y_test, dummy_predictions, dummy_probabilities),
        "rule_based": calculate_metrics(y_test, rule_predictions),
        "logistic_regression": calculate_metrics(
            y_test, logistic_predictions, logistic_probabilities
        ),
    }
    return ExperimentResult(
        split,
        dummy,
        logistic,
        dummy_predictions,
        rule_predictions,
        logistic_predictions,
        dummy_probabilities,
        logistic_probabilities,
        metrics,
    )


def calculate_metrics(
    actual: Sequence[int],
    predicted: Sequence[int],
    probabilities: Sequence[float] | None = None,
) -> dict[str, object]:
    """Calculate imbalance-aware binary metrics and confusion counts."""
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
        "brier_score": None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_positive_rows": int(sum(predicted)),
    }
    if probabilities is not None:
        result.update(
            roc_auc=float(roc_auc_score(actual, probabilities)),
            average_precision=float(average_precision_score(actual, probabilities)),
            brier_score=float(brier_score_loss(actual, probabilities)),
        )
    return result


def feature_matrix(rows: Iterable[ModelRow]) -> list[list[float]]:
    """Convert approved features to numeric rows, representing nulls as NaN."""
    validate_training_feature_schema(MODEL_FEATURE_COLUMNS)
    return [
        [math.nan if row.get(column) is None else float(row[column]) for column in MODEL_FEATURE_COLUMNS]
        for row in rows
    ]


def target_vector(rows: Iterable[ModelRow]) -> list[int]:
    """Convert boolean or binary target values safely to integers."""
    targets: list[int] = []
    for row in rows:
        value = row["future_inactivity"]
        if value not in {True, False}:
            raise ValueError(f"Invalid future_inactivity target: {value!r}")
        targets.append(int(value))
    return targets


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Persist a fitted logistic-regression pipeline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def _parse_target(value: str) -> int:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(f"Invalid future_inactivity target: {value!r}")


def _positive_probabilities(model: DummyClassifier, features: list[list[float]]) -> list[float]:
    probabilities = model.predict_proba(features)
    positive_index = list(model.classes_).index(1)
    return [float(value) for value in probabilities[:, positive_index]]
