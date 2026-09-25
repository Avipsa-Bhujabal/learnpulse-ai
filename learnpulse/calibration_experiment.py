"""Nested, leakage-safe calibration and threshold evaluation for Model B."""

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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.baseline_models import build_logistic_pipeline
from learnpulse.crosscourse_experiment import (
    FORBIDDEN_COLUMNS,
    calculate_metrics_safe,
    create_course_holdout_splits,
    load_parquet_rows,  # noqa: F401 -- retained as a documented compatibility re-export
)

MODEL_B_FEATURE_COLUMNS = list(PERSONAL_CHANGE_FEATURE_COLUMNS)
CALIBRATION_METHODS = ("uncalibrated", "sigmoid", "isotonic")
THRESHOLD_POLICIES = ("fixed", "recall_focused", "f1_focused", "alert_capacity", "cost_sensitive")
THRESHOLD_CHART_METRICS = ("precision", "recall", "f1", "alert_rate")
CLIP_EPSILON = 1e-15


@dataclass(frozen=True)
class InnerFold:
    fold: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_students: frozenset[int]
    validation_students: frozenset[int]


@dataclass(frozen=True)
class CalibrationResult:
    holdout_results: list[dict[str, object]]
    predictions: list[dict[str, object]]
    calibration_curves: list[dict[str, object]]
    threshold_results: list[dict[str, object]]
    threshold_policies: dict[str, object]
    report: dict[str, object]


class SigmoidCalibrator:
    """Platt-style logistic mapping fitted to clipped score logits."""

    def __init__(self, random_seed: int = 42):
        self.model = LogisticRegression(random_state=random_seed, max_iter=1000)

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> SigmoidCalibrator:
        self.model.fit(_logits(scores), labels)
        return self

    def predict(self, scores: Sequence[float]) -> np.ndarray:
        return self.model.predict_proba(_logits(scores))[:, 1]


def validate_calibration_schema() -> None:
    """Require the exact Step 7 Model B feature schema with no forbidden fields."""
    if MODEL_B_FEATURE_COLUMNS != list(PERSONAL_CHANGE_FEATURE_COLUMNS):
        raise ValueError("Calibration must use the exact Model B feature schema")
    forbidden = sorted(set(MODEL_B_FEATURE_COLUMNS) & FORBIDDEN_COLUMNS)
    if forbidden:
        raise ValueError(f"Forbidden calibration features: {forbidden}")


def create_inner_folds(
    rows: Sequence[dict[str, object]], outer_train_indices: Sequence[int],
    requested_folds: int = 5, random_seed: int = 42,
) -> tuple[list[InnerFold], str | None]:
    """Create the largest class-valid student-grouped folds inside outer training."""
    outer = tuple(outer_train_indices)
    for count in range(min(requested_folds, len({int(rows[i]["student_id"]) for i in outer})), 1, -1):
        targets = [int(rows[i]["future_inactivity"]) for i in outer]
        groups = [int(rows[i]["student_id"]) for i in outer]
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=random_seed)
        result = []
        valid = True
        for fold, (train_local, validation_local) in enumerate(splitter.split(np.zeros(len(outer)), targets, groups), 1):
            train = tuple(outer[int(i)] for i in train_local)
            validation = tuple(outer[int(i)] for i in validation_local)
            train_students = frozenset(int(rows[i]["student_id"]) for i in train)
            validation_students = frozenset(int(rows[i]["student_id"]) for i in validation)
            if train_students & validation_students or _classes(rows, train) != {0, 1} or _classes(rows, validation) != {0, 1}:
                valid = False; break
            result.append(InnerFold(fold, train, validation, train_students, validation_students))
        if valid:
            reason = None if count == requested_folds else f"Reduced from {requested_folds} to {count} because a higher count lacked both classes"
            return result, reason
    raise ValueError("Could not create class-valid student-grouped inner folds")


def inner_oof_scores(
    rows: Sequence[dict[str, object]], outer_train_indices: Sequence[int],
    folds: Sequence[InnerFold], random_seed: int,
) -> tuple[list[float], list[int], int, list[object]]:
    """Generate exactly one score per outer-training row from an unseen-student fold."""
    scores: dict[int, float] = {}
    fitted = []
    for fold in folds:
        pipeline = build_logistic_pipeline(random_seed)
        pipeline.fit(_matrix(rows, fold.train_indices), _targets(rows, fold.train_indices))
        fitted.append(pipeline)
        values = pipeline.predict_proba(_matrix(rows, fold.validation_indices))[:, 1]
        for index, value in zip(fold.validation_indices, values):
            if index in scores: raise RuntimeError("Outer-training row received duplicate inner OOF scores")
            scores[index] = float(value)
    if set(scores) != set(outer_train_indices):
        raise RuntimeError("Every outer-training row must receive one inner OOF score")
    ordered = list(outer_train_indices)
    return [scores[i] for i in ordered], _targets(rows, ordered), len(folds), fitted


def fit_calibrators(
    scores: Sequence[float], labels: Sequence[int], random_seed: int,
) -> tuple[dict[str, object | None], list[str]]:
    """Fit sigmoid and isotonic mappings only from inner OOF scores."""
    if set(labels) != {0, 1}: raise ValueError("Calibration training requires both classes")
    messages = []
    sigmoid = SigmoidCalibrator(random_seed).fit(scores, labels)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
    positives = sum(labels)
    if positives < 100:
        messages.append(f"Isotonic calibration has only {positives} positive inner OOF rows and may be unstable")
    return {"uncalibrated": None, "sigmoid": sigmoid, "isotonic": isotonic}, messages


def apply_calibrator(method: str, calibrator: object | None, scores: Sequence[float]) -> list[float]:
    """Return unchanged base scores or bounded calibrated outputs."""
    if method == "uncalibrated": return [float(value) for value in scores]
    if method == "sigmoid" or method == "isotonic": values = calibrator.predict(scores)  # type: ignore[union-attr]
    else: raise ValueError(f"Unknown calibration method: {method}")
    return [float(np.clip(value, 0.0, 1.0)) for value in values]


def calibration_bins(
    labels: Sequence[int], probabilities: Sequence[float], bins: int = 10,
) -> list[dict[str, object]]:
    """Return all fixed-width bins; exactly 1.0 belongs to the final bin."""
    if bins <= 0: raise ValueError("bins must be greater than zero")
    output = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        positions = [i for i, value in enumerate(probabilities) if lower <= value < upper or index == bins - 1 and value == 1.0]
        mean = statistics.fmean(probabilities[i] for i in positions) if positions else None
        observed = statistics.fmean(labels[i] for i in positions) if positions else None
        output.append({"bin_lower": lower, "bin_upper": upper, "row_count": len(positions), "mean_predicted_value": mean, "observed_positive_rate": observed, "absolute_calibration_gap": abs(mean-observed) if mean is not None and observed is not None else None})
    return output


def calibration_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, object]:
    """Calculate calibration, discrimination, and fixed-threshold diagnostics."""
    predicted = [int(value >= .5) for value in probabilities]
    bins = calibration_bins(labels, probabilities)
    ece = sum(int(row["row_count"]) / len(labels) * float(row["absolute_calibration_gap"]) for row in bins if row["row_count"])
    mce = max((float(row["absolute_calibration_gap"]) for row in bins if row["row_count"]), default=None)
    intercept, slope, warning = calibration_intercept_slope(labels, probabilities)
    base = calculate_metrics_safe(labels, predicted, probabilities)
    return {**base, "brier_score": float(brier_score_loss(labels, probabilities)), "log_loss": float(log_loss(labels, np.clip(probabilities, CLIP_EPSILON, 1-CLIP_EPSILON), labels=[0, 1])), "expected_calibration_error": ece, "maximum_calibration_error": mce, "calibration_intercept": intercept, "calibration_slope": slope, "calibration_diagnostic_warning": warning}


def calibration_intercept_slope(labels: Sequence[int], probabilities: Sequence[float]) -> tuple[float | None, float | None, str | None]:
    """Fit a diagnostic outcome~logit(score) regression; never alter predictions."""
    if set(labels) != {0, 1}: return None, None, "Calibration intercept/slope require both classes"
    try:
        model = LogisticRegression(penalty=None, max_iter=1000).fit(_logits(probabilities), labels)
        if max(model.n_iter_) >= model.max_iter: return None, None, "Calibration diagnostic did not converge"
        return float(model.intercept_[0]), float(model.coef_[0][0]), None
    except Exception as error:
        return None, None, f"Calibration diagnostic failed: {error}"


def threshold_candidates(scores: Sequence[float], labels: Sequence[int]) -> list[dict[str, float | int]]:
    """Evaluate all unique score thresholds plus warning-all/none boundaries efficiently."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    sorted_scores = np.asarray(scores)[order]; sorted_labels = np.asarray(labels)[order]
    positives, negatives = int(sum(labels)), len(labels)-int(sum(labels))
    output = [_threshold_point(1.0000000000000002, 0, 0, positives, negatives, len(labels))]
    tp = fp = 0; index = 0
    while index < len(order):
        value = float(sorted_scores[index])
        while index < len(order) and float(sorted_scores[index]) == value:
            if int(sorted_labels[index]): tp += 1
            else: fp += 1
            index += 1
        output.append(_threshold_point(value, tp, fp, positives-tp, negatives-fp, len(labels)))
    if not output or float(output[-1]["threshold"]) != 0.0:
        output.append(_threshold_point(0.0, positives, negatives, 0, 0, len(labels)))
    return output


def _threshold_point(threshold: float, tp: int, fp: int, fn: int, tn: int, total: int) -> dict[str, float | int]:
    precision = tp/(tp+fp) if tp+fp else 0.0; recall = tp/(tp+fn) if tp+fn else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0, "alert_rate": (tp+fp)/total, "false_positives": fp, "false_negatives": fn, "true_positives": tp, "true_negatives": tn}


def select_thresholds(
    scores: Sequence[float], labels: Sequence[int], fixed: float = .5,
    recall_target: float = .8, alert_capacity: float = .1,
    false_negative_cost: float = 5, false_positive_cost: float = 1,
) -> dict[str, dict[str, object]]:
    """Select five deterministic policies using training OOF predictions only."""
    candidates = threshold_candidates(scores, labels)
    fixed_point = _evaluate_threshold(scores, labels, fixed)
    feasible = [row for row in candidates if float(row["recall"]) >= recall_target]
    recall_row = max(feasible, key=lambda row: (float(row["threshold"]), float(row["precision"]))) if feasible else max(candidates, key=lambda row: (float(row["recall"]), float(row["precision"]), float(row["threshold"])))
    f1_row = max(candidates, key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["threshold"])))
    capacity = [row for row in candidates if float(row["alert_rate"]) <= alert_capacity]
    capacity_row = max(capacity, key=lambda row: (float(row["recall"]), -float(row["threshold"])))
    cost_row = min(candidates, key=lambda row: (false_negative_cost*int(row["false_negatives"])+false_positive_cost*int(row["false_positives"]), -float(row["recall"]), -float(row["threshold"])))
    chosen = {"fixed": fixed_point, "recall_focused": recall_row, "f1_focused": f1_row, "alert_capacity": capacity_row, "cost_sensitive": cost_row}
    for policy, row in chosen.items():
        row["selection_dataset"] = "inner_oof_training"
        row["fallback"] = policy == "recall_focused" and not feasible
    return chosen


def _evaluate_threshold(scores: Sequence[float], labels: Sequence[int], threshold: float) -> dict[str, object]:
    predicted = [int(value >= threshold) for value in scores]
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0,1]).ravel()
    return _threshold_point(threshold, int(tp), int(fp), int(fn), int(tn), len(labels))


def _logits(scores: Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), CLIP_EPSILON, 1-CLIP_EPSILON)
    return np.log(clipped/(1-clipped)).reshape(-1, 1)


def _matrix(rows, indices):
    return [[math.nan if rows[i].get(column) is None else float(rows[i][column]) for column in MODEL_B_FEATURE_COLUMNS] for i in indices]


def _targets(rows, indices): return [int(rows[i]["future_inactivity"]) for i in indices]
def _classes(rows, indices): return set(_targets(rows, indices))


def run_calibration_experiment(
    rows: Sequence[dict[str, object]], observation_days: Sequence[int] = (28, 56),
    methods: Sequence[str] = CALIBRATION_METHODS, inner_folds: int = 5,
    fixed_threshold: float = .5, recall_target: float = .8,
    alert_capacity: float = .1, false_negative_cost: float = 5,
    false_positive_cost: float = 1, random_seed: int = 42,
    only_module: str | None = None, only_presentation: str | None = None,
) -> CalibrationResult:
    """Run strict nested calibration for each day and held-out presentation."""
    validate_calibration_schema()
    _validate_args(observation_days, methods, inner_folds, fixed_threshold, recall_target, alert_capacity, false_negative_cost, false_positive_cost)
    started = time.perf_counter()
    holdout_results, predictions, threshold_results = [], [], []
    warnings_: list[dict[str, object]] = []
    outer_fits = inner_fits = 0
    for day in observation_days:
        splits = create_course_holdout_splits(rows, day, "presentation", ["student-disjoint"], only_module, only_presentation)
        for split in splits:
            if set(split.train_indices) & set(split.test_indices): raise RuntimeError("Outer train/test row overlap")
            train_students = {int(rows[i]["student_id"]) for i in split.train_indices}
            test_students = {int(rows[i]["student_id"]) for i in split.test_indices}
            overlap = train_students & test_students
            if overlap: raise RuntimeError("Strict outer holdout contains student overlap")
            folds, reduction = create_inner_folds(rows, split.train_indices, inner_folds, random_seed)
            oof_scores, train_labels, used_folds, fitted_inner = inner_oof_scores(rows, split.train_indices, folds, random_seed)
            inner_fits += len(fitted_inner)
            calibrators, calibration_warnings = fit_calibrators(oof_scores, train_labels, random_seed)
            final_pipeline = build_logistic_pipeline(random_seed)
            final_pipeline.fit(_matrix(rows, split.train_indices), train_labels)
            outer_fits += 1
            base_test = [float(value) for value in final_pipeline.predict_proba(_matrix(rows, split.test_indices))[:,1]]
            test_labels = _targets(rows, split.test_indices)
            for warning in calibration_warnings:
                warnings_.append({"day": day, "holdout": split.holdout_id, "warning": warning})
            if reduction: warnings_.append({"day": day, "holdout": split.holdout_id, "warning": reduction})
            for method in methods:
                train_outputs = apply_calibrator(method, calibrators[method], oof_scores)
                test_outputs = apply_calibrator(method, calibrators[method], base_test)
                selected = select_thresholds(train_outputs, train_labels, fixed_threshold, recall_target, alert_capacity, false_negative_cost, false_positive_cost)
                metrics = calibration_metrics(test_labels, test_outputs)
                holdout_results.append({
                    "observation_day": day, "held_out_module": split.held_out_module,
                    "held_out_presentation": split.held_out_presentation,
                    "outer_holdout_id": split.holdout_id, "calibration_method": method,
                    "test_rows": len(split.test_indices), "test_students": len(test_students),
                    "positive_rows": sum(test_labels), "prevalence": sum(test_labels)/len(test_labels),
                    "inner_fold_count": used_folds, "inner_fold_reduction_reason": reduction,
                    "overlapping_students_before_removal": split.overlapping_students_before_removal,
                    "training_students_removed": split.training_students_removed,
                    "training_rows_removed": split.training_rows_removed,
                    "student_overlap_after_removal": len(overlap),
                    "outer_model_converged": max(final_pipeline.named_steps["classifier"].n_iter_) < final_pipeline.named_steps["classifier"].max_iter,
                    "outer_model_iterations": int(max(final_pipeline.named_steps["classifier"].n_iter_)),
                    **metrics,
                })
                threshold_by_policy = {}
                prediction_by_policy = {}
                for policy, training in selected.items():
                    threshold = float(training["threshold"])
                    outer = _evaluate_threshold(test_outputs, test_labels, threshold)
                    threshold_results.append({
                        "observation_day": day, "held_out_module": split.held_out_module,
                        "held_out_presentation": split.held_out_presentation,
                        "outer_holdout_id": split.holdout_id, "calibration_method": method,
                        "threshold_policy": policy, "selection_dataset": "inner_oof_training",
                        "selected_threshold": threshold,
                        **{f"training_oof_{name}": training[name] for name in ["precision","recall","f1","alert_rate","false_positives","false_negatives"]},
                        "selection_fallback": training["fallback"],
                        **{f"outer_test_{name}": outer[name] for name in ["precision","recall","f1","alert_rate","false_positives","false_negatives","true_positives","true_negatives"]},
                        "outer_test_specificity": int(outer["true_negatives"])/(int(outer["true_negatives"])+int(outer["false_positives"])) if int(outer["true_negatives"])+int(outer["false_positives"]) else None,
                        "outer_test_cost": false_negative_cost*int(outer["false_negatives"])+false_positive_cost*int(outer["false_positives"]),
                    })
                    threshold_by_policy[f"{policy}_threshold"] = threshold
                    prediction_by_policy[f"{policy}_prediction"] = [int(value >= threshold) for value in test_outputs]
                for position, index in enumerate(split.test_indices):
                    row = rows[index]
                    predictions.append({
                        "module": row["module"], "presentation": row["presentation"],
                        "student_id": int(row["student_id"]), "observation_day": day,
                        "held_out_module": split.held_out_module,
                        "held_out_presentation": split.held_out_presentation,
                        "actual_label": test_labels[position], "base_model_score": base_test[position],
                        "calibration_method": method, "calibrated_output": test_outputs[position],
                        "outer_holdout_id": split.holdout_id, "inner_fold_count": used_folds,
                        "student_overlap_after_removal": len(overlap), **threshold_by_policy,
                        **{name: values[position] for name, values in prediction_by_policy.items()},
                    })
    curves = pooled_calibration_curves(predictions, observation_days, methods)
    policies = summarize_threshold_policies(threshold_results, predictions)
    summary = summarize_calibration(holdout_results, predictions)
    report = {
        "runtime_seconds": time.perf_counter()-started, "outer_model_fits": outer_fits,
        "inner_model_fits": inner_fits, "holdout_method_results": len(holdout_results),
        "prediction_rows": len(predictions), "warnings": warnings_,
        "calibration_summaries": summary,
        "paired_calibration_comparisons": paired_calibration_comparisons(holdout_results, threshold_results),
        "worst_presentations": worst_presentations(holdout_results, threshold_results),
    }
    return CalibrationResult(holdout_results, predictions, curves, threshold_results, policies, report)


def pooled_calibration_curves(predictions, days, methods):
    output = []
    for day in days:
        for method in methods:
            selected = [row for row in predictions if row["observation_day"] == day and row["calibration_method"] == method]
            for bin_ in calibration_bins([int(row["actual_label"]) for row in selected], [float(row["calibrated_output"]) for row in selected]):
                output.append({"observation_day": day, "calibration_method": method, **bin_})
    return output


def summarize_calibration(holdouts, predictions):
    output = []
    keys = sorted({(row["observation_day"], row["calibration_method"]) for row in holdouts})
    metric_names = ["brier_score","log_loss","expected_calibration_error","maximum_calibration_error","calibration_intercept","calibration_slope","roc_auc","average_precision","precision","recall","f1","predicted_positive_rows"]
    for day, method in keys:
        selected = [row for row in holdouts if row["observation_day"] == day and row["calibration_method"] == method]
        macro = {"observation_day": day, "calibration_method": method, "summary_type": "macro", "holdouts": len(selected)}
        for metric in metric_names:
            values = [float(row[metric]) for row in selected if row.get(metric) is not None]
            macro[f"{metric}_mean"] = statistics.fmean(values) if values else None
            macro[f"{metric}_standard_deviation"] = statistics.stdev(values) if len(values)>1 else None
        output.append(macro)
        rows_ = [row for row in predictions if row["observation_day"] == day and row["calibration_method"] == method]
        pooled = calibration_metrics([int(row["actual_label"]) for row in rows_], [float(row["calibrated_output"]) for row in rows_])
        output.append({"observation_day": day, "calibration_method": method, "summary_type": "pooled", "holdouts": len(selected), **pooled})
    return output


def summarize_threshold_policies(threshold_results, predictions):
    output = []
    keys = sorted({(row["observation_day"],row["calibration_method"],row["threshold_policy"]) for row in threshold_results})
    for day, method, policy in keys:
        held = [row for row in threshold_results if (row["observation_day"],row["calibration_method"],row["threshold_policy"])==(day,method,policy)]
        pred_rows = [row for row in predictions if row["observation_day"]==day and row["calibration_method"]==method]
        actual = [int(row["actual_label"]) for row in pred_rows]
        predicted = [int(row[f"{policy}_prediction"]) for row in pred_rows]
        tn,fp,fn,tp = confusion_matrix(actual,predicted,labels=[0,1]).ravel()
        precision=float(precision_score(actual,predicted,zero_division=0)); recall=float(recall_score(actual,predicted,zero_division=0))
        output.append({
            "observation_day":day,"calibration_method":method,"threshold_policy":policy,
            "selected_threshold_mean":statistics.fmean(float(row["selected_threshold"]) for row in held),
            "selected_threshold_standard_deviation":statistics.stdev(float(row["selected_threshold"]) for row in held) if len(held)>1 else None,
            "precision":precision,"recall":recall,"f1":float(f1_score(actual,predicted,zero_division=0)),
            "alert_rate":sum(predicted)/len(predicted),"false_positives":int(fp),"false_negatives":int(fn),
            "true_positives":int(tp),"true_negatives":int(tn),
        })
    return {"selection_rule":"Thresholds are selected separately inside each outer holdout using inner OOF training predictions only.","policies":output}


def paired_calibration_comparisons(holdouts, thresholds):
    """Return paired holdout method differences for calibration and policy metrics."""
    pairs = {"sigmoid_minus_uncalibrated":("sigmoid","uncalibrated"),"isotonic_minus_uncalibrated":("isotonic","uncalibrated"),"isotonic_minus_sigmoid":("isotonic","sigmoid")}
    output=[]
    base_metrics=["brier_score","log_loss","expected_calibration_error","average_precision"]
    for day in sorted({row["observation_day"] for row in holdouts}):
        index={(row["outer_holdout_id"],row["calibration_method"]):row for row in holdouts if row["observation_day"]==day}
        holdout_ids=sorted({key[0] for key in index})
        for label,(left,right) in pairs.items():
            if not all((unit,left) in index and (unit,right) in index for unit in holdout_ids):
                continue
            for metric in base_metrics:
                differences=[float(index[(unit,left)][metric])-float(index[(unit,right)][metric]) for unit in holdout_ids if index[(unit,left)].get(metric) is not None and index[(unit,right)].get(metric) is not None]
                output.append(_paired_summary(day,label,None,metric,differences))
    threshold_index={(row["observation_day"],row["outer_holdout_id"],row["calibration_method"],row["threshold_policy"]):row for row in thresholds}
    for day in sorted({row["observation_day"] for row in thresholds}):
        units=sorted({row["outer_holdout_id"] for row in thresholds if row["observation_day"]==day})
        policies=sorted({row["threshold_policy"] for row in thresholds})
        for label,(left,right) in pairs.items():
            if not all((day,unit,left,policy) in threshold_index and (day,unit,right,policy) in threshold_index for unit in units for policy in policies):
                continue
            for policy in policies:
                for metric in ["outer_test_f1","outer_test_recall","outer_test_precision"]:
                    differences=[float(threshold_index[(day,unit,left,policy)][metric])-float(threshold_index[(day,unit,right,policy)][metric]) for unit in units]
                    output.append(_paired_summary(day,label,policy,metric,differences))
    return output


def _paired_summary(day, comparison, policy, metric, values):
    return {"observation_day":day,"comparison":comparison,"threshold_policy":policy,"metric":metric,"mean_difference":statistics.fmean(values) if values else None,"standard_deviation":statistics.stdev(values) if len(values)>1 else None,"median_difference":statistics.median(values) if values else None,"left_favored":sum(value<0 for value in values) if metric in {"brier_score","log_loss","expected_calibration_error"} else sum(value>0 for value in values),"right_favored":sum(value>0 for value in values) if metric in {"brier_score","log_loss","expected_calibration_error"} else sum(value<0 for value in values),"ties":sum(value==0 for value in values),"valid_holdouts":len(values)}


def worst_presentations(holdouts, thresholds):
    """Rank error rates by their denominators, retaining raw-count rankings."""
    output=[]
    for day,method in sorted({(row["observation_day"],row["calibration_method"]) for row in holdouts}):
        selected=[row for row in holdouts if row["observation_day"]==day and row["calibration_method"]==method]
        for label,metric,operation in [("worst_brier","brier_score",max),("worst_ece","expected_calibration_error",max)]:
            row=operation(selected,key=lambda value:float(value[metric])); output.append({"observation_day":day,"calibration_method":method,"measure":label,"holdout":row["outer_holdout_id"],"value":row[metric]})
    for key in sorted({(r["observation_day"],r["calibration_method"],r["threshold_policy"]) for r in thresholds}):
        selected=[r for r in thresholds if (r["observation_day"],r["calibration_method"],r["threshold_policy"])==key]
        for label,metric,operation in [("worst_recall","outer_test_recall",min),("highest_false_positive_count","outer_test_false_positives",max),("highest_false_negative_count","outer_test_false_negatives",max)]:
            row=operation(selected,key=lambda value:float(value[metric])); output.append({"observation_day":key[0],"calibration_method":key[1],"threshold_policy":key[2],"measure":label,"holdout":row["outer_holdout_id"],"value":row[metric]})
        for label,numerator,other in [("highest_false_positive_rate","outer_test_false_positives","outer_test_true_negatives"),("highest_false_negative_rate","outer_test_false_negatives","outer_test_true_positives")]:
            valid=[]
            for row in selected:
                count=int(row[numerator]); denominator=count+int(row[other])
                if denominator:
                    valid.append((count/denominator,row))
                else:
                    output.append({"observation_day":key[0],"calibration_method":key[1],"threshold_policy":key[2],"measure":label,"holdout":row["outer_holdout_id"],"value":None,"warning":f"Undefined rate: {numerator} + {other} is zero"})
            if valid:
                rate,row=max(valid,key=lambda item:(item[0],item[1]["outer_holdout_id"]))
                output.append({"observation_day":key[0],"calibration_method":key[1],"threshold_policy":key[2],"measure":label,"holdout":row["outer_holdout_id"],"value":rate,"false_positives":int(row["outer_test_false_positives"]),"true_negatives":int(row["outer_test_true_negatives"]),"false_negatives":int(row["outer_test_false_negatives"]),"true_positives":int(row["outer_test_true_positives"])})
            else:
                output.append({"observation_day":key[0],"calibration_method":key[1],"threshold_policy":key[2],"measure":label,"holdout":None,"value":None,"warning":"No holdout has a defined denominator"})
    return output


def threshold_tradeoff_curves(predictions, days, methods):
    """Retrospective pooled 0.00–1.00 curves; never used for selection."""
    output=[]
    for day in days:
        for method in methods:
            selected=[row for row in predictions if row["observation_day"]==day and row["calibration_method"]==method]
            labels=[int(row["actual_label"]) for row in selected]; scores=[float(row["calibrated_output"]) for row in selected]
            for threshold in np.linspace(0,1,101):
                point=_evaluate_threshold(scores,labels,float(threshold))
                output.append({"observation_day":day,"calibration_method":method,"threshold":float(threshold),"precision":point["precision"],"recall":point["recall"],"f1":point["f1"],"alert_rate":point["alert_rate"],"selection_dataset":"retrospective_pooled_outer_test_visualization_only"})
    return output


def save_calibration_outputs(
    result: CalibrationResult, source: Path, results: Path, holdouts: Path,
    predictions: Path, curves: Path, threshold_results: Path,
    policies: Path, figures_dir: Path, configuration: dict[str,object],
) -> dict[str,object]:
    """Stage tables, JSON, and figures, then atomically publish every artifact."""
    source_before=_signature(source)
    targets=[results,holdouts,predictions,curves,threshold_results,policies,
             figures_dir/"day28_calibration.png",figures_dir/"day56_calibration.png",
             figures_dir/"day28_threshold_tradeoff.png",figures_dir/"day56_threshold_tradeoff.png"]
    if len({p.resolve() for p in targets})!=len(targets): raise ValueError("Output paths must be distinct")
    root=Path(tempfile.mkdtemp(prefix=".calibration_build_",dir=results.parent))
    try:
        staged=[root/p.name for p in targets]
        tradeoffs=threshold_tradeoff_curves(result.predictions,configuration["observation_days"],configuration["calibration_methods"])
        _write_csv(staged[1],result.holdout_results); _write_predictions(staged[2],result.predictions)
        _write_csv(staged[3],result.calibration_curves); _write_csv(staged[4],result.threshold_results)
        staged[5].write_text(json.dumps(result.threshold_policies,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        _plot_calibration(result.calibration_curves,28,staged[6]); _plot_calibration(result.calibration_curves,56,staged[7])
        _plot_thresholds(tradeoffs,28,staged[8]); _plot_thresholds(tradeoffs,56,staged[9])
        report={"created_at_utc":datetime.now(timezone.utc).isoformat(),"configuration":configuration,"runtime":result.report,"feature_schema":MODEL_B_FEATURE_COLUMNS,"package_versions":_versions(),"source_integrity":{"before":source_before,"unchanged":source_before==_signature(source)},"risk_bands_created":False,"risk_band_reason":"Calibration is evaluated as a research sensitivity analysis; no probability bands are justified without stable external evidence."}
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        _publish(list(zip(staged,targets))); return report
    finally:
        if root.exists(): shutil.rmtree(root)


def _plot_calibration(curves,day,path):
    fig,ax=plt.subplots(figsize=(7,5)); ax.plot([0,1],[0,1],"--",color="gray",label="perfect calibration")
    for method in CALIBRATION_METHODS:
        rows=[r for r in curves if r["observation_day"]==day and r["calibration_method"]==method and r["row_count"]]
        if rows: ax.plot([r["mean_predicted_value"] for r in rows],[r["observed_positive_rate"] for r in rows],marker="o",label=method)
    ax.set(xlabel="Model output",ylabel="Observed future-inactivity frequency",title=f"Day {day}: held-out presentation calibration"); ax.legend(); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def _plot_thresholds(rows,day,path):
    fig,axes=plt.subplots(1,4,figsize=(17,4),sharex=True)
    for method in CALIBRATION_METHODS:
        selected=[r for r in rows if r["observation_day"]==day and r["calibration_method"]==method]
        for ax,metric in zip(axes,THRESHOLD_CHART_METRICS): ax.plot([r["threshold"] for r in selected],[r[metric] for r in selected],label=method)
    for ax,title in zip(axes,["Precision","Recall","F1","Alert rate"]): ax.set(xlabel="Threshold",ylabel=title); ax.grid(alpha=.2)
    axes[0].legend(); fig.suptitle(f"Day {day}: retrospective held-out threshold trade-offs"); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def _write_csv(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def _write_predictions(path,rows):
    temporary=path.with_suffix(".csv"); _write_csv(temporary,rows)
    escaped=path.resolve().as_posix().replace("'","''")
    with duckdb.connect() as connection: connection.execute(f"COPY (SELECT * FROM read_csv_auto(?,header=true)) TO '{escaped}' (FORMAT PARQUET,COMPRESSION ZSTD)",[str(temporary.resolve())])
    temporary.unlink()


def refresh_saved_reporting(results: Path, holdouts: Path, thresholds: Path,
                            predictions: Path, figures_dir: Path) -> dict[str, object]:
    """Correct derived reporting from saved artifacts without fitting or selecting."""
    for path in (results, holdouts, thresholds, predictions):
        if not path.is_file(): raise FileNotFoundError(path)
    report=json.loads(results.read_text(encoding="utf-8"))
    with holdouts.open(newline="",encoding="utf-8") as handle:
        held=list(csv.DictReader(handle))
    with thresholds.open(newline="",encoding="utf-8") as handle:
        threshold_rows=list(csv.DictReader(handle))
    for row in (*held,*threshold_rows):
        row["observation_day"]=int(row["observation_day"])
    expected_h=report["runtime"]["holdout_method_results"]
    if len(held)!=expected_h or len(threshold_rows)!=expected_h*len(THRESHOLD_POLICIES):
        raise ValueError("Saved holdout or threshold artifact is incomplete")
    with duckdb.connect() as connection:
        score_rows=connection.execute(
            "SELECT observation_day, calibration_method, actual_label, calibrated_output "
            "FROM read_parquet(?)",[str(predictions.resolve())]
        ).fetchall()
    if len(score_rows)!=report["runtime"]["prediction_rows"]:
        raise ValueError("Saved prediction artifact is incomplete")
    curve_input=[{"observation_day":int(day),"calibration_method":method,
                  "actual_label":int(label),"calibrated_output":float(score)}
                 for day,method,label,score in score_rows]
    days=report["configuration"]["observation_days"]
    methods=report["configuration"]["calibration_methods"]
    tradeoffs=threshold_tradeoff_curves(curve_input,days,methods)
    report["runtime"]["worst_presentations"]=worst_presentations(held,threshold_rows)
    report["reporting_correction"]={"rate_definitions":"FPR=FP/(FP+TN); FNR=FN/(FN+TP)",
        "chart_metrics":list(THRESHOLD_CHART_METRICS),
        "inputs":"Previously published holdout, threshold, and prediction artifacts; no refitting or threshold selection"}
    targets=[results,figures_dir/"day28_threshold_tradeoff.png",figures_dir/"day56_threshold_tradeoff.png"]
    root=Path(tempfile.mkdtemp(prefix=".calibration_reporting_",dir=results.parent))
    try:
        staged=[root/path.name for path in targets]
        staged[0].write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        _plot_thresholds(tradeoffs,28,staged[1]); _plot_thresholds(tradeoffs,56,staged[2])
        _publish(list(zip(staged,targets)))
    finally:
        if root.exists(): shutil.rmtree(root)
    return report


def _publish(pairs):
    backups=[]; published=[]
    try:
        for _,target in pairs:
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                backup=target.with_name(f".{target.name}.previous")
                if backup.exists(): backup.unlink()
                os.replace(target,backup); backups.append((backup,target))
        for source,target in pairs: os.replace(source,target); published.append(target)
    except Exception:
        for target in published:
            if target.exists(): target.unlink()
        for backup,target in backups:
            if backup.exists(): os.replace(backup,target)
        raise
    for backup,_ in backups:
        if backup.exists(): backup.unlink()


def _signature(path):
    stat=path.stat(); return {"path":str(path.resolve()),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}


def _versions(): return {"python":platform.python_version(),"duckdb":duckdb.__version__,"numpy":np.__version__,"scikit-learn":importlib.metadata.version("scikit-learn"),"matplotlib":importlib.metadata.version("matplotlib")}


def _validate_args(days,methods,folds,threshold,recall,capacity,fn_cost,fp_cost):
    if not days or len(days)!=len(set(days)): raise ValueError("Observation days must be unique and non-empty")
    if not methods or not set(methods)<=set(CALIBRATION_METHODS): raise ValueError("Invalid calibration methods")
    if folds<2: raise ValueError("inner_folds must be at least two")
    if not 0<=threshold<=1 or not 0<=recall<=1 or not 0<=capacity<=1: raise ValueError("Threshold, recall target, and capacity must be between zero and one")
    if fn_cost<0 or fp_cost<0: raise ValueError("Costs cannot be negative")
