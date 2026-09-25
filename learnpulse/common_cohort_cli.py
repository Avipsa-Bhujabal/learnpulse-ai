"""CLI for the common-cohort early-warning sensitivity analysis."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.baseline_models import load_modeling_rows
from learnpulse.common_cohort_experiment import run_common_cohort_experiment

DEFAULT_RESULTS = Path("experiments/common_cohort_results.json")
DEFAULT_FOLDS = Path("experiments/common_cohort_fold_results.csv")
DEFAULT_PREDICTIONS = Path("experiments/common_cohort_predictions.csv")
DEFAULT_STUDENT_FOLDS = Path("experiments/common_cohort_student_folds.csv")


def parse_args() -> argparse.Namespace:
    """Parse common-cohort, fold, and bootstrap configuration."""
    parser = argparse.ArgumentParser(description="Run common-cohort sensitivity analysis")
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--observation-days", nargs="+", type=int, required=True)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fold-results", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--student-folds", type=Path, default=DEFAULT_STUDENT_FOLDS)
    return parser.parse_args()


def main() -> None:
    """Run, save, and summarize the shared-cohort experiment."""
    args = parse_args()
    days = sorted(set(args.observation_days))
    if len(days) != len(args.observation_days):
        raise SystemExit("--observation-days must not contain duplicates")
    if args.bootstrap_samples <= 0:
        raise SystemExit("--bootstrap-samples must be greater than zero")
    rows = load_modeling_rows(args.modeling_table)
    result = run_common_cohort_experiment(
        rows, days, args.max_folds, args.bootstrap_samples, args.random_seed
    )
    _write_csv(args.fold_results, result.fold_results)
    _write_csv(args.predictions, result.predictions)
    assignments = [
        {"student_id": student, "fold": fold.fold}
        for fold in result.folds
        for student in sorted(fold.validation_students)
    ]
    _write_csv(args.student_folds, assignments)
    report = {
        "observation_days": days,
        "random_seed": args.random_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "students_before_filtering": result.cohort.students_before,
        "students_after_filtering": result.cohort.students_after,
        "students_removed": result.cohort.students_removed,
        "common_cohort_rows": len(result.cohort.rows),
        "fold_count": len(result.folds),
        "features": PERSONAL_CHANGE_FEATURE_COLUMNS,
        "fold_results": result.fold_results,
        "summaries": result.summaries,
        "pooled_out_of_fold": result.pooled_results,
        "bootstrap_confidence_intervals": result.confidence_intervals,
        "paired_day_differences": result.paired_day_differences,
        "calibration_warning": (
            "Separate balanced models produce uncalibrated probabilities; changes "
            "are exploratory and are not changes in true risk percentages."
        ),
        "fold_results_path": str(args.fold_results),
        "predictions_path": str(args.predictions),
        "student_folds_path": str(args.student_folds),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"cohort: {json.dumps({key: report[key] for key in ['students_before_filtering', 'students_after_filtering', 'students_removed', 'common_cohort_rows']})}")
    print(f"fold_count: {len(result.folds)}")
    print(f"fold_student_counts: {json.dumps({fold.fold: len(fold.validation_students) for fold in result.folds})}")
    print(f"pooled_out_of_fold: {json.dumps(result.pooled_results)}")
    print(f"summaries: {json.dumps(result.summaries)}")
    print(f"bootstrap_confidence_intervals: {json.dumps(result.confidence_intervals)}")
    print(f"paired_day_differences: {json.dumps(result.paired_day_differences)}")
    print("Identical student fold assignments reused at every observation day: true")
    print("Student overlap in every fold: 0")
    print(f"Results: {args.results}")
    print(f"Fold results: {args.fold_results}")
    print(f"Predictions: {args.predictions}")
    print(f"Student folds: {args.student_folds}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if "confusion_matrix" in output:
                output["confusion_matrix"] = json.dumps(output["confusion_matrix"])
            writer.writerow(output)


if __name__ == "__main__":
    main()
