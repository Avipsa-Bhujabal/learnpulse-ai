"""CLI for grouped cross-validation feature ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from learnpulse.ablation_experiment import (
    MODEL_FEATURE_GROUPS,
    run_ablation_experiment,
)
from learnpulse.baseline_models import load_modeling_rows

DEFAULT_RESULTS = Path("experiments/ablation_results.json")
DEFAULT_FOLDS = Path("experiments/ablation_fold_results.csv")
DEFAULT_PREDICTIONS = Path("experiments/ablation_predictions.csv")


def parse_args() -> argparse.Namespace:
    """Parse input path and grouped cross-validation settings."""
    parser = argparse.ArgumentParser(description="Compare LearnPulse feature groups")
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fold-results", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    return parser.parse_args()


def main() -> None:
    """Run ablation, save fold/OOF artifacts, and print full summaries."""
    args = parse_args()
    rows = load_modeling_rows(args.modeling_table)
    result = run_ablation_experiment(rows, args.folds, args.random_seed)
    _write_fold_results(args.fold_results, result.fold_results)
    _write_predictions(args.predictions, result.predictions)

    report = {
        "research_question": (
            "Does adding within-student personal-change information improve "
            "future-inactivity prediction over absolute activity alone?"
        ),
        "folds": args.folds,
        "random_seed": args.random_seed,
        "threshold": 0.50,
        "feature_groups": MODEL_FEATURE_GROUPS,
        "fold_definitions": [
            {
                "fold": fold.fold,
                "training_students": len(fold.train_students),
                "validation_students": len(fold.validation_students),
                "training_rows": len(fold.train_indices),
                "validation_rows": len(fold.validation_indices),
                "validation_positive_labels": sum(
                    int(rows[index]["future_inactivity"])
                    for index in fold.validation_indices
                ),
                "student_overlap": len(fold.train_students & fold.validation_students),
            }
            for fold in result.folds
        ],
        "fold_results": result.fold_results,
        "summaries": result.summaries,
        "pooled_out_of_fold": result.pooled_results,
        "paired_differences": result.paired_differences,
        "calibration_warning": (
            "Balanced class weights may improve rare-case detection while producing "
            "poorly calibrated probabilities; no calibration was applied."
        ),
        "fold_results_path": str(args.fold_results),
        "predictions_path": str(args.predictions),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for fold in report["fold_definitions"]:
        print(f"fold: {json.dumps(fold)}")
    for row in result.fold_results:
        print(f"fold_result: {json.dumps(row)}")
    print(f"summaries: {json.dumps(result.summaries)}")
    print(f"pooled_out_of_fold: {json.dumps(result.pooled_results)}")
    print(f"paired_differences: {json.dumps(result.paired_differences)}")
    print("Threshold: 0.50")
    print("Student overlap in every fold: 0")
    print(f"Results: {args.results}")
    print(f"Fold results: {args.fold_results}")
    print(f"Predictions: {args.predictions}")


def _write_fold_results(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["confusion_matrix"] = json.dumps(output["confusion_matrix"])
            writer.writerow(output)


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
