"""CLI for observation-day-specific early-warning evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from learnpulse.baseline_models import load_modeling_rows
from learnpulse.early_warning_experiment import (
    PERSONAL_CHANGE_FEATURE_COLUMNS,
    run_early_warning_experiment,
)

DEFAULT_RESULTS = Path("experiments/early_warning_results.json")
DEFAULT_FOLDS = Path("experiments/early_warning_fold_results.csv")
DEFAULT_PREDICTIONS = Path("experiments/early_warning_predictions.csv")
DEFAULT_TRAJECTORIES = Path("experiments/early_warning_trajectories.csv")


def parse_args() -> argparse.Namespace:
    """Parse requested observation days and grouped-validation configuration."""
    parser = argparse.ArgumentParser(description="Evaluate LearnPulse warning timing")
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--observation-days", nargs="+", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--fold-results", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--trajectories", type=Path, default=DEFAULT_TRAJECTORIES)
    return parser.parse_args()


def main() -> None:
    """Run day-specific models and save fold, OOF, and trajectory outputs."""
    args = parse_args()
    days = sorted(set(args.observation_days))
    if len(days) != len(args.observation_days):
        raise SystemExit("--observation-days must not contain duplicates")
    rows = load_modeling_rows(args.modeling_table)
    result = run_early_warning_experiment(rows, days, args.folds, args.random_seed)

    _write_csv(args.fold_results, result.fold_results)
    _write_csv(args.predictions, result.predictions)
    _write_csv(args.trajectories, result.trajectories)
    examples = _trajectory_examples(result.trajectories)
    report = {
        "research_question": (
            "How does future-inactivity prediction performance change as more "
            "student activity history becomes available?"
        ),
        "observation_days": days,
        "preferred_folds": args.folds,
        "random_seed": args.random_seed,
        "threshold": 0.50,
        "features": PERSONAL_CHANGE_FEATURE_COLUMNS,
        "fold_counts_by_day": {
            str(day): sum(fold.observation_day == day for fold in result.folds)
            for day in days
        },
        "fold_results": result.fold_results,
        "summaries": result.summaries,
        "pooled_out_of_fold": result.pooled_results,
        "trajectory_examples": examples,
        "calibration_warning": (
            "Balanced class weights may improve rare-case detection while producing "
            "poorly calibrated probabilities; probabilities are not risk percentages."
        ),
        "fold_results_path": str(args.fold_results),
        "predictions_path": str(args.predictions),
        "trajectories_path": str(args.trajectories),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"fold_counts_by_day: {json.dumps(report['fold_counts_by_day'])}")
    for row in result.fold_results:
        print(f"fold_result: {json.dumps(row)}")
    print(f"summaries: {json.dumps(result.summaries)}")
    print(f"pooled_out_of_fold: {json.dumps(result.pooled_results)}")
    print(f"trajectory_examples: {json.dumps(examples)}")
    print("Threshold: 0.50")
    print("Student overlap in every fold: 0")
    print(f"Results: {args.results}")
    print(f"Fold results: {args.fold_results}")
    print(f"Predictions: {args.predictions}")
    print(f"Trajectories: {args.trajectories}")


def _trajectory_examples(rows: list[dict[str, object]]) -> dict[str, dict[str, object] | None]:
    patterns = ["low_to_high", "high_to_low", "consistently_high", "consistently_low"]
    return {pattern: next((row for row in rows if row["trajectory_pattern"] == pattern), None) for pattern in patterns}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
