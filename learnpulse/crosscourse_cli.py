"""CLI for day-specific grouped and unseen-course logistic evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.crosscourse_experiment import (
    load_parquet_rows,
    run_crosscourse_experiment,
    save_crosscourse_outputs,
)

DEFAULTS = {
    "results": Path("experiments/crosscourse_results.json"),
    "fold_results": Path("experiments/crosscourse_fold_results.csv"),
    "predictions": Path("experiments/crosscourse_predictions.parquet"),
    "summary": Path("experiments/crosscourse_summary.csv"),
    "feature_results": Path("experiments/crosscourse_feature_results.csv"),
}


def parse_args() -> argparse.Namespace:
    """Parse experiment selections, filters, and artifact paths."""
    parser = argparse.ArgumentParser(description="Evaluate cross-course OULAD generalization")
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--observation-days", nargs="+", type=int, default=[14, 28, 42, 56])
    parser.add_argument("--models", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"])
    parser.add_argument("--evaluation-designs", nargs="+", choices=["grouped", "presentation", "module"], default=["grouped", "presentation", "module"])
    parser.add_argument("--overlap-policies", nargs="+", choices=["course-only", "student-disjoint"], default=["course-only", "student-disjoint"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=DEFAULTS["results"])
    parser.add_argument("--fold-results", type=Path, default=DEFAULTS["fold_results"])
    parser.add_argument("--predictions", type=Path, default=DEFAULTS["predictions"])
    parser.add_argument("--summary", type=Path, default=DEFAULTS["summary"])
    parser.add_argument("--feature-results", type=Path, default=DEFAULTS["feature_results"])
    parser.add_argument("--only-module")
    parser.add_argument("--only-presentation")
    parser.add_argument("--only-observation-day", type=int)
    parser.add_argument("--only-model", choices=["A", "B", "C"])
    parser.add_argument("--only-evaluation-design", choices=["grouped", "presentation", "module"])
    return parser.parse_args()


def main() -> None:
    """Load once from Parquet, execute selected holdouts, and publish atomically."""
    args = parse_args()
    filtered = any(value is not None for value in [args.only_module, args.only_presentation, args.only_observation_day, args.only_model, args.only_evaluation_design])
    paths = {
        "results": args.results, "fold_results": args.fold_results,
        "predictions": args.predictions, "summary": args.summary,
        "feature_results": args.feature_results,
    }
    if filtered and any(paths[name].resolve() == DEFAULTS[name].resolve() for name in paths):
        raise SystemExit("Filtered runs must use separate paths for all outputs; complete experiment artifacts are protected")
    days = [args.only_observation_day] if args.only_observation_day is not None else args.observation_days
    models = [args.only_model] if args.only_model else args.models
    designs = [args.only_evaluation_design] if args.only_evaluation_design else args.evaluation_designs
    try:
        rows = load_parquet_rows(args.modeling_table, days)
        result = run_crosscourse_experiment(
            rows, days, models, designs, args.overlap_policies, args.folds,
            args.threshold, args.random_seed, args.only_module, args.only_presentation,
        )
        configuration = {
            "modeling_table": str(args.modeling_table.resolve()), "observation_days": days,
            "models": models, "evaluation_designs": designs,
            "overlap_policies": args.overlap_policies, "folds": args.folds,
            "threshold": args.threshold, "random_seed": args.random_seed,
            "only_module": args.only_module, "only_presentation": args.only_presentation,
            "filtered_run": filtered,
        }
        report = save_crosscourse_outputs(
            result, args.modeling_table, args.results, args.fold_results,
            args.predictions, args.summary, args.feature_results, configuration,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "fitted_models": result.fitted_models,
        "prediction_rows": len(result.predictions),
        "runtime_seconds": result.report["runtime_seconds"],
        "filtered_run": filtered,
        "source_unchanged": report["source_integrity"]["unchanged"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
