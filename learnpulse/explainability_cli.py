"""Command line interface for transparent logistic-regression explanations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from learnpulse.baseline_models import load_modeling_rows
from learnpulse.explainability import (
    load_fold_assignments,
    run_explainability_experiment,
)

DEFAULT_RESULTS = Path("experiments/explainability_results.json")
DEFAULT_COEFFICIENTS = Path("experiments/global_coefficients.csv")
DEFAULT_IMPORTANCE = Path("experiments/permutation_importance.csv")
DEFAULT_INDIVIDUAL = Path("experiments/individual_explanations.csv")
DEFAULT_FIGURES = Path("experiments/figures")


def parse_args() -> argparse.Namespace:
    """Parse input, output, reproducibility, and permutation settings."""
    parser = argparse.ArgumentParser(description="Explain out-of-fold personal-change models")
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--fold-assignments", type=Path, required=True)
    parser.add_argument("--observation-days", nargs="+", type=int, required=True)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--permutation-repeats", type=int, default=30)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--permutation-importance", type=Path, default=DEFAULT_IMPORTANCE)
    parser.add_argument("--individual-explanations", type=Path, default=DEFAULT_INDIVIDUAL)
    parser.add_argument("--figures-directory", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> None:
    """Run explanations, persist detailed artifacts, and print core findings."""
    args = parse_args()
    rows = load_modeling_rows(args.modeling_table)
    assignments = load_fold_assignments(args.fold_assignments)
    result = run_explainability_experiment(
        rows, assignments, args.observation_days, args.random_seed, args.permutation_repeats
    )
    coefficient_output = [
        *({"record_scope": "fold", **row} for row in result.coefficient_rows),
        *({"record_scope": "summary", **row} for row in result.coefficient_summaries),
    ]
    importance_output = [
        *({"record_scope": "fold", **row} for row in result.permutation_rows),
        *({"record_scope": "summary", **row} for row in result.permutation_summaries),
    ]
    _write_csv(args.coefficients, coefficient_output)
    _write_csv(args.permutation_importance, importance_output)
    flattened_individual = [
        {
            **{key: value for key, value in row.items() if key not in {"top_increasing_contributions", "top_decreasing_contributions"}},
            "top_increasing_contributions": json.dumps(row.get("top_increasing_contributions")),
            "top_decreasing_contributions": json.dumps(row.get("top_decreasing_contributions")),
        }
        for row in result.individual_explanations
    ]
    _write_csv(args.individual_explanations, flattened_individual)
    charts = _create_charts(result.coefficient_summaries, result.permutation_summaries, args.figures_directory)
    report = {
        "observation_days": list(args.observation_days),
        "random_seed": args.random_seed,
        "permutation_repeats": args.permutation_repeats,
        "permutation_scoring": "average_precision",
        "permutation_partition": "validation_only",
        "students_explained_out_of_fold": all(not row["student_was_in_training"] for row in result.predictions),
        "coefficient_summaries": result.coefficient_summaries,
        "permutation_importance_summaries": result.permutation_summaries,
        "individual_explanations": result.individual_explanations,
        "stability": result.stability,
        "charts": [str(path) for path in charts],
        "interpretation_warning": "Coefficients describe associations, not causal effects. Correlated features can exchange importance.",
        "calibration_warning": "Balanced logistic model outputs are not calibrated risk percentages.",
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for day in args.observation_days:
        coefficients = [row for row in result.coefficient_summaries if row["observation_day"] == day][:5]
        importance = [row for row in result.permutation_summaries if row["observation_day"] == day][:5]
        print(f"day_{day}_top_coefficients: {json.dumps(coefficients)}")
        print(f"day_{day}_top_permutation_importance: {json.dumps(importance)}")
    print(f"stability: {json.dumps(result.stability)}")
    print(f"individual_explanations: {json.dumps(result.individual_explanations)}")
    print("Students explained with validation-fold models only: true")
    print(f"Results: {args.results}")
    print(f"Coefficients: {args.coefficients}")
    print(f"Permutation importance: {args.permutation_importance}")
    print(f"Individual explanations: {args.individual_explanations}")
    print(f"Charts: {', '.join(str(path) for path in charts)}")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write heterogeneous records using their ordered union of fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _create_charts(coefficient_rows, importance_rows, directory: Path) -> list[Path]:
    """Create readable coefficient and validation permutation bar charts."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for day in (28, 56):
        coefficients = [row for row in coefficient_rows if row["observation_day"] == day]
        coefficients.sort(key=lambda row: float(row["mean_standardized_coefficient"]))
        coefficient_path = directory / f"day{day}_coefficients.png"
        _bar_chart(
            coefficients,
            "feature",
            "mean_standardized_coefficient",
            "standard_deviation_across_folds",
            f"Day {day}: standardized logistic coefficients (associations)",
            "Mean coefficient across folds",
            coefficient_path,
            directional=True,
        )
        paths.append(coefficient_path)
        importance = [row for row in importance_rows if row["observation_day"] == day]
        importance.sort(key=lambda row: float(row["mean_importance"]))
        importance_path = directory / f"day{day}_permutation_importance.png"
        _bar_chart(
            importance,
            "feature",
            "mean_importance",
            "standard_deviation_across_folds",
            f"Day {day}: validation permutation importance (PR-AUC)",
            "Mean decrease in validation PR-AUC; ≤0 is weak or unstable",
            importance_path,
            directional=False,
        )
        paths.append(importance_path)
    return paths


def _bar_chart(rows, label_key, value_key, error_key, title, xlabel, path, directional):
    labels = [str(row[label_key]).replace("missing_indicator__", "missing: ") for row in rows]
    values = [float(row[value_key]) for row in rows]
    errors = [float(row[error_key]) for row in rows]
    colors = (["#c44e52" if value > 0 else "#4c72b0" for value in values] if directional else ["#55a868" if value > 0 else "#999999" for value in values])
    height = max(5.5, len(rows) * 0.42)
    fig, axis = plt.subplots(figsize=(11, height))
    axis.barh(labels, values, xerr=errors, color=colors, alpha=0.88, capsize=3)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
