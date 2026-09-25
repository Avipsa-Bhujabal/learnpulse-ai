"""CLI for the first LearnPulse future-inactivity model comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from learnpulse.baseline_models import (
    MODEL_FEATURE_COLUMNS,
    RULE_DESCRIPTION,
    load_modeling_rows,
    run_baseline_experiment,
    save_pipeline,
)

DEFAULT_RESULTS = Path("experiments/baseline_results.json")
DEFAULT_PREDICTIONS = Path("experiments/baseline_predictions.csv")
DEFAULT_ARTIFACT = Path("artifacts/logistic_regression_pipeline.joblib")


def parse_args() -> argparse.Namespace:
    """Parse model-ready input and reproducible student-split parameters."""
    parser = argparse.ArgumentParser(description="Evaluate simple LearnPulse models")
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    """Train three simple models and save auditable evaluation outputs."""
    args = parse_args()
    rows = load_modeling_rows(args.modeling_table)
    result = run_baseline_experiment(rows, args.test_size, args.random_seed)
    save_pipeline(result.logistic_pipeline, args.artifact)
    _write_predictions(args.predictions, result)

    train_positive = _positive_rate(result.split.train_rows)
    test_positive = _positive_rate(result.split.test_rows)
    classifier = result.logistic_pipeline.named_steps["classifier"]
    converged = all(iterations < classifier.max_iter for iterations in classifier.n_iter_)
    report = {
        "research_target": "future_inactivity",
        "dummy_strategy": "prior",
        "rule": RULE_DESCRIPTION,
        "logistic_threshold": 0.50,
        "random_seed": args.random_seed,
        "test_size": args.test_size,
        "train_students": len(result.split.train_student_ids),
        "test_students": len(result.split.test_student_ids),
        "student_overlap": len(result.split.train_student_ids & result.split.test_student_ids),
        "train_rows": len(result.split.train_rows),
        "test_rows": len(result.split.test_rows),
        "train_positive_rate": train_positive,
        "test_positive_rate": test_positive,
        "features": MODEL_FEATURE_COLUMNS,
        "preprocessing": "training-only median imputation with missing indicators, then standard scaling",
        "logistic_converged": converged,
        "logistic_iterations": [int(value) for value in classifier.n_iter_],
        "metrics": result.metrics,
        "artifact": str(args.artifact),
        "predictions": str(args.predictions),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Training students: {report['train_students']}")
    print(f"Test students: {report['test_students']}")
    print(f"Training rows: {report['train_rows']}")
    print(f"Test rows: {report['test_rows']}")
    print(f"Training positive-label rate: {train_positive:.4%}")
    print(f"Test positive-label rate: {test_positive:.4%}")
    print(f"Student overlap: {report['student_overlap']}")
    print(f"Rule: {RULE_DESCRIPTION}")
    for name, metrics in result.metrics.items():
        print(f"{name}: {json.dumps(metrics)}")
    print("Logistic threshold: 0.50")
    print(f"Logistic converged: {converged}; iterations: {report['logistic_iterations']}")
    print(f"Results: {args.results}")
    print(f"Predictions: {args.predictions}")
    print(f"Pipeline artifact: {args.artifact}")


def _positive_rate(rows: list[dict[str, object]]) -> float:
    return sum(int(row["future_inactivity"]) for row in rows) / len(rows)


def _write_predictions(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "module",
        "presentation",
        "student_id",
        "observation_day",
        "actual_future_inactivity",
        "dummy_prediction",
        "rule_prediction",
        "logistic_prediction",
        "logistic_probability",
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(result.split.test_rows):
            writer.writerow(
                {
                    "module": row["module"],
                    "presentation": row["presentation"],
                    "student_id": row["student_id"],
                    "observation_day": row["observation_day"],
                    "actual_future_inactivity": row["future_inactivity"],
                    "dummy_prediction": result.dummy_predictions[index],
                    "rule_prediction": result.rule_predictions[index],
                    "logistic_prediction": result.logistic_predictions[index],
                    "logistic_probability": result.logistic_probabilities[index],
                }
            )


if __name__ == "__main__":
    main()
