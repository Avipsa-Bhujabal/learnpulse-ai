"""Build validated CSV modeling tables from the real OULAD pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from learnpulse.features import aggregate_daily_activity
from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    build_modeling_table,
    load_student_results,
    missing_values_by_feature,
    write_modeling_csv,
)
from learnpulse.oulad import iter_daily_events, load_activity_types
from learnpulse.outcomes import (
    calculate_future_inactivity_outcomes,
    load_course_lengths,
    load_withdrawal_days,
)
from learnpulse.personal_baseline import calculate_personal_baselines
from learnpulse.rolling_features import calculate_rolling_features

DEFAULT_DATA_DIR = Path("data/raw/oulad")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    """Parse course, observation days, and eligibility parameters."""
    parser = argparse.ArgumentParser(description="Build an OULAD modeling table")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--observation-days", nargs="+", type=int, default=[14, 28, 42, 56])
    parser.add_argument("--future-window-days", type=int, default=14)
    parser.add_argument("--minimum-history", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    """Run Steps 1–4, join candidates, validate, write, and summarize."""
    args = parse_args()
    days = sorted(args.observation_days)
    if len(days) != len(set(days)):
        raise SystemExit("--observation-days must not contain duplicates")
    if args.future_window_days <= 0:
        raise SystemExit("--future-window-days must be greater than zero")
    if args.minimum_history < 0:
        raise SystemExit("--minimum-history cannot be negative")

    student_results = load_student_results(args.data_dir / "studentInfo.csv")
    student_keys = sorted(
        key for key in student_results if key[:2] == (args.module, args.presentation)
    )
    lookup = load_activity_types(args.data_dir / "vle.csv")
    daily = aggregate_daily_activity(
        iter_daily_events(
            args.data_dir / "studentVle.csv", lookup, args.module, args.presentation
        )
    )
    rolling = calculate_rolling_features(daily, start_day=0, end_day=max(days))
    baselines = calculate_personal_baselines(rolling, args.minimum_history)
    observations = (
        (module, presentation, student_id, day)
        for module, presentation, student_id in student_keys
        for day in days
    )
    outcomes = calculate_future_inactivity_outcomes(
        daily,
        observations,
        load_course_lengths(args.data_dir / "courses.csv"),
        load_withdrawal_days(args.data_dir / "studentRegistration.csv"),
        args.future_window_days,
    )
    result = build_modeling_table(rolling, baselines, outcomes, student_results, days)

    included_path = args.output_dir / "modeling_table.csv"
    excluded_path = args.output_dir / "modeling_table_excluded.csv"
    write_modeling_csv(included_path, result.model_ready_rows)
    write_modeling_csv(excluded_path, result.excluded_rows)

    all_rows = result.all_rows
    targets = Counter(row["future_inactivity"] for row in result.model_ready_rows)
    day_counts = Counter(int(row["observation_day"]) for row in all_rows)
    exclusion_counts: Counter[str] = Counter()
    for row in result.excluded_rows:
        exclusion_counts.update(str(row["exclusion_reasons"]).split("|"))

    print(f"Total candidate rows: {len(all_rows)}")
    print(f"Model-ready rows: {len(result.model_ready_rows)}")
    print(f"Excluded rows: {len(result.excluded_rows)}")
    print(f"Unique students: {len({row['student_id'] for row in all_rows})}")
    print(f"Rows at each observation day: {dict(sorted(day_counts.items()))}")
    print(f"Future-inactive count: {targets[True]}")
    print(f"Future-active count: {targets[False]}")
    percentage = targets[True] / len(result.model_ready_rows) * 100 if result.model_ready_rows else 0.0
    print(f"Future-inactivity percentage: {percentage:.2f}%")
    print(f"Missing values by feature: {missing_values_by_feature(result.model_ready_rows)}")
    print(f"Exclusions by reason: {dict(sorted(exclusion_counts.items()))}")
    print(f"Duplicate-key count: {result.duplicate_key_count}")
    print(f"Model feature columns: {MODEL_FEATURE_COLUMNS}")
    print(f"Model-ready output: {included_path}")
    print(f"Excluded output: {excluded_path}")


if __name__ == "__main__":
    main()
