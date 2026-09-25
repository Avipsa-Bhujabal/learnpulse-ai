"""CLI for future-inactivity outcomes from real OULAD data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import iter_daily_events, load_activity_types
from learnpulse.outcomes import (
    calculate_future_inactivity_outcomes,
    load_course_lengths,
    load_withdrawal_days,
)

DEFAULT_DATA_DIR = Path("data/raw/oulad")


def parse_args() -> argparse.Namespace:
    """Parse one student, inclusive observation range, and future length."""
    parser = argparse.ArgumentParser(description="Create OULAD future-inactivity outcomes")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--student-id", type=int, required=True)
    parser.add_argument("--start-day", type=int, required=True)
    parser.add_argument("--end-day", type=int, required=True)
    parser.add_argument("--future-window-days", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    """Read real metadata/activity and print chronological outcome rows."""
    args = parse_args()
    if args.start_day > args.end_day:
        raise SystemExit("--start-day cannot be greater than --end-day")
    if args.future_window_days <= 0:
        raise SystemExit("--future-window-days must be greater than zero")

    lookup = load_activity_types(args.data_dir / "vle.csv")
    events = iter_daily_events(
        args.data_dir / "studentVle.csv", lookup, args.module, args.presentation
    )
    daily = aggregate_daily_activity(
        event for event in events if event.student_id == args.student_id
    )
    observations = (
        (args.module, args.presentation, args.student_id, day)
        for day in range(args.start_day, args.end_day + 1)
    )
    outcomes = calculate_future_inactivity_outcomes(
        daily_summaries=daily,
        observation_days=observations,
        course_lengths=load_course_lengths(args.data_dir / "courses.csv"),
        withdrawal_days=load_withdrawal_days(args.data_dir / "studentRegistration.csv"),
        future_window_days=args.future_window_days,
    )
    for outcome in outcomes:
        print(json.dumps(outcome.as_message()))


if __name__ == "__main__":
    main()
