"""CLI for leakage-safe personal baselines from real OULAD activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import iter_daily_events, load_activity_types
from learnpulse.personal_baseline import calculate_personal_baselines
from learnpulse.rolling_features import calculate_rolling_features

DEFAULT_DATA_DIR = Path("data/raw/oulad")


def parse_args() -> argparse.Namespace:
    """Parse the student, inclusive course-day range, and history threshold."""
    parser = argparse.ArgumentParser(description="Calculate personal OULAD baselines")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--student-id", type=int, required=True)
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    parser.add_argument("--minimum-history", type=int, default=7)
    return parser.parse_args()


def _rounded_message(message: dict[str, object]) -> dict[str, object]:
    """Round floating-point fields for readable CLI output only."""
    return {
        key: round(value, 4) if isinstance(value, float) else value
        for key, value in message.items()
    }


def main() -> None:
    """Read real events and print chronological personal-baseline rows."""
    args = parse_args()
    if args.minimum_history < 0:
        raise SystemExit("--minimum-history cannot be negative")
    if args.start_day is not None and args.end_day is not None and args.start_day > args.end_day:
        raise SystemExit("--start-day cannot be greater than --end-day")

    lookup = load_activity_types(args.data_dir / "vle.csv")
    events = iter_daily_events(
        args.data_dir / "studentVle.csv", lookup, args.module, args.presentation
    )
    daily = aggregate_daily_activity(
        event for event in events if event.student_id == args.student_id
    )
    if not daily:
        raise SystemExit("No matching OULAD records were found")

    rolling = calculate_rolling_features(daily, args.start_day, args.end_day)
    baselines = calculate_personal_baselines(rolling, args.minimum_history)
    for baseline in baselines:
        print(json.dumps(_rounded_message(baseline.as_message())))


if __name__ == "__main__":
    main()
