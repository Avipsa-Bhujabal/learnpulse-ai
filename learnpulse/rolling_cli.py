"""CLI for rolling seven-day features from real OULAD activity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import iter_daily_events, load_activity_types
from learnpulse.rolling_features import calculate_rolling_features

DEFAULT_DATA_DIR = Path("data/raw/oulad")


def parse_args() -> argparse.Namespace:
    """Parse one student's course and optional inclusive day range."""
    parser = argparse.ArgumentParser(description="Calculate rolling OULAD activity")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--student-id", type=int, required=True)
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="include each active day's internal resource and activity-type IDs",
    )
    return parser.parse_args()


def main() -> None:
    """Read real events and print one rolling row per selected course day."""
    args = parse_args()
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

    rows = calculate_rolling_features(daily, args.start_day, args.end_day)
    daily_by_day = {summary.day: summary for summary in daily}
    for row in rows:
        message = row.as_message()
        if args.debug:
            summary = daily_by_day.get(row.day)
            message["daily_resource_ids"] = sorted(summary.resource_ids) if summary else []
            message["daily_activity_types"] = sorted(summary.activity_types) if summary else []
        print(json.dumps(message))


if __name__ == "__main__":
    main()
