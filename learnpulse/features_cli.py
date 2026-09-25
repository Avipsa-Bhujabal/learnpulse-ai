"""Print daily student activity summaries from the real OULAD files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import iter_daily_events, load_activity_types

DEFAULT_DATA_DIR = Path("data/raw/oulad")


def parse_args() -> argparse.Namespace:
    """Parse command-line filters for one OULAD student."""
    parser = argparse.ArgumentParser(description="Summarize daily OULAD activity")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--student-id", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    """Load, aggregate, and print a student's summaries chronologically."""
    args = parse_args()
    activity_types = load_activity_types(args.data_dir / "vle.csv")
    course_events = iter_daily_events(
        args.data_dir / "studentVle.csv",
        activity_types,
        args.module,
        args.presentation,
    )
    summaries = aggregate_daily_activity(
        event for event in course_events if event.student_id == args.student_id
    )
    if not summaries:
        raise SystemExit("No matching OULAD records were found")

    for summary in summaries:
        print(json.dumps(summary.as_message()))


if __name__ == "__main__":
    main()
