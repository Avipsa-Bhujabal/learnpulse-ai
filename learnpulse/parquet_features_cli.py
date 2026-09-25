"""CLI for trusted rolling and personal-baseline features from Parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.parquet_reader import (
    calculate_parquet_features,
    combined_feature_messages,
)


def parse_args() -> argparse.Namespace:
    """Parse the Parquet partition, student, timeline, and display limit."""
    parser = argparse.ArgumentParser(description="Calculate trusted features from daily Parquet")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--presentation", required=True)
    parser.add_argument("--student-id", type=int)
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    parser.add_argument("--minimum-history", type=int, default=7)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    """Calculate every row before applying an optional display-only limit."""
    args = parse_args()
    if args.start_day is not None and args.end_day is not None and args.start_day > args.end_day:
        raise SystemExit("--start-day cannot be greater than --end-day")
    if args.minimum_history < 0:
        raise SystemExit("--minimum-history cannot be negative")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit cannot be negative")
    try:
        result = calculate_parquet_features(
            args.dataset, args.module, args.presentation, args.student_id,
            args.start_day, args.end_day, args.minimum_history,
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if not result.daily:
        raise SystemExit("No matching daily activity records were found")
    messages = combined_feature_messages(result)
    displayed = limit_messages(messages, args.limit)
    for message in displayed:
        print(json.dumps(message, allow_nan=False))


def limit_messages(
    messages: list[dict[str, object]], limit: int | None
) -> list[dict[str, object]]:
    """Limit already-calculated display rows without altering feature history."""
    if limit is not None and limit < 0:
        raise ValueError("limit cannot be negative")
    return messages if limit is None else messages[:limit]


if __name__ == "__main__":
    main()
