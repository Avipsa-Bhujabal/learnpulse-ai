"""Command-line replay of real OULAD interaction data."""

import argparse
import json
import time
from pathlib import Path

from learnpulse.oulad import load_replay_window

DEFAULT_DATA_DIR = Path("data/raw/oulad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay real OULAD daily events")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--module", default="AAA")
    parser.add_argument("--presentation", default="2013J")
    parser.add_argument("--start-day", type=int)
    parser.add_argument("--end-day", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.delay < 0:
        raise SystemExit("--delay cannot be negative")

    events = load_replay_window(
        args.data_dir,
        args.module,
        args.presentation,
        args.start_day,
        args.end_day,
    )
    if not events:
        raise SystemExit("No matching OULAD records were found")

    print(
        f"Loaded {len(events):,} real daily records for "
        f"{args.module} {args.presentation}. Replaying {min(args.limit, len(events)):,}."
    )
    for event in events[: args.limit]:
        print(json.dumps(event.as_message()), flush=True)
        if args.delay:
            time.sleep(args.delay)


if __name__ == "__main__":
    main()

