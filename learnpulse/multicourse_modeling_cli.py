"""Command-line entry point for the partition-wise multi-course dataset build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from learnpulse.multicourse_modeling_table import build_multicourse_modeling_table


def parse_args() -> argparse.Namespace:
    """Parse paths, methodology parameters, filters, and resume behavior."""
    parser = argparse.ArgumentParser(description="Build the OULAD multi-course modeling table")
    parser.add_argument("--daily-dataset", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--observation-days", nargs="+", type=int, default=[14, 28, 42, 56])
    parser.add_argument("--future-window-days", type=int, default=14)
    parser.add_argument("--minimum-history", type=int, default=7)
    parser.add_argument("--module")
    parser.add_argument("--presentation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temporary-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    """Build, validate, publish, and print a concise JSON summary."""
    args = parse_args()
    try:
        result = build_multicourse_modeling_table(
            daily_dataset=args.daily_dataset,
            metadata_dir=args.metadata_dir,
            output_parquet=args.output_parquet,
            output_csv=args.output_csv,
            report_path=args.report,
            observation_days=args.observation_days,
            future_window_days=args.future_window_days,
            minimum_history=args.minimum_history,
            module=args.module,
            presentation=args.presentation,
            resume=args.resume,
            temporary_dir=args.temporary_dir,
            logger=lambda message: print(message, flush=True),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    report = result.report
    print(json.dumps({
        "model_ready_rows": report["model_ready_rows"],
        "unique_students": report["unique_students"],
        "partitions": len(report["source_partitions_processed"]),
        "processing_time_seconds": report["processing_time_seconds"],
        "output_parquet": str(result.output_parquet),
        "output_csv": str(result.output_csv),
        "report": str(result.report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
