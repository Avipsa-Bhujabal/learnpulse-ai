"""CLI for building and validating the scalable OULAD activity layer."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from learnpulse.scalable_pipeline import (
    DuckDBConfig,
    build_scalable_pipeline,
    validate_duckdb_config,
    validate_scalable_pipeline,
    write_report,
)

DEFAULT_REPORT = Path("experiments/scalable_pipeline_report.json")


def parse_args() -> argparse.Namespace:
    """Parse source, generated-output, database, and validation settings."""
    parser = argparse.ArgumentParser(description="Build scalable OULAD daily activity data")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-temp-size", default="20GB")
    return parser.parse_args()


def main() -> None:
    """Run the requested mode, save its report, and print major results."""
    args = parse_args()
    config = DuckDBConfig(args.memory_limit, args.threads, args.max_temp_size)
    try:
        validate_duckdb_config(config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    clock = time.perf_counter()

    def log(message: str) -> None:
        print(f"[{time.perf_counter() - clock:8.2f}s] {message}", flush=True)

    previous_report = None
    if args.validate_only and args.report.exists():
        try:
            previous_report = json.loads(args.report.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_report = None
    if args.validate_only:
        result = validate_scalable_pipeline(
            args.data_dir, args.output_dir, args.database, include_parity=True, logger=log,
            config=config,
        )
    else:
        result = build_scalable_pipeline(
            args.data_dir, args.output_dir, args.database, logger=log, config=config
        )
    saved_report = result.report
    if args.validate_only and previous_report and "processing_time_seconds" in previous_report:
        saved_report = {
            **previous_report,
            "last_validation": result.report,
            "last_validation_successful": True,
        }
    write_report(saved_report, args.report)
    summary_keys = [
        "mode", "raw_student_vle_row_count", "daily_aggregated_row_count",
        "number_of_modules", "number_of_presentations", "number_of_students",
        "total_clicks_before_aggregation", "total_clicks_after_aggregation",
        "click_totals_match", "duplicate_daily_keys", "unmatched_site_ids",
        "invalid_negative_click_counts", "processed_invalid_negative_click_counts",
        "null_counts", "output_sizes_bytes", "processing_time_seconds",
        "validation_time_seconds", "duckdb_configuration",
        "temporary_disk_usage_bytes_observed", "spill_disk_usage_bytes_observed",
    ]
    print("summary: " + json.dumps({key: result.report.get(key) for key in summary_keys if key in result.report}))
    print("aaa_2013j_parity: " + json.dumps(result.report.get("aaa_2013j_parity")))
    print(f"rebuilt: {str(result.rebuilt).lower()}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
