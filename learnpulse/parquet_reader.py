"""Read scalable daily OULAD Parquet into the trusted feature data models."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import duckdb

from learnpulse.features import DailyStudentActivity
from learnpulse.personal_baseline import PersonalBaseline, calculate_personal_baselines
from learnpulse.rolling_features import (
    RollingStudentActivity,
    calculate_rolling_features,
)

REQUIRED_COLUMNS = frozenset(
    {
        "module", "presentation", "student_id", "day", "total_clicks",
        "number_of_resources_visited", "number_of_activity_types_used",
        "homepage_clicks", "forum_clicks", "content_clicks", "quiz_clicks",
        "resource_ids", "activity_types",
    }
)


@dataclass(frozen=True)
class ParquetFeatureResult:
    """Trusted daily, rolling, and personal-baseline records from Parquet."""

    daily: list[DailyStudentActivity]
    rolling: list[RollingStudentActivity]
    baselines: list[PersonalBaseline]
    candidate_parquet_files: int


def iter_daily_activity(
    dataset: Path,
    module: str,
    presentation: str,
    student_id: int | None = None,
    start_day: int | None = None,
    end_day: int | None = None,
    batch_size: int = 2048,
) -> Iterator[DailyStudentActivity]:
    """Yield filtered daily summaries using one pruned Parquet partition.

    Bounds are inclusive. DuckDB receives the exact module-presentation file
    list and parameterized row predicates, so unrelated partitions are never
    scanned and the complete dataset is never materialized in Python.
    """
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start_day cannot be greater than end_day")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    files = partition_files(dataset, module, presentation)
    if not files:
        return
    file_names = [str(path.resolve()) for path in files]
    connection = duckdb.connect(database=":memory:")
    try:
        columns = {
            str(row[0])
            for row in connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?, hive_partitioning=true)",
                [file_names],
            ).fetchall()
        }
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Parquet dataset is missing required columns: {missing}")
        conditions = ["module = ?", "presentation = ?"]
        parameters: list[object] = [file_names, module, presentation]
        if student_id is not None:
            conditions.append("student_id = ?")
            parameters.append(student_id)
        if start_day is not None:
            conditions.append("day >= ?")
            parameters.append(start_day)
        if end_day is not None:
            conditions.append("day <= ?")
            parameters.append(end_day)
        query = f"""
            SELECT module, presentation, student_id, day, total_clicks,
                   number_of_resources_visited, number_of_activity_types_used,
                   homepage_clicks, forum_clicks, content_clicks, quiz_clicks,
                   resource_ids, activity_types
            FROM read_parquet(?, hive_partitioning=true)
            WHERE {' AND '.join(conditions)}
            ORDER BY module, presentation, student_id, day
        """
        cursor = connection.execute(query, parameters)
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break
            for row in batch:
                yield _daily_from_row(row)
    except duckdb.Error as error:
        raise ValueError(f"Unable to read daily-activity Parquet: {error}") from error
    finally:
        connection.close()


def read_daily_activity(
    dataset: Path,
    module: str,
    presentation: str,
    student_id: int | None = None,
    start_day: int | None = None,
    end_day: int | None = None,
) -> list[DailyStudentActivity]:
    """Return the filtered daily summaries as a deterministic list."""
    return list(
        iter_daily_activity(
            dataset, module, presentation, student_id, start_day, end_day
        )
    )


def calculate_parquet_features(
    dataset: Path,
    module: str,
    presentation: str,
    student_id: int | None = None,
    start_day: int | None = None,
    end_day: int | None = None,
    minimum_history: int = 7,
) -> ParquetFeatureResult:
    """Feed Parquet daily rows through the existing rolling and baseline code.

    Daily history is intentionally not truncated to the output range. The
    trusted rolling function uses earlier observations for the first requested
    window and inactivity gap while using ``start_day`` as the completeness
    timeline boundary.
    """
    if minimum_history < 0:
        raise ValueError("minimum_history cannot be negative")
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start_day cannot be greater than end_day")
    daily = read_daily_activity(dataset, module, presentation, student_id)
    rolling = calculate_rolling_features(daily, start_day, end_day)
    baselines = calculate_personal_baselines(rolling, minimum_history)
    return ParquetFeatureResult(
        daily=daily,
        rolling=rolling,
        baselines=baselines,
        candidate_parquet_files=len(partition_files(dataset, module, presentation)),
    )


def combined_feature_messages(result: ParquetFeatureResult) -> list[dict[str, object]]:
    """Merge matching rolling and baseline fields for JSON Lines output."""
    rolling = {
        (row.module, row.presentation, row.student_id, row.day): row.as_message()
        for row in result.rolling
    }
    messages = []
    for baseline in result.baselines:
        key = (baseline.module, baseline.presentation, baseline.student_id, baseline.day)
        messages.append({**rolling[key], **baseline.as_message()})
    return messages


def partition_files(dataset: Path, module: str, presentation: str) -> list[Path]:
    """Resolve only the requested Hive partition without accepting traversal."""
    if not dataset.is_dir():
        raise FileNotFoundError(f"Parquet dataset directory does not exist: {dataset}")
    all_parquet = list(dataset.rglob("*.parquet"))
    if not all_parquet:
        raise FileNotFoundError(f"No Parquet files were found under: {dataset}")
    module_directory = _exact_child(dataset, f"module={module}")
    if module_directory is None:
        return []
    presentation_directory = _exact_child(module_directory, f"presentation={presentation}")
    if presentation_directory is None:
        return []
    return sorted(path for path in presentation_directory.glob("*.parquet") if path.is_file())


def _exact_child(parent: Path, name: str) -> Path | None:
    """Match one literal child name, preventing path-separator interpretation."""
    return next((path for path in parent.iterdir() if path.is_dir() and path.name == name), None)


def _daily_from_row(row: Sequence[object]) -> DailyStudentActivity:
    resources = frozenset(int(value) for value in row[11])
    activity_types = frozenset(str(value) for value in row[12])
    summary = DailyStudentActivity(
        module=str(row[0]), presentation=str(row[1]), student_id=int(row[2]), day=int(row[3]),
        total_clicks=int(row[4]), number_of_resources_visited=int(row[5]),
        number_of_activity_types_used=int(row[6]), homepage_clicks=int(row[7]),
        forum_clicks=int(row[8]), content_clicks=int(row[9]), quiz_clicks=int(row[10]),
        resource_ids=resources, activity_types=activity_types,
    )
    if summary.number_of_resources_visited != len(resources):
        raise ValueError("Stored resource count does not match resource_ids")
    if summary.number_of_activity_types_used != len(activity_types):
        raise ValueError("Stored activity-type count does not match activity_types")
    return summary
