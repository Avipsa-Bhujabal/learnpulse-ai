"""DuckDB/Parquet preparation layer for all OULAD module presentations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import duckdb

from learnpulse.features import DailyStudentActivity, aggregate_daily_activity
from learnpulse.oulad import iter_daily_events, load_activity_types

SOURCE_FILES = (
    "studentVle.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "courses.csv",
)
DAILY_KEY = ("module", "presentation", "student_id", "day")
DAILY_COLUMNS = (
    *DAILY_KEY,
    "total_clicks",
    "number_of_resources_visited",
    "number_of_activity_types_used",
    "homepage_clicks",
    "forum_clicks",
    "content_clicks",
    "quiz_clicks",
    "resource_ids",
    "activity_types",
)


@dataclass(frozen=True)
class PipelineResult:
    """Build/validation report and whether generated data was rebuilt."""

    report: dict[str, object]
    rebuilt: bool


@dataclass(frozen=True)
class DuckDBConfig:
    """Conservative DuckDB resource limits for a local batch build."""

    memory_limit: str = "2GB"
    threads: int = 2
    max_temp_size: str = "20GB"


_DUCKDB_SIZE = re.compile(r"^[1-9]\d*(?:\.\d+)?\s*(?:B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)$", re.IGNORECASE)


def validate_duckdb_config(config: DuckDBConfig) -> None:
    """Reject invalid thread counts and unsafe/non-DuckDB size strings."""
    if isinstance(config.threads, bool) or config.threads <= 0:
        raise ValueError("threads must be greater than zero")
    for name, value in (("memory_limit", config.memory_limit), ("max_temp_size", config.max_temp_size)):
        if not isinstance(value, str) or not _DUCKDB_SIZE.fullmatch(value.strip()):
            raise ValueError(f"{name} must be a positive DuckDB size such as 2GB or 512MB")


def configure_duckdb(
    connection: duckdb.DuckDBPyConnection, config: DuckDBConfig, temp_directory: Path
) -> dict[str, str]:
    """Apply and return DuckDB's effective resource configuration safely."""
    validate_duckdb_config(config)
    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute("SET memory_limit = ?", [config.memory_limit.strip()])
    connection.execute("SET threads = ?", [config.threads])
    connection.execute("SET preserve_insertion_order = false")
    connection.execute("SET temp_directory = ?", [str(temp_directory.resolve())])
    connection.execute("SET max_temp_directory_size = ?", [config.max_temp_size.strip()])
    names = ["memory_limit", "threads", "preserve_insertion_order", "temp_directory", "max_temp_directory_size"]
    return {
        name: str(connection.execute("SELECT current_setting(?)", [name]).fetchone()[0])
        for name in names
    }


def build_scalable_pipeline(
    data_dir: Path,
    output_dir: Path,
    database_path: Path,
    logger: Callable[[str], None] | None = None,
    include_parity: bool = True,
    config: DuckDBConfig | None = None,
) -> PipelineResult:
    """Build partitioned daily activity atomically, then validate it."""
    config = config or DuckDBConfig()
    log = logger or (lambda message: None)
    validate_duckdb_config(config)
    _validate_paths(data_dir, output_dir, database_path)
    started = time.perf_counter()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    controlled_temp = output_dir.parent / ".duckdb_temp"
    controlled_temp.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="build_", dir=controlled_temp))
    temporary_output = temporary_root / "daily_activity"
    joined_output = temporary_root / "joined_activity"
    spill_directory = temporary_root / "spill"
    work_database = temporary_root / "aggregation.duckdb"
    temporary_database = database_path.with_name(f".{database_path.name}.building")
    backup_output = output_dir.with_name(f".{output_dir.name}.previous")
    source_hashes = _source_hashes(data_dir)
    try:
        log("stage=duckdb_configuration")
        temporary_output.mkdir()
        joined_output.mkdir()
        with duckdb.connect(str(work_database)) as connection:
            effective_config = configure_duckdb(connection, config, spill_directory)
            _create_source_views(connection, data_dir)
            raw_stats = _raw_stats(connection)
            if int(raw_stats["invalid_negative_click_counts"]) != 0:
                raise ValueError("Negative studentVle sum_click values are not allowed")
            log("stage=join")
            _copy_joined_partitions(connection, joined_output)
            log("stage=aggregation")
            _copy_partitioned_daily(connection, joined_output, temporary_output, log)
            log("stage=validation")
            parquet_glob = _sql_path(temporary_output / "**" / "*.parquet")
            processed_stats = _processed_stats(connection, parquet_glob)
            ranges = _presentation_ranges(connection, parquet_glob)
        report: dict[str, object] = {
            **raw_stats,
            **processed_stats,
            "presentation_day_ranges": ranges,
            "click_totals_match": raw_stats["total_clicks_before_aggregation"]
            == processed_stats["total_clicks_after_aggregation"],
            "duckdb_configuration": effective_config,
            "partition_structure": "module=<code>/presentation=<code>/*.parquet",
        }
        if not report["click_totals_match"]:
            raise ValueError("Click totals before and after aggregation do not match")
        if int(report["duplicate_daily_keys"]) != 0:
            raise ValueError("Duplicate daily keys detected")
        if include_parity:
            log("stage=validation_AAA_2013J_python_parity")
            report["aaa_2013j_parity"] = validate_python_parity(data_dir, temporary_output, config, spill_directory)
        report["temporary_disk_usage_bytes_observed"] = _directory_size(temporary_root)
        report["spill_disk_usage_bytes_observed"] = _directory_size(spill_directory)
        log("stage=atomic_parquet_replace")
        _replace_directory(temporary_output, output_dir, backup_output)
        log("stage=create_metadata_database")
        _create_metadata_database(temporary_database, data_dir, output_dir, raw_stats)
        os.replace(temporary_database, database_path)
        report["output_sizes_bytes"] = _output_sizes(data_dir, output_dir, database_path)
        report["processing_time_seconds"] = time.perf_counter() - started
        report["mode"] = "build"
        report["source_sha256"] = source_hashes
        if source_hashes != _source_hashes(data_dir):
            raise RuntimeError("An original source file changed during processing")
        return PipelineResult(report, True)
    finally:
        log("stage=cleanup")
        if temporary_database.exists():
            temporary_database.unlink()
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if controlled_temp.exists() and not any(controlled_temp.iterdir()):
            controlled_temp.rmdir()


def validate_scalable_pipeline(
    data_dir: Path,
    output_dir: Path,
    database_path: Path,
    include_parity: bool = True,
    logger: Callable[[str], None] | None = None,
    config: DuckDBConfig | None = None,
) -> PipelineResult:
    """Validate existing generated data without rebuilding Parquet or DuckDB."""
    config = config or DuckDBConfig()
    log = logger or (lambda message: None)
    validate_duckdb_config(config)
    _validate_paths(data_dir, output_dir, database_path)
    if not output_dir.exists() or not database_path.exists():
        raise FileNotFoundError("Processed Parquet and metadata database must already exist")
    started = time.perf_counter()
    before = _generated_fingerprints(output_dir, database_path)
    controlled_temp = output_dir.parent / ".duckdb_temp"
    controlled_temp.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="validate_", dir=controlled_temp))
    try:
        log("stage=duckdb_configuration")
        parquet_glob = _sql_path(output_dir / "**" / "*.parquet")
        with duckdb.connect() as connection:
            effective_config = configure_duckdb(connection, config, temporary_root / "spill")
            log("stage=validation")
            _create_source_views(connection, data_dir)
            raw_stats = _raw_stats(connection)
            processed_stats = _processed_stats(connection, parquet_glob)
            ranges = _presentation_ranges(connection, parquet_glob)
    finally:
        log("stage=cleanup")
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        if controlled_temp.exists() and not any(controlled_temp.iterdir()):
            controlled_temp.rmdir()
    report: dict[str, object] = {
        **raw_stats,
        **processed_stats,
        "presentation_day_ranges": ranges,
        "click_totals_match": raw_stats["total_clicks_before_aggregation"]
        == processed_stats["total_clicks_after_aggregation"],
        "output_sizes_bytes": _output_sizes(data_dir, output_dir, database_path),
        "partition_structure": "module=<code>/presentation=<code>/*.parquet",
        "duckdb_configuration": effective_config,
        "validation_time_seconds": time.perf_counter() - started,
        "mode": "validate_only",
    }
    if not report["click_totals_match"]:
        raise ValueError("Click totals before and after aggregation do not match")
    if int(report["duplicate_daily_keys"]) != 0:
        raise ValueError("Duplicate daily keys detected")
    if int(report["invalid_negative_click_counts"]) != 0:
        raise ValueError("Negative click counts detected")
    if include_parity:
        log("stage=validate_AAA_2013J_python_parity")
        report["aaa_2013j_parity"] = validate_python_parity(data_dir, output_dir, config)
    report["validation_time_seconds"] = time.perf_counter() - started
    after = _generated_fingerprints(output_dir, database_path)
    if before != after:
        raise RuntimeError("Validation-only mode modified generated data")
    return PipelineResult(report, False)


def validate_python_parity(
    data_dir: Path,
    output_dir: Path,
    config: DuckDBConfig | None = None,
    temp_directory: Path | None = None,
) -> dict[str, object]:
    """Compare AAA 2013J Parquet rows with the trusted Python aggregation."""
    config = config or DuckDBConfig()
    activity_types = load_activity_types(data_dir / "vle.csv")
    python_rows = aggregate_daily_activity(
        iter_daily_events(
            data_dir / "studentVle.csv", activity_types, module="AAA", presentation="2013J"
        )
    )
    python_map = {_summary_key(row): _summary_values(row) for row in python_rows}
    parquet_glob = _sql_path(output_dir / "**" / "*.parquet")
    owned_temp: Path | None = None
    if temp_directory is None:
        controlled = output_dir.parent / ".duckdb_temp"
        controlled.mkdir(parents=True, exist_ok=True)
        owned_temp = Path(tempfile.mkdtemp(prefix="parity_", dir=controlled))
        temp_directory = owned_temp / "spill"
    try:
        with duckdb.connect() as connection:
            configure_duckdb(connection, config, temp_directory)
            result = connection.execute(
            f"""SELECT module, presentation, student_id, day, total_clicks,
                       number_of_resources_visited, number_of_activity_types_used,
                       homepage_clicks, forum_clicks, content_clicks, quiz_clicks,
                       resource_ids, activity_types
                FROM read_parquet('{parquet_glob}', hive_partitioning=true)
                WHERE module='AAA' AND presentation='2013J'
                ORDER BY student_id, day"""
            ).fetchall()
    finally:
        if owned_temp is not None and owned_temp.exists():
            shutil.rmtree(owned_temp)
            controlled = output_dir.parent / ".duckdb_temp"
            if controlled.exists() and not any(controlled.iterdir()):
                controlled.rmdir()
    duck_map = {
        (str(row[0]), str(row[1]), int(row[2]), int(row[3])): (
            *(int(row[index]) for index in range(4, 11)),
            frozenset(int(value) for value in row[11]),
            frozenset(str(value) for value in row[12]),
        )
        for row in result
    }
    keys_match = set(python_map) == set(duck_map)
    deterministic_keys = sorted(python_map)[:100]
    samples_match = len(deterministic_keys) == 100 and all(
        python_map[key] == duck_map.get(key) for key in deterministic_keys
    )
    student_key = ("AAA", "2013J", 28400, -10)
    student_comparison = {
        "key": list(student_key),
        "python": _serializable_values(python_map.get(student_key)),
        "duckdb": _serializable_values(duck_map.get(student_key)),
        "match": python_map.get(student_key) == duck_map.get(student_key),
    }
    category_indexes = range(3, 7)
    category_match = all(
        sum(values[index] for values in python_map.values())
        == sum(values[index] for values in duck_map.values())
        for index in category_indexes
    )
    report = {
        "python_row_count": len(python_map),
        "duckdb_row_count": len(duck_map),
        "row_count_match": len(python_map) == len(duck_map),
        "python_unique_students": len({key[2] for key in python_map}),
        "duckdb_unique_students": len({key[2] for key in duck_map}),
        "unique_student_count_match": len({key[2] for key in python_map})
        == len({key[2] for key in duck_map}),
        "python_total_clicks": sum(values[0] for values in python_map.values()),
        "duckdb_total_clicks": sum(values[0] for values in duck_map.values()),
        "total_clicks_match": sum(values[0] for values in python_map.values())
        == sum(values[0] for values in duck_map.values()),
        "daily_keys_match": keys_match,
        "deterministic_sample_size": len(deterministic_keys),
        "deterministic_samples_match": samples_match,
        "category_click_totals_match": category_match,
        "distinct_resources_match": keys_match and all(python_map[key][7] == duck_map[key][7] for key in python_map),
        "distinct_activity_types_match": keys_match and all(python_map[key][8] == duck_map[key][8] for key in python_map),
        "student_28400_day_minus_10": student_comparison,
    }
    report["all_checks_passed"] = all(
        value for key, value in report.items() if key.endswith("_match") or key == "all_checks_passed"
    ) and student_comparison["match"]
    if not report["all_checks_passed"]:
        raise ValueError("AAA 2013J parity validation failed")
    return report


def write_report(report: dict[str, object], path: Path) -> None:
    """Write the small machine-readable report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _create_source_views(connection: duckdb.DuckDBPyConnection, data_dir: Path) -> None:
    student_vle = _sql_path(data_dir / "studentVle.csv")
    vle = _sql_path(data_dir / "vle.csv")
    connection.execute(
        f"""CREATE OR REPLACE TEMP VIEW raw_student_vle AS
            SELECT code_module, code_presentation,
                   id_student::BIGINT AS id_student, id_site::BIGINT AS id_site,
                   date::INTEGER AS day, sum_click::BIGINT AS sum_click
            FROM read_csv_auto('{student_vle}', header=true)"""
    )
    connection.execute(
        f"""CREATE OR REPLACE TEMP VIEW raw_vle AS
            SELECT code_module, code_presentation, id_site::BIGINT AS id_site,
                   activity_type
            FROM read_csv_auto('{vle}', header=true, nullstr='?')"""
    )


def _daily_query(relation: str = "joined_activity") -> str:
    """Return presentation-local aggregation SQL without a global sort."""
    return f"""
        SELECT student_id, day,
               SUM(sum_click)::BIGINT AS total_clicks,
               COUNT(DISTINCT id_site)::BIGINT AS number_of_resources_visited,
               COUNT(DISTINCT activity_type)::BIGINT AS number_of_activity_types_used,
               SUM(CASE WHEN activity_type='homepage' THEN sum_click ELSE 0 END)::BIGINT AS homepage_clicks,
               SUM(CASE WHEN activity_type='forumng' THEN sum_click ELSE 0 END)::BIGINT AS forum_clicks,
               SUM(CASE WHEN activity_type='oucontent' THEN sum_click ELSE 0 END)::BIGINT AS content_clicks,
               SUM(CASE WHEN activity_type IN ('quiz','externalquiz') THEN sum_click ELSE 0 END)::BIGINT AS quiz_clicks,
               list_sort(list(DISTINCT id_site)) AS resource_ids,
               list_sort(list(DISTINCT activity_type)) AS activity_types
        FROM {relation}
        GROUP BY student_id, day
    """


def _copy_joined_partitions(connection, joined_output):
    """Stream the large join to disk partitions before distinct aggregation."""
    target = _sql_path(joined_output)
    connection.execute(
        f"""COPY (
                SELECT s.code_module AS module, s.code_presentation AS presentation,
                       s.id_student AS student_id, s.day, s.sum_click, s.id_site,
                       COALESCE(v.activity_type, 'unknown') AS activity_type
                FROM raw_student_vle s
                LEFT JOIN raw_vle v USING (code_module, code_presentation, id_site)
             ) TO '{target}'
             (FORMAT PARQUET, PARTITION_BY (module, presentation), COMPRESSION ZSTD,
              ROW_GROUP_SIZE 100000, OVERWRITE_OR_IGNORE true)"""
    )


def _copy_partitioned_daily(connection, joined_output, output_dir, logger=lambda message: None):
    """Aggregate one on-disk module presentation at a time and write Parquet."""
    partitions = sorted(path for path in joined_output.glob("module=*/presentation=*") if path.is_dir())
    if not partitions:
        raise RuntimeError("The joined activity stage produced no presentation partitions")
    for index, partition in enumerate(partitions, start=1):
        relative = partition.relative_to(joined_output)
        destination = output_dir / relative
        destination.mkdir(parents=True)
        source_glob = _sql_path(partition / "*.parquet")
        target_file = _sql_path(destination / "data_0.parquet")
        logger(f"stage=aggregation presentation={index}/{len(partitions)} path={relative.as_posix()}")
        query = _daily_query(f"read_parquet('{source_glob}')")
        logger(f"stage=parquet_writing presentation={index}/{len(partitions)} path={relative.as_posix()}")
        connection.execute(
            f"""COPY ({query}) TO '{target_file}'
                 (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"""
        )


def _raw_stats(connection):
    row = connection.execute(
        """SELECT COUNT(*), COALESCE(SUM(sum_click),0),
                  COUNT(*) FILTER (WHERE sum_click < 0)
           FROM raw_student_vle"""
    ).fetchone()
    unmatched = connection.execute(
        """SELECT COUNT(*) FROM raw_student_vle s LEFT JOIN raw_vle v
           USING (code_module, code_presentation, id_site)
           WHERE v.id_site IS NULL"""
    ).fetchone()[0]
    return {
        "raw_student_vle_row_count": int(row[0]),
        "total_clicks_before_aggregation": int(row[1]),
        "invalid_negative_click_counts": int(row[2]),
        "unmatched_site_ids": int(unmatched),
    }


def _processed_stats(connection, parquet_glob):
    relation = f"read_parquet('{parquet_glob}', hive_partitioning=true)"
    basic = connection.execute(
        f"""SELECT COUNT(*), COUNT(DISTINCT module),
                    COUNT(DISTINCT (module, presentation)), COUNT(DISTINCT student_id),
                    COALESCE(SUM(total_clicks),0),
                    COUNT(*) FILTER (WHERE total_clicks < 0 OR homepage_clicks < 0
                        OR forum_clicks < 0 OR content_clicks < 0 OR quiz_clicks < 0)
             FROM {relation}"""
    ).fetchone()
    duplicates = connection.execute(
        f"""SELECT COUNT(*) FROM (
                SELECT module, presentation, student_id, day, COUNT(*) n
                FROM {relation} GROUP BY ALL HAVING n > 1)"""
    ).fetchone()[0]
    null_counts = {}
    for column in DAILY_COLUMNS:
        null_counts[column] = int(
            connection.execute(f"SELECT COUNT(*) FILTER (WHERE {column} IS NULL) FROM {relation}").fetchone()[0]
        )
    return {
        "daily_aggregated_row_count": int(basic[0]),
        "number_of_modules": int(basic[1]),
        "number_of_presentations": int(basic[2]),
        "number_of_students": int(basic[3]),
        "total_clicks_after_aggregation": int(basic[4]),
        "processed_invalid_negative_click_counts": int(basic[5]),
        "duplicate_daily_keys": int(duplicates),
        "null_counts": null_counts,
    }


def _presentation_ranges(connection, parquet_glob):
    rows = connection.execute(
        f"""SELECT module, presentation, MIN(day), MAX(day)
             FROM read_parquet('{parquet_glob}', hive_partitioning=true)
             GROUP BY module, presentation ORDER BY module, presentation"""
    ).fetchall()
    return [
        {"module": row[0], "presentation": row[1], "minimum_day": int(row[2]), "maximum_day": int(row[3])}
        for row in rows
    ]


def _create_metadata_database(path, data_dir, output_dir, raw_stats):
    if path.exists():
        path.unlink()
    with duckdb.connect(str(path)) as connection:
        connection.execute("CREATE TABLE source_files (name VARCHAR, path VARCHAR, size_bytes BIGINT)")
        connection.executemany(
            "INSERT INTO source_files VALUES (?, ?, ?)",
            [(name, str((data_dir / name).resolve()), (data_dir / name).stat().st_size) for name in SOURCE_FILES],
        )
        connection.execute("CREATE TABLE build_summary (metric VARCHAR, value VARCHAR)")
        connection.executemany(
            "INSERT INTO build_summary VALUES (?, ?)",
            [(key, str(value)) for key, value in raw_stats.items()],
        )
        parquet_glob = _sql_path(output_dir.resolve() / "**" / "*.parquet")
        connection.execute(
            f"""CREATE VIEW daily_activity AS
                SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)"""
        )
        connection.execute(
            """CREATE TABLE presentation_day_ranges AS
               SELECT module, presentation, MIN(day) minimum_day, MAX(day) maximum_day
               FROM daily_activity GROUP BY module, presentation"""
        )


def _replace_directory(new_path, destination, backup):
    if backup.exists():
        shutil.rmtree(backup)
    if destination.exists():
        destination.rename(backup)
    try:
        new_path.rename(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.rename(destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _validate_paths(data_dir, output_dir, database_path):
    missing = [name for name in SOURCE_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing OULAD source files: {missing}")
    if output_dir.resolve() == data_dir.resolve() or data_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("Generated output must not be placed inside the source directory")
    if database_path.resolve() in [data_dir.resolve(), output_dir.resolve()]:
        raise ValueError("Invalid database target")


def _output_sizes(data_dir, output_dir, database_path):
    parquet_files = list(output_dir.rglob("*.parquet"))
    return {
        "studentVle_csv": (data_dir / "studentVle.csv").stat().st_size,
        "all_raw_source_files": sum((data_dir / name).stat().st_size for name in SOURCE_FILES),
        "parquet_dataset": sum(path.stat().st_size for path in parquet_files),
        "parquet_file_count": len(parquet_files),
        "duckdb_database": database_path.stat().st_size,
    }


def _directory_size(path: Path) -> int:
    """Return current generated disk use beneath a controlled directory."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def _generated_fingerprints(output_dir, database_path):
    paths = sorted(output_dir.rglob("*.parquet")) + [database_path]
    return {str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}


def _source_hashes(data_dir):
    return {name: _sha256(data_dir / name) for name in SOURCE_FILES}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_key(row: DailyStudentActivity):
    return (row.module, row.presentation, row.student_id, row.day)


def _summary_values(row: DailyStudentActivity):
    return (
        row.total_clicks,
        row.number_of_resources_visited,
        row.number_of_activity_types_used,
        row.homepage_clicks,
        row.forum_clicks,
        row.content_clicks,
        row.quiz_clicks,
        row.resource_ids,
        row.activity_types,
    )


def _serializable_values(values):
    if values is None:
        return None
    return [*values[:7], sorted(values[7]), sorted(values[8])]


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")
