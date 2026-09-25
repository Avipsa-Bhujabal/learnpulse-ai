"""Partition-wise, leakage-safe modeling-table construction for all OULAD courses."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from learnpulse.modeling_table import (
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_modeling_table,
    load_student_results,
    validate_model_feature_columns,
)
from learnpulse.outcomes import (
    calculate_future_inactivity_outcomes,
    load_course_lengths,
    load_withdrawal_days,
)
from learnpulse.parquet_reader import read_daily_activity
from learnpulse.personal_baseline import calculate_personal_baselines
from learnpulse.rolling_features import calculate_rolling_features

DATASET_SCHEMA_VERSION = "multicourse-modeling-v1"
IDENTIFIER_COLUMNS = ["module", "presentation", "student_id", "observation_day"]
NON_FEATURE_COLUMNS = [
    *IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
    "future_window_start_day", "future_window_end_day", "future_activity_days",
    "future_clicks", "official_withdrawal_day",
    "officially_withdrawn_by_observation_day",
    "officially_withdrawn_by_window_end", "final_result", "outcome_eligible",
    "prediction_eligible", "exclusion_reason", "source_partition",
    "dataset_schema_version",
]
AUDIT_ONLY_COLUMNS = NON_FEATURE_COLUMNS[5:]
OUTPUT_COLUMNS = IDENTIFIER_COLUMNS + list(MODEL_FEATURE_COLUMNS) + [
    TARGET_COLUMN,
    "future_window_start_day", "future_window_end_day", "future_activity_days",
    "future_clicks", "official_withdrawal_day",
    "officially_withdrawn_by_observation_day",
    "officially_withdrawn_by_window_end", "final_result", "outcome_eligible",
    "prediction_eligible", "exclusion_reason", "source_partition",
    "dataset_schema_version",
]
SOURCE_METADATA_FILES = ("courses.csv", "studentRegistration.csv", "studentInfo.csv")


@dataclass(frozen=True)
class PartitionIdentity:
    """One discovered Hive partition and the files that establish its identity."""

    module: str
    presentation: str
    files: tuple[Path, ...]

    @property
    def name(self) -> str:
        return f"module={self.module}/presentation={self.presentation}"


@dataclass(frozen=True)
class MultiCourseBuildResult:
    """Published paths and the complete machine-readable build report."""

    output_parquet: Path
    output_csv: Path
    report_path: Path
    report: dict[str, object]


def discover_partitions(
    dataset: Path, module: str | None = None, presentation: str | None = None
) -> list[PartitionIdentity]:
    """Discover actual Hive partitions, optionally applying exact-name filters."""
    if not dataset.is_dir():
        raise FileNotFoundError(f"Daily Parquet dataset does not exist: {dataset}")
    found: list[PartitionIdentity] = []
    for module_dir in sorted(dataset.iterdir()):
        if not module_dir.is_dir() or not module_dir.name.startswith("module="):
            continue
        module_value = module_dir.name.removeprefix("module=")
        if module is not None and module_value != module:
            continue
        for presentation_dir in sorted(module_dir.iterdir()):
            if not presentation_dir.is_dir() or not presentation_dir.name.startswith("presentation="):
                continue
            presentation_value = presentation_dir.name.removeprefix("presentation=")
            if presentation is not None and presentation_value != presentation:
                continue
            files = tuple(sorted(presentation_dir.glob("*.parquet")))
            if files:
                found.append(PartitionIdentity(module_value, presentation_value, files))
    if not found:
        raise ValueError("No source partitions matched the requested filters")
    return found


def configuration_fingerprint(
    dataset: Path,
    metadata_dir: Path,
    observation_days: Sequence[int],
    future_window_days: int,
    minimum_history: int,
    partitions: Sequence[PartitionIdentity],
) -> str:
    """Hash methodology and file metadata (not full large-file contents)."""
    payload = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "daily_dataset": str(dataset.resolve()),
        "observation_days": list(observation_days),
        "future_window_days": future_window_days,
        "minimum_history": minimum_history,
        "partitions": [_partition_signature(item) for item in partitions],
        "metadata": {
            name: _file_signature(metadata_dir / name) for name in SOURCE_METADATA_FILES
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_multicourse_modeling_table(
    daily_dataset: Path,
    metadata_dir: Path,
    output_parquet: Path,
    output_csv: Path,
    report_path: Path,
    observation_days: Sequence[int] = (14, 28, 42, 56),
    future_window_days: int = 14,
    minimum_history: int = 7,
    module: str | None = None,
    presentation: str | None = None,
    resume: bool = False,
    temporary_dir: Path | None = None,
    logger: Callable[[str], None] | None = None,
) -> MultiCourseBuildResult:
    """Build each presentation independently and publish validated outputs atomically."""
    log = logger or (lambda message: None)
    days = _validate_arguments(observation_days, future_window_days, minimum_history)
    _validate_paths(daily_dataset, metadata_dir, output_parquet, output_csv, report_path)
    validate_model_feature_columns(MODEL_FEATURE_COLUMNS)
    if set(MODEL_FEATURE_COLUMNS) & set(NON_FEATURE_COLUMNS):
        raise ValueError("Feature and non-feature schemas overlap")
    partitions = discover_partitions(daily_dataset, module, presentation)
    fingerprint = configuration_fingerprint(
        daily_dataset, metadata_dir, days, future_window_days, minimum_history, partitions
    )
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    source_before = _source_signatures(daily_dataset, metadata_dir)
    controlled = temporary_dir or output_parquet.parent / ".multicourse_temp"
    work = controlled / f"build_{fingerprint[:16]}"
    if work.exists() and not resume:
        shutil.rmtree(work)
    partitions_dir = work / "partitions"
    partitions_dir.mkdir(parents=True, exist_ok=True)

    course_lengths = load_course_lengths(metadata_dir / "courses.csv")
    withdrawals = load_withdrawal_days(metadata_dir / "studentRegistration.csv")
    final_results = load_student_results(metadata_dir / "studentInfo.csv")
    students_by_partition: dict[tuple[str, str], list[int]] = defaultdict(list)
    for module_name, presentation_name, student_id in final_results:
        students_by_partition[(module_name, presentation_name)].append(student_id)

    completed: list[str] = []
    reused: list[str] = []
    failed: list[dict[str, str]] = []
    manifests: list[dict[str, object]] = []
    try:
        for index, partition in enumerate(partitions, 1):
            slug = f"{partition.module}_{partition.presentation}"
            part_output = partitions_dir / f"{slug}.parquet"
            manifest_path = partitions_dir / f"{slug}.manifest.json"
            signature = _partition_signature(partition)
            manifest = _load_manifest(manifest_path)
            if resume and _manifest_compatible(
                manifest, fingerprint, signature, part_output
            ):
                log(f"stage=partition_reuse partition={partition.name}")
                reused.append(partition.name)
                manifests.append(manifest)
                continue
            log(f"stage=partition_build partition={partition.name} index={index}/{len(partitions)}")
            try:
                if part_output.exists():
                    part_output.unlink()
                if manifest_path.exists():
                    manifest_path.unlink()
                manifest = _build_partition(
                    daily_dataset=daily_dataset,
                    partition=partition,
                    student_ids=students_by_partition.get(
                        (partition.module, partition.presentation), []
                    ),
                    course_lengths=course_lengths,
                    withdrawals=withdrawals,
                    final_results=final_results,
                    observation_days=days,
                    future_window_days=future_window_days,
                    minimum_history=minimum_history,
                    output=part_output,
                    fingerprint=fingerprint,
                    source_signature=signature,
                )
                _write_json_atomic(manifest_path, manifest)
                completed.append(partition.name)
                manifests.append(manifest)
            except Exception as error:
                failed.append({"partition": partition.name, "error": str(error)})
                if part_output.exists():
                    part_output.unlink()
                raise

        log("stage=combine_and_validate")
        staging = work / "publish"
        staging.mkdir(exist_ok=True)
        staged_parquet = staging / output_parquet.name
        staged_csv = staging / output_csv.name
        part_files = [partitions_dir / f"{p.module}_{p.presentation}.parquet" for p in partitions]
        consistency = _combine_and_validate(
            part_files, staged_parquet, staged_csv, days,
            require_both_classes=module is None and presentation is None,
        )
        parity = _validate_aaa_parity(staged_parquet, output_parquet.parent / "modeling_table.csv")
        source_after = _source_signatures(daily_dataset, metadata_dir)
        if source_before != source_after:
            raise RuntimeError("Source Parquet or metadata files changed during the build")

        report = _create_report(
            staged_parquet, manifests, partitions, days, future_window_days,
            minimum_history, module, presentation, fingerprint, started_at,
            time.perf_counter() - started, completed, reused, failed,
            consistency, parity, source_before == source_after,
        )
        staged_report = staging / report_path.name
        _write_json_atomic(staged_report, report)
        log("stage=atomic_publish")
        _publish_files(
            [(staged_parquet, output_parquet), (staged_csv, output_csv), (staged_report, report_path)]
        )
        shutil.rmtree(work)
        if controlled.exists() and not any(controlled.iterdir()):
            controlled.rmdir()
        return MultiCourseBuildResult(output_parquet, output_csv, report_path, report)
    except Exception:
        # Valid final outputs are untouched. Completed manifests remain for --resume.
        raise


def _build_partition(
    daily_dataset: Path,
    partition: PartitionIdentity,
    student_ids: Sequence[int],
    course_lengths: dict[tuple[str, str], int],
    withdrawals: dict[tuple[str, str, int], int | None],
    final_results: dict[tuple[str, str, int], str],
    observation_days: Sequence[int],
    future_window_days: int,
    minimum_history: int,
    output: Path,
    fingerprint: str,
    source_signature: dict[str, object],
) -> dict[str, object]:
    daily = read_daily_activity(daily_dataset, partition.module, partition.presentation)
    rolling = calculate_rolling_features(daily, start_day=0, end_day=max(observation_days))
    baselines = calculate_personal_baselines(rolling, minimum_history)
    observations = [
        (partition.module, partition.presentation, student_id, day)
        for student_id in sorted(student_ids)
        for day in observation_days
    ]
    outcomes = calculate_future_inactivity_outcomes(
        daily, observations, course_lengths, withdrawals, future_window_days
    )
    result = build_modeling_table(
        rolling, baselines, outcomes, final_results, observation_days
    )
    rows = [_multicourse_row(row, partition.name) for row in result.model_ready_rows]
    _write_rows_parquet(output, rows)
    exclusions: Counter[str] = Counter()
    for row in result.excluded_rows:
        exclusions.update(str(row["exclusion_reasons"]).split("|"))
    targets = Counter(bool(row[TARGET_COLUMN]) for row in rows)
    return {
        "complete": True,
        "schema_version": DATASET_SCHEMA_VERSION,
        "configuration_fingerprint": fingerprint,
        "source_partition": partition.name,
        "source_signature": source_signature,
        "output_signature": _file_signature(output),
        "candidate_rows": len(result.all_rows),
        "model_ready_rows": len(rows),
        "unique_students": len({int(row["student_id"]) for row in rows}),
        "target_positive": targets[True],
        "target_negative": targets[False],
        "exclusions": dict(sorted(exclusions.items())),
    }


def _multicourse_row(row: dict[str, object], source_partition: str) -> dict[str, object]:
    return {
        **{name: row.get(name) for name in IDENTIFIER_COLUMNS + list(MODEL_FEATURE_COLUMNS)},
        TARGET_COLUMN: row[TARGET_COLUMN],
        "future_window_start_day": row["future_window_start_day"],
        "future_window_end_day": row["future_window_end_day"],
        "future_activity_days": row["future_activity_days"],
        "future_clicks": row["future_clicks"],
        "official_withdrawal_day": row["official_withdrawal_day"],
        "officially_withdrawn_by_observation_day": row["officially_withdrawn_by_observation_day"],
        "officially_withdrawn_by_window_end": row["officially_withdrawn_by_window_end"],
        "final_result": row["final_result"],
        "outcome_eligible": row["outcome_eligible"],
        "prediction_eligible": row["prediction_eligible"],
        "exclusion_reason": row["exclusion_reasons"],
        "source_partition": source_partition,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
    }


_SQL_TYPES = {
    **{name: "VARCHAR" for name in ["module", "presentation", "final_result", "exclusion_reason", "source_partition", "dataset_schema_version"]},
    **{name: "BIGINT" for name in ["student_id", "observation_day", "clicks_last_7_days", "active_days_last_7_days", "resources_last_7_days", "activity_types_last_7_days", "homepage_clicks_last_7_days", "forum_clicks_last_7_days", "content_clicks_last_7_days", "quiz_clicks_last_7_days", "days_since_last_activity", "historical_windows_available", "current_inactivity_gap", "longest_previous_inactivity_gap", "future_window_start_day", "future_window_end_day", "future_activity_days", "future_clicks", "official_withdrawal_day"]},
    **{name: "DOUBLE" for name in ["historical_average_clicks", "historical_standard_deviation_clicks", "click_difference_from_personal_average", "click_percentage_change_from_personal_average", "click_z_score", "historical_average_active_days", "active_days_difference_from_personal_average"]},
    **{name: "BOOLEAN" for name in [TARGET_COLUMN, "officially_withdrawn_by_observation_day", "officially_withdrawn_by_window_end", "outcome_eligible", "prediction_eligible"]},
}


def _write_rows_parquet(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".writing.parquet")
    if temporary.exists():
        temporary.unlink()
    columns_sql = ", ".join(f'"{name}" {_SQL_TYPES[name]}' for name in OUTPUT_COLUMNS)
    placeholders = ",".join("?" for _ in OUTPUT_COLUMNS)
    with duckdb.connect() as connection:
        connection.execute(f"CREATE TABLE result ({columns_sql})")
        if rows:
            connection.executemany(
                f"INSERT INTO result VALUES ({placeholders})",
                [[row.get(name) for name in OUTPUT_COLUMNS] for row in rows],
            )
        connection.execute("COPY result TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(temporary.resolve())])
    os.replace(temporary, path)


def _combine_and_validate(
    partition_files_: Sequence[Path], staged_parquet: Path, staged_csv: Path,
    observation_days: Sequence[int], require_both_classes: bool = True,
) -> dict[str, object]:
    files = [str(path.resolve()) for path in partition_files_]
    order = "module, presentation, student_id, observation_day"
    parquet_target = _sql_literal_path(staged_parquet)
    csv_target = _sql_literal_path(staged_csv)
    with duckdb.connect() as connection:
        connection.execute(
            f"COPY (SELECT * FROM read_parquet(?) ORDER BY {order}) TO '{parquet_target}' (FORMAT PARQUET, COMPRESSION ZSTD)",
            [files],
        )
        connection.execute(
            f"COPY (SELECT * FROM read_parquet(?) ORDER BY {order}) TO '{csv_target}' (FORMAT CSV, HEADER, NULL '')",
            [files],
        )
        source = "read_parquet(?)"
        row_count = int(connection.execute(f"SELECT count(*) FROM {source}", [files]).fetchone()[0])
        duplicate_count = int(connection.execute(
            f"SELECT count(*)-count(DISTINCT (module,presentation,student_id,observation_day)) FROM {source}", [files]
        ).fetchone()[0])
        invalid = int(connection.execute(
            f"SELECT count(*) FROM {source} WHERE outcome_eligible IS NOT TRUE OR prediction_eligible IS NOT TRUE OR officially_withdrawn_by_observation_day IS TRUE OR future_inactivity IS NULL OR future_window_start_day <> observation_day+1 OR future_window_end_day < future_window_start_day", [files]
        ).fetchone()[0])
        negative = int(connection.execute(
            f"SELECT count(*) FROM {source} WHERE clicks_last_7_days<0 OR active_days_last_7_days<0 OR resources_last_7_days<0 OR activity_types_last_7_days<0 OR future_activity_days<0 OR future_clicks<0", [files]
        ).fetchone()[0])
        actual_days = [int(row[0]) for row in connection.execute(
            f"SELECT DISTINCT observation_day FROM {source} ORDER BY 1", [files]
        ).fetchall()]
        classes = int(connection.execute(
            f"SELECT count(DISTINCT future_inactivity) FROM {source}", [files]
        ).fetchone()[0])
        parquet_keys = connection.execute(
            f"SELECT module,presentation,student_id,observation_day,future_inactivity FROM read_parquet(?) ORDER BY {order}", [str(staged_parquet.resolve())]
        ).fetchall()
        csv_keys = connection.execute(
            f"SELECT module,presentation,student_id,observation_day,future_inactivity FROM read_csv_auto(?, header=true) ORDER BY {order}", [str(staged_csv.resolve())]
        ).fetchall()
    if duplicate_count:
        raise ValueError(f"Duplicate modeling keys detected: {duplicate_count}")
    if invalid:
        raise ValueError(f"Ineligible or impossible rows detected: {invalid}")
    if negative:
        raise ValueError(f"Negative activity values detected: {negative}")
    if not set(actual_days).issubset(observation_days):
        raise ValueError(f"Output contains unrequested observation days: {actual_days}")
    if require_both_classes and classes != 2:
        raise ValueError("Complete dataset must contain both target classes")
    if parquet_keys != csv_keys:
        raise ValueError("CSV and Parquet keys or targets differ")
    return {
        "row_count_match": len(parquet_keys) == row_count == len(csv_keys),
        "keys_match": parquet_keys == csv_keys,
        "target_totals_match": Counter(row[4] for row in parquet_keys) == Counter(row[4] for row in csv_keys),
        "duplicate_key_count": duplicate_count,
        "negative_activity_row_count": negative,
        "invalid_eligibility_row_count": invalid,
    }


def _create_report(
    parquet_path: Path,
    manifests: Sequence[dict[str, object]],
    partitions: Sequence[PartitionIdentity],
    days: Sequence[int],
    future_window_days: int,
    minimum_history: int,
    module_filter: str | None,
    presentation_filter: str | None,
    fingerprint: str,
    started_at: datetime,
    elapsed: float,
    completed: Sequence[str],
    reused: Sequence[str],
    failed: Sequence[dict[str, str]],
    consistency: dict[str, object],
    parity: dict[str, object],
    source_unchanged: bool,
) -> dict[str, object]:
    exclusions: Counter[str] = Counter()
    for manifest in manifests:
        exclusions.update({str(k): int(v) for k, v in dict(manifest["exclusions"]).items()})
    with duckdb.connect() as connection:
        path = str(parquet_path.resolve())
        rows = int(connection.execute("SELECT count(*) FROM read_parquet(?)", [path]).fetchone()[0])
        students = int(connection.execute("SELECT count(DISTINCT student_id) FROM read_parquet(?)", [path]).fetchone()[0])
        by_day_rows = connection.execute(
            "SELECT observation_day,count(*),sum(future_inactivity::INT),avg(future_inactivity::INT) FROM read_parquet(?) GROUP BY 1 ORDER BY 1", [path]
        ).fetchall()
        by_partition_rows = connection.execute(
            "SELECT module,presentation,count(*),count(DISTINCT student_id),sum(future_inactivity::INT),avg(future_inactivity::INT) FROM read_parquet(?) GROUP BY 1,2 ORDER BY 1,2", [path]
        ).fetchall()
        missing_rows = connection.execute(
            "SELECT " + ",".join(f'count(*) FILTER (WHERE "{name}" IS NULL)' for name in OUTPUT_COLUMNS) + " FROM read_parquet(?)", [path]
        ).fetchone()
    by_day = {
        str(day): {"rows": int(count), "positive": int(positive), "negative": int(count-positive), "prevalence": float(prevalence)}
        for day, count, positive, prevalence in by_day_rows
    }
    by_partition = {
        f"{module} {presentation}": {
            "rows": int(count), "students": int(student_count),
            "positive": int(positive), "negative": int(count-positive),
            "prevalence": float(prevalence),
            "training_suitability": "both_classes" if 0 < int(positive) < int(count) else "single_class_descriptive_only",
        }
        for module, presentation, count, student_count, positive, prevalence in by_partition_rows
    }
    warnings = [
        f"{name} has only one target class and is unsuitable for standalone model training"
        for name, stats in by_partition.items()
        if stats["training_suitability"] != "both_classes"
    ]
    warnings.extend(
        f"{name} has only {stats['positive']} positive rows"
        for name, stats in by_partition.items()
        if 0 < int(stats["positive"]) < 20
    )
    return {
        "build_configuration": {
            "daily_dataset": str(partitions[0].files[0].parents[2].resolve()),
            "observation_days": list(days), "future_window_days": future_window_days,
            "minimum_history": minimum_history, "module_filter": module_filter,
            "presentation_filter": presentation_filter,
            "filtered_build": module_filter is not None or presentation_filter is not None,
        },
        "configuration_fingerprint": fingerprint,
        "fingerprint_method": "SHA-256 of methodology, resolved source path, schema version, and file size/mtime metadata; not full dataset content hashing",
        "schema_version": DATASET_SCHEMA_VERSION,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": elapsed,
        "modules_processed": sorted({p.module for p in partitions}),
        "presentations_processed": sorted({f"{p.module} {p.presentation}" for p in partitions}),
        "source_partitions_processed": [p.name for p in partitions],
        "rows_before_eligibility_filtering": sum(int(m["candidate_rows"]) for m in manifests),
        "model_ready_rows": rows,
        "unique_students": students,
        "rows_by_observation_day": by_day,
        "rows_by_module_presentation": by_partition,
        "students_by_module_presentation": {name: stats["students"] for name, stats in by_partition.items()},
        "target_counts_by_module_presentation": {name: {"positive": stats["positive"], "negative": stats["negative"]} for name, stats in by_partition.items()},
        "exclusion_counts_by_reason": dict(sorted(exclusions.items())),
        "missing_values_by_field": dict(zip(OUTPUT_COLUMNS, (int(value) for value in missing_rows))),
        "duplicate_key_count": consistency["duplicate_key_count"],
        "aaa_2013j_parity": parity,
        "csv_parquet_consistency": consistency,
        "source_integrity": {"unchanged": source_unchanged},
        "completed_partitions": list(completed), "reused_partitions": list(reused),
        "failed_partitions": list(failed), "warnings": warnings,
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "non_feature_columns": NON_FEATURE_COLUMNS,
    }


def _validate_aaa_parity(new_parquet: Path, trusted_csv: Path) -> dict[str, object]:
    """Compare shared AAA 2013J keys against the trusted Step 5 table."""
    if not trusted_csv.is_file():
        project_trusted = Path("data/processed/modeling_table.csv")
        if project_trusted.is_file():
            trusted_csv = project_trusted
    if not trusted_csv.is_file():
        return {"performed": False, "reason": f"Trusted table unavailable: {trusted_csv}"}
    compare_columns = IDENTIFIER_COLUMNS + list(MODEL_FEATURE_COLUMNS) + [
        TARGET_COLUMN, "future_window_start_day", "future_window_end_day",
        "future_activity_days", "future_clicks", "official_withdrawal_day",
        "officially_withdrawn_by_observation_day", "officially_withdrawn_by_window_end",
        "final_result", "outcome_eligible", "prediction_eligible",
    ]
    with duckdb.connect() as connection:
        trusted_description = [row[0] for row in connection.execute(
            "DESCRIBE SELECT * FROM read_csv_auto(?, header=true)", [str(trusted_csv.resolve())]
        ).fetchall()]
        if not set(compare_columns).issubset(trusted_description):
            return {"performed": False, "reason": "Trusted table schema differs from required shared fields"}
        selected = ",".join(f'"{name}"' for name in compare_columns)
        trusted_rows = connection.execute(
            f"SELECT {selected} FROM read_csv_auto(?, header=true) WHERE module='AAA' AND presentation='2013J' ORDER BY student_id,observation_day",
            [str(trusted_csv.resolve())],
        ).fetchall()
        new_selected = connection.execute(
            f"SELECT {selected} FROM read_parquet(?) WHERE module='AAA' AND presentation='2013J' ORDER BY student_id,observation_day",
            [str(new_parquet.resolve())],
        ).fetchall()
    mismatches: Counter[str] = Counter()
    if len(new_selected) == len(trusted_rows):
        for new, old in zip(new_selected, trusted_rows):
            for name, left, right in zip(compare_columns, new, old):
                if isinstance(left, float) or isinstance(right, float):
                    equal = left is None and right is None or left is not None and right is not None and abs(float(left)-float(right)) <= 1e-9
                else:
                    equal = left == right
                if not equal:
                    mismatches[name] += 1
    return {
        "performed": True, "trusted_rows": len(trusted_rows), "new_rows": len(new_selected),
        "row_count_match": len(new_selected) == len(trusted_rows),
        "field_mismatch_counts": dict(mismatches),
        "passed": len(new_selected) == len(trusted_rows) and not mismatches,
        "float_tolerance": 1e-9,
    }


def _publish_files(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Replace a set of generated files with rollback if any replace fails."""
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for _, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                backup = target.with_name(f".{target.name}.previous")
                if backup.exists():
                    backup.unlink()
                os.replace(target, backup)
                backups.append((backup, target))
        for source, target in pairs:
            os.replace(source, target)
            published.append(target)
    except Exception:
        for target in published:
            if target.exists():
                target.unlink()
        for backup, target in backups:
            if backup.exists():
                os.replace(backup, target)
        raise
    for backup, _ in backups:
        if backup.exists():
            backup.unlink()


def _validate_arguments(days: Sequence[int], future_days: int, minimum_history: int) -> tuple[int, ...]:
    ordered = tuple(sorted(days))
    if not ordered or len(ordered) != len(set(ordered)):
        raise ValueError("observation_days must be non-empty and contain no duplicates")
    if any(isinstance(day, bool) for day in ordered):
        raise ValueError("observation_days must be integers")
    if future_days <= 0:
        raise ValueError("future_window_days must be greater than zero")
    if minimum_history < 0:
        raise ValueError("minimum_history cannot be negative")
    return ordered


def _validate_paths(dataset: Path, metadata: Path, parquet: Path, csv_path: Path, report: Path) -> None:
    if not dataset.is_dir():
        raise FileNotFoundError(f"Daily dataset does not exist: {dataset}")
    missing = [str(metadata / name) for name in SOURCE_METADATA_FILES if not (metadata / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing metadata files: {missing}")
    targets = [parquet.resolve(), csv_path.resolve(), report.resolve()]
    if len(set(targets)) != 3:
        raise ValueError("Output Parquet, CSV, and report paths must be different")
    for target in targets:
        if dataset.resolve() == target or dataset.resolve() in target.parents:
            raise ValueError("Generated outputs cannot be written inside the source dataset")
        if metadata.resolve() == target or metadata.resolve() in target.parents:
            raise ValueError("Generated outputs cannot be written inside metadata sources")


def _partition_signature(partition: PartitionIdentity) -> dict[str, object]:
    return {"name": partition.name, "files": [_file_signature(path) for path in partition.files]}


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_signatures(dataset: Path, metadata: Path) -> dict[str, object]:
    return {
        "partitions": [_file_signature(path) for path in sorted(dataset.rglob("*.parquet"))],
        "metadata": {name: _file_signature(metadata / name) for name in SOURCE_METADATA_FILES},
    }


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _manifest_compatible(
    manifest: dict[str, object], fingerprint: str,
    source_signature: dict[str, object], output: Path,
) -> bool:
    return (
        manifest.get("complete") is True
        and manifest.get("schema_version") == DATASET_SCHEMA_VERSION
        and manifest.get("configuration_fingerprint") == fingerprint
        and manifest.get("source_signature") == source_signature
        and output.is_file()
        and manifest.get("output_signature") == _file_signature(output)
        and _parquet_has_schema(output)
    )


def _parquet_has_schema(path: Path) -> bool:
    try:
        with duckdb.connect() as connection:
            names = [row[0] for row in connection.execute("DESCRIBE SELECT * FROM read_parquet(?)", [str(path.resolve())]).fetchall()]
        return names == OUTPUT_COLUMNS
    except duckdb.Error:
        return False


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sql_literal_path(path: Path) -> str:
    """Escape a controlled path used where DuckDB COPY cannot bind targets."""
    return path.resolve().as_posix().replace("'", "''")
