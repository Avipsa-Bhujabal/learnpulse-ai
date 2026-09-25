"""Tests for the partition-wise multi-course modeling-table builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from learnpulse.modeling_table import MODEL_FEATURE_COLUMNS
from learnpulse.multicourse_modeling_table import (
    DATASET_SCHEMA_VERSION,
    IDENTIFIER_COLUMNS,
    NON_FEATURE_COLUMNS,
    _manifest_compatible,
    _partition_signature,
    build_multicourse_modeling_table,
    configuration_fingerprint,
    discover_partitions,
)


def _fixture(root: Path) -> tuple[Path, Path]:
    dataset = root / "daily"
    metadata = root / "raw"
    dataset.mkdir(); metadata.mkdir()
    rows = []
    for module, presentation in (("AAA", "2013J"), ("BBB", "2014B")):
        # Student 1 remains active; student 2 becomes inactive after Day 7.
        for day in range(0, 72, 2):
            rows.append((module, presentation, 1, day, 3, 1, 1, 3, 0, 0, 0, [10], ["homepage"]))
        for day in range(0, 8, 2):
            rows.append((module, presentation, 2, day, 2, 1, 1, 0, 2, 0, 0, [11], ["forumng"]))
    with duckdb.connect() as connection:
        connection.execute("""CREATE TABLE daily(module VARCHAR,presentation VARCHAR,
            student_id BIGINT,day BIGINT,total_clicks BIGINT,
            number_of_resources_visited BIGINT,number_of_activity_types_used BIGINT,
            homepage_clicks BIGINT,forum_clicks BIGINT,content_clicks BIGINT,
            quiz_clicks BIGINT,resource_ids BIGINT[],activity_types VARCHAR[])""")
        connection.executemany("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        connection.execute("COPY daily TO ? (FORMAT PARQUET, PARTITION_BY(module,presentation))", [str(dataset.resolve())])
    (metadata / "courses.csv").write_text(
        "code_module,code_presentation,module_presentation_length\nAAA,2013J,100\nBBB,2014B,100\n", encoding="utf-8"
    )
    registration = ["code_module,code_presentation,id_student,date_registration,date_unregistration"]
    info = ["code_module,code_presentation,id_student,final_result"]
    for module, presentation in (("AAA", "2013J"), ("BBB", "2014B")):
        for student in (1, 2):
            registration.append(f"{module},{presentation},{student},-10,?")
            info.append(f"{module},{presentation},{student},Pass")
    (metadata / "studentRegistration.csv").write_text("\n".join(registration)+"\n", encoding="utf-8")
    (metadata / "studentInfo.csv").write_text("\n".join(info)+"\n", encoding="utf-8")
    return dataset, metadata


class MultiCourseModelingTableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset, self.metadata = _fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _build(self, **kwargs):
        return build_multicourse_modeling_table(
            self.dataset, self.metadata, self.root / "out.parquet",
            self.root / "out.csv", self.root / "report.json",
            observation_days=kwargs.pop("observation_days", (14, 28, 42, 56)),
            temporary_dir=self.root / "temporary", **kwargs,
        )

    def test_automatic_partition_discovery_and_isolation(self):
        found = discover_partitions(self.dataset)
        self.assertEqual([(p.module, p.presentation) for p in found], [("AAA", "2013J"), ("BBB", "2014B")])

    def test_module_and_presentation_filtering(self):
        self.assertEqual(len(discover_partitions(self.dataset, "AAA", "2013J")), 1)
        self.assertEqual(discover_partitions(self.dataset, presentation="2014B")[0].module, "BBB")

    def test_build_has_requested_days_unique_keys_and_order(self):
        result = self._build()
        with duckdb.connect() as connection:
            rows = connection.execute("SELECT module,presentation,student_id,observation_day FROM read_parquet(?)", [str(result.output_parquet)]).fetchall()
        self.assertEqual(rows, sorted(rows))
        self.assertEqual({row[3] for row in rows}, {14, 28, 42, 56})
        self.assertEqual(len(rows), len(set(rows)))

    def test_chronology_baseline_and_future_boundaries(self):
        self._build(module="AAA", presentation="2013J")
        with duckdb.connect() as connection:
            row = connection.execute("SELECT historical_windows_available,future_window_start_day,future_window_end_day FROM read_parquet(?) WHERE observation_day=14 LIMIT 1", [str(self.root / "out.parquet")]).fetchone()
        self.assertEqual(row, (8, 15, 28))

    def test_target_classes_course_end_and_withdrawal_eligibility(self):
        result = self._build(module="AAA", presentation="2013J")
        report = result.report
        self.assertEqual(report["model_ready_rows"], 8)
        self.assertEqual(set(report["rows_by_observation_day"]), {"14", "28", "42", "56"})
        self.assertEqual(report["csv_parquet_consistency"]["invalid_eligibility_row_count"], 0)

    def test_withdrawal_by_observation_is_excluded_by_existing_rules(self):
        path = self.metadata / "studentRegistration.csv"
        text = path.read_text(encoding="utf-8").replace("AAA,2013J,2,-10,?", "AAA,2013J,2,-10,28")
        path.write_text(text, encoding="utf-8")
        result = self._build(module="AAA", presentation="2013J")
        self.assertEqual(result.report["model_ready_rows"], 5)
        self.assertEqual(result.report["exclusion_counts_by_reason"]["withdrawn_by_observation_day"], 3)

    def test_course_end_ineligible_rows_are_reported_not_published(self):
        path = self.metadata / "courses.csv"
        path.write_text("code_module,code_presentation,module_presentation_length\nAAA,2013J,65\nBBB,2014B,100\n", encoding="utf-8")
        result = self._build(module="AAA", presentation="2013J")
        self.assertEqual(result.report["exclusion_counts_by_reason"]["future_window_exceeds_course_end"], 2)
        self.assertNotIn("56", result.report["rows_by_observation_day"])

    def test_feature_schema_excludes_identifiers_target_and_audit(self):
        self.assertTrue(set(MODEL_FEATURE_COLUMNS).isdisjoint(IDENTIFIER_COLUMNS))
        self.assertTrue(set(MODEL_FEATURE_COLUMNS).isdisjoint(NON_FEATURE_COLUMNS))
        self.assertNotIn("future_inactivity", MODEL_FEATURE_COLUMNS)

    def test_missing_statistical_values_are_not_coerced(self):
        # The writer accepts and preserves scientifically meaningful nulls.
        result = self._build(minimum_history=0, observation_days=(6,))
        self.assertGreater(result.report["missing_values_by_field"]["click_z_score"], 0)

    def test_csv_and_parquet_consistency(self):
        result = self._build()
        self.assertEqual(result.report["csv_parquet_consistency"], {
            "row_count_match": True, "keys_match": True, "target_totals_match": True,
            "duplicate_key_count": 0, "negative_activity_row_count": 0,
            "invalid_eligibility_row_count": 0,
        })

    def test_repeated_build_is_deterministic(self):
        first = self._build().report
        first_bytes = (self.root / "out.csv").read_bytes()
        second = self._build().report
        self.assertEqual(first["configuration_fingerprint"], second["configuration_fingerprint"])
        self.assertEqual(first_bytes, (self.root / "out.csv").read_bytes())

    def test_configuration_fingerprint_changes_with_methodology(self):
        parts = discover_partitions(self.dataset)
        first = configuration_fingerprint(self.dataset, self.metadata, (14, 28), 14, 7, parts)
        second = configuration_fingerprint(self.dataset, self.metadata, (14, 28), 7, 7, parts)
        self.assertNotEqual(first, second)

    def test_compatible_manifest_requires_matching_identity_and_schema(self):
        part = discover_partitions(self.dataset)[0]
        output = self.root / "part.parquet"
        self._build(module="AAA", presentation="2013J")
        output.write_bytes((self.root / "out.parquet").read_bytes())
        manifest = {"complete": True, "schema_version": DATASET_SCHEMA_VERSION,
                    "configuration_fingerprint": "abc", "source_signature": _partition_signature(part),
                    "output_signature": {"path": str(output.resolve()), "size": output.stat().st_size, "mtime_ns": output.stat().st_mtime_ns}}
        self.assertTrue(_manifest_compatible(manifest, "abc", _partition_signature(part), output))
        manifest["configuration_fingerprint"] = "wrong"
        self.assertFalse(_manifest_compatible(manifest, "abc", _partition_signature(part), output))

    def test_success_cleans_controlled_temporary_output(self):
        self._build()
        self.assertFalse((self.root / "temporary").exists())

    def test_failed_build_preserves_previous_outputs(self):
        (self.root / "out.parquet").write_bytes(b"old parquet")
        (self.root / "out.csv").write_text("old csv", encoding="utf-8")
        (self.root / "report.json").write_text("old report", encoding="utf-8")
        with patch("learnpulse.multicourse_modeling_table._combine_and_validate", side_effect=RuntimeError("forced")):
            with self.assertRaises(RuntimeError):
                self._build()
        self.assertEqual((self.root / "out.parquet").read_bytes(), b"old parquet")
        self.assertEqual((self.root / "out.csv").read_text(encoding="utf-8"), "old csv")

    def test_resume_reuses_compatible_completed_partitions(self):
        with patch("learnpulse.multicourse_modeling_table._combine_and_validate", side_effect=RuntimeError("forced")):
            with self.assertRaises(RuntimeError):
                self._build()
        result = self._build(resume=True)
        self.assertEqual(len(result.report["reused_partitions"]), 2)

    def test_resume_rejects_incompatible_manifest(self):
        with patch("learnpulse.multicourse_modeling_table._combine_and_validate", side_effect=RuntimeError("forced")):
            with self.assertRaises(RuntimeError): self._build()
        manifest = next((self.root / "temporary").rglob("*.manifest.json"))
        data = json.loads(manifest.read_text(encoding="utf-8")); data["schema_version"] = "old"
        manifest.write_text(json.dumps(data), encoding="utf-8")
        result = self._build(resume=True)
        self.assertEqual(len(result.report["completed_partitions"]), 1)
        self.assertEqual(len(result.report["reused_partitions"]), 1)

    def test_source_files_are_not_modified(self):
        paths = list(self.dataset.rglob("*.parquet")) + list(self.metadata.glob("*.csv"))
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
        self._build()
        self.assertEqual(before, {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths})

    def test_real_aaa_parity_when_available(self):
        daily = Path("data/processed/daily_activity"); raw = Path("data/raw/oulad")
        trusted = Path("data/processed/modeling_table.csv")
        if not daily.is_dir() or not trusted.is_file() or not (raw / "courses.csv").is_file():
            self.skipTest("Real processed OULAD and trusted modeling table are unavailable")
        root = self.root / "real"; root.mkdir()
        result = build_multicourse_modeling_table(daily, raw, root / "x.parquet", root / "x.csv", root / "x.json", module="AAA", presentation="2013J", temporary_dir=root / "tmp")
        self.assertTrue(result.report["aaa_2013j_parity"]["passed"])


if __name__ == "__main__":
    unittest.main()
