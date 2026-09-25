"""Miniature-data tests for the DuckDB/Parquet preparation layer."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import time
import unittest
from pathlib import Path

import duckdb

from learnpulse.scalable_pipeline import (
    DuckDBConfig,
    _copy_partitioned_daily,
    _processed_stats,
    _sql_path,
    build_scalable_pipeline,
    configure_duckdb,
    validate_duckdb_config,
    validate_scalable_pipeline,
)


def _write_sources(root: Path, negative: bool = False) -> None:
    root.mkdir(parents=True)
    click = -2 if negative else 2
    (root / "studentVle.csv").write_text(
        "code_module,code_presentation,id_student,id_site,date,sum_click\n"
        f"AAA,2013J,10,1,1,{click}\nAAA,2013J,10,1,1,3\n"
        "AAA,2013J,10,2,1,4\nAAA,2013J,10,3,1,5\n"
        "AAA,2013J,10,4,1,6\nAAA,2013J,10,5,1,7\n"
        "AAA,2013J,11,999,1,8\nAAA,2014J,10,1,1,9\nBBB,2013J,10,1,1,10\n",
        encoding="utf-8",
    )
    (root / "vle.csv").write_text(
        "id_site,code_module,code_presentation,activity_type,week_from,week_to\n"
        "1,AAA,2013J,homepage,?,?\n2,AAA,2013J,forumng,?,?\n"
        "3,AAA,2013J,oucontent,?,?\n4,AAA,2013J,quiz,?,?\n"
        "5,AAA,2013J,externalquiz,?,?\n1,AAA,2014J,homepage,?,?\n"
        "1,BBB,2013J,forumng,?,?\n",
        encoding="utf-8",
    )
    (root / "studentInfo.csv").write_text(
        "code_module,code_presentation,id_student,final_result\nAAA,2013J,10,Pass\n",
        encoding="utf-8",
    )
    (root / "studentRegistration.csv").write_text(
        "code_module,code_presentation,id_student,date_registration,date_unregistration\nAAA,2013J,10,0,?\n",
        encoding="utf-8",
    )
    (root / "courses.csv").write_text(
        "code_module,code_presentation,module_presentation_length\nAAA,2013J,268\n",
        encoding="utf-8",
    )


class ScalablePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.raw = cls.root / "raw"
        cls.output = cls.root / "daily_activity"
        cls.database = cls.root / "metadata.duckdb"
        _write_sources(cls.raw)
        cls.source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in cls.raw.iterdir()}
        cls.result = build_scalable_pipeline(
            cls.raw, cls.output, cls.database, include_parity=False
        )
        cls.rows = cls._read_rows()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def _read_rows(cls):
        with duckdb.connect() as connection:
            return connection.execute(
                f"SELECT module,presentation,student_id,day,total_clicks,number_of_resources_visited,number_of_activity_types_used,homepage_clicks,forum_clicks,content_clicks,quiz_clicks,resource_ids,activity_types FROM read_parquet('{_sql_path(cls.output / '**' / '*.parquet')}', hive_partitioning=true) ORDER BY module,presentation,student_id,day"
            ).fetchall()

    def _row(self, module="AAA", presentation="2013J", student=10):
        return next(row for row in self.rows if row[0] == module and row[1] == presentation and row[2] == student)

    def test_correct_join_keys(self):
        row = self._row()
        self.assertEqual(row[4:11], (27, 5, 5, 5, 4, 5, 13))

    def test_correct_daily_grouping(self): self.assertEqual(len(self.rows), 4)
    def test_exact_click_totals(self): self.assertEqual(self.result.report["total_clicks_before_aggregation"], 54); self.assertEqual(self.result.report["total_clicks_after_aggregation"], 54)
    def test_exact_distinct_resources(self): self.assertEqual(self._row()[5], 5); self.assertEqual(set(self._row()[11]), {1, 2, 3, 4, 5})
    def test_exact_distinct_activity_types(self): self.assertEqual(self._row()[6], 5); self.assertEqual(set(self._row()[12]), {"homepage", "forumng", "oucontent", "quiz", "externalquiz"})
    def test_homepage_totals(self): self.assertEqual(self._row()[7], 5)
    def test_forum_totals(self): self.assertEqual(self._row()[8], 4)
    def test_content_totals(self): self.assertEqual(self._row()[9], 5)
    def test_quiz_and_externalquiz_totals(self): self.assertEqual(self._row()[10], 13)
    def test_modules_remain_separate(self): self.assertEqual(self._row("BBB", "2013J")[4], 10)
    def test_presentations_remain_separate(self): self.assertEqual(self._row("AAA", "2014J")[4], 9)
    def test_students_remain_separate(self): self.assertEqual(self._row(student=11)[4], 8)

    def test_duplicate_daily_keys_are_reported(self):
        with duckdb.connect() as connection:
            connection.execute("CREATE TABLE duplicate_rows AS SELECT 'AAA' module, '2013J' presentation, 1 student_id, 1 AS day, 0 total_clicks, 1 number_of_resources_visited, 1 number_of_activity_types_used, 0 homepage_clicks, 0 forum_clicks, 0 content_clicks, 0 quiz_clicks, [1] resource_ids, ['homepage'] activity_types")
            connection.execute("INSERT INTO duplicate_rows SELECT * FROM duplicate_rows")
            folder = self.root / "duplicate_parquet"
            folder.mkdir(exist_ok=True)
            connection.execute(f"COPY duplicate_rows TO '{_sql_path(folder / 'part.parquet')}' (FORMAT PARQUET)")
            stats = _processed_stats(connection, _sql_path(folder / "*.parquet"))
        self.assertEqual(stats["duplicate_daily_keys"], 1)

    def test_negative_clicks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / "raw"; _write_sources(raw, negative=True)
            with self.assertRaises(ValueError):
                build_scalable_pipeline(raw, root / "out", root / "db.duckdb", include_parity=False)

    def test_unmatched_site_ids_are_reported(self): self.assertEqual(self.result.report["unmatched_site_ids"], 1)
    def test_parquet_output_can_be_read(self): self.assertEqual(len(self.rows), 4)

    def test_partition_columns_are_preserved(self):
        names = [path.as_posix() for path in self.output.rglob("*.parquet")]
        self.assertTrue(any("module=AAA/presentation=2013J" in name for name in names))

    def test_repeated_builds_are_deterministic(self):
        before = self._read_rows()
        build_scalable_pipeline(self.raw, self.output, self.database, include_parity=False)
        self.assertEqual(before, self._read_rows())

    def test_validation_only_does_not_rebuild(self):
        before = {path: path.stat().st_mtime_ns for path in [*self.output.rglob("*.parquet"), self.database]}
        time.sleep(0.01)
        result = validate_scalable_pipeline(self.raw, self.output, self.database, include_parity=False)
        after = {path: path.stat().st_mtime_ns for path in before}
        self.assertFalse(result.rebuilt)
        self.assertEqual(before, after)

    def test_original_sources_are_not_modified(self):
        after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.raw.iterdir()}
        self.assertEqual(self.source_hashes, after)

    def test_duckdb_resource_configuration_is_applied(self):
        with tempfile.TemporaryDirectory() as directory, duckdb.connect() as connection:
            temp_path = Path(directory) / "controlled_spill"
            effective = configure_duckdb(
                connection, DuckDBConfig("64MB", 1, "1GB"), temp_path
            )
            self.assertIn("MiB", effective["memory_limit"])
            self.assertEqual(effective["threads"], "1")
            self.assertEqual(effective["preserve_insertion_order"].lower(), "false")
            self.assertEqual(Path(effective["temp_directory"]), temp_path.resolve())
            self.assertEqual(effective["max_temp_directory_size"], "953.6 MiB")

    def test_invalid_resource_configuration_is_rejected(self):
        for config in (
            DuckDBConfig("2GB'; DROP TABLE x;--", 2, "20GB"),
            DuckDBConfig("2GB", 0, "20GB"),
            DuckDBConfig("2GB", 2, "unlimited"),
            DuckDBConfig("-1GB", 2, "20GB"),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate_duckdb_config(config)

    def test_partitioned_copy_has_no_global_order_by(self):
        self.assertNotIn("ORDER BY", inspect.getsource(_copy_partitioned_daily).upper())

    def test_memory_limited_miniature_build_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / "raw"; _write_sources(raw)
            result = build_scalable_pipeline(
                raw, root / "out", root / "db.duckdb", include_parity=False,
                config=DuckDBConfig("128MB", 1, "1GB"),
            )
            self.assertTrue(result.rebuilt)
            self.assertEqual(result.report["total_clicks_after_aggregation"], 54)

    def test_failed_build_preserves_existing_output(self):
        before_rows = self._read_rows()
        before_database = hashlib.sha256(self.database.read_bytes()).hexdigest()
        original = self.raw / "studentVle.csv"
        original_text = original.read_text(encoding="utf-8")
        try:
            original.write_text(original_text.replace("AAA,2013J,10,1,1,2", "AAA,2013J,10,1,1,-2"), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_scalable_pipeline(self.raw, self.output, self.database, include_parity=False)
        finally:
            original.write_text(original_text, encoding="utf-8")
        self.assertEqual(before_rows, self._read_rows())
        self.assertEqual(before_database, hashlib.sha256(self.database.read_bytes()).hexdigest())

    def test_temporary_build_and_spill_files_are_cleaned(self):
        controlled = self.output.parent / ".duckdb_temp"
        self.assertFalse(controlled.exists())


if __name__ == "__main__":
    unittest.main()
