"""Tests for Parquet-backed reuse of trusted feature calculations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import DailyLearningEvent, iter_daily_events, load_activity_types
from learnpulse.parquet_features_cli import limit_messages
from learnpulse.parquet_reader import (
    calculate_parquet_features,
    combined_feature_messages,
    partition_files,
    read_daily_activity,
)
from learnpulse.personal_baseline import calculate_personal_baselines
from learnpulse.rolling_features import calculate_rolling_features


def _create_dataset(path: Path) -> None:
    path.mkdir()
    with duckdb.connect() as connection:
        connection.execute(
            """CREATE TABLE daily AS SELECT * FROM (VALUES
               ('AAA','2013J',1,1,5,2,2,3,2,0,0,[10,11],['forumng','homepage']),
               ('AAA','2013J',1,3,7,1,1,7,0,0,0,[10],['homepage']),
               ('AAA','2013J',1,10,2,1,1,0,0,0,2,[12],['quiz']),
               ('AAA','2013J',2,2,9,1,1,0,9,0,0,[11],['forumng']),
               ('AAA','2014J',1,1,20,1,1,20,0,0,0,[20],['homepage']),
               ('BBB','2013J',1,1,30,1,1,0,0,30,0,[30],['oucontent'])
            ) t(module,presentation,student_id,day,total_clicks,
                number_of_resources_visited,number_of_activity_types_used,
                homepage_clicks,forum_clicks,content_clicks,quiz_clicks,
                resource_ids,activity_types)"""
        )
        connection.execute(
            "COPY daily TO ? (FORMAT PARQUET, PARTITION_BY(module,presentation))",
            [str(path.resolve())],
        )


def _trusted_events() -> list[DailyLearningEvent]:
    return [
        DailyLearningEvent("AAA", "2013J", 1, 10, 1, 3, "homepage"),
        DailyLearningEvent("AAA", "2013J", 1, 11, 1, 2, "forumng"),
        DailyLearningEvent("AAA", "2013J", 1, 10, 3, 7, "homepage"),
        DailyLearningEvent("AAA", "2013J", 1, 12, 10, 2, "quiz"),
    ]


def _assert_messages_equal(test: unittest.TestCase, left, right) -> None:
    test.assertEqual(set(left), set(right))
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, float) or isinstance(b, float):
            test.assertIsNotNone(a); test.assertIsNotNone(b)
            test.assertLessEqual(abs(float(a) - float(b)), 1e-9, key)
        else:
            test.assertEqual(a, b, key)


class ParquetReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.dataset = Path(cls.temp.name) / "daily_activity"
        _create_dataset(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_module_and_presentation_filtering(self):
        rows = read_daily_activity(self.dataset, "AAA", "2013J")
        self.assertTrue(rows)
        self.assertEqual({(row.module, row.presentation) for row in rows}, {("AAA", "2013J")})

    def test_student_filtering(self):
        rows = read_daily_activity(self.dataset, "AAA", "2013J", student_id=2)
        self.assertEqual([(row.student_id, row.total_clicks) for row in rows], [(2, 9)])

    def test_inclusive_day_filtering(self):
        rows = read_daily_activity(self.dataset, "AAA", "2013J", 1, 1, 3)
        self.assertEqual([row.day for row in rows], [1, 3])

    def test_chronological_ordering(self):
        rows = read_daily_activity(self.dataset, "AAA", "2013J")
        keys = [(row.module, row.presentation, row.student_id, row.day) for row in rows]
        self.assertEqual(keys, sorted(keys))

    def test_partition_isolation(self):
        self.assertEqual(len(partition_files(self.dataset, "AAA", "2014J")), 1)
        self.assertEqual(read_daily_activity(self.dataset, "AAA", "2014J")[0].total_clicks, 20)
        self.assertEqual(read_daily_activity(self.dataset, "BBB", "2013J")[0].total_clicks, 30)

    def test_resource_ids_are_exact(self):
        row = read_daily_activity(self.dataset, "AAA", "2013J", 1, 1, 1)[0]
        self.assertEqual(row.resource_ids, frozenset({10, 11}))

    def test_activity_types_are_exact(self):
        row = read_daily_activity(self.dataset, "AAA", "2013J", 1, 1, 1)[0]
        self.assertEqual(row.activity_types, frozenset({"homepage", "forumng"}))

    def test_rolling_feature_compatibility(self):
        daily = read_daily_activity(self.dataset, "AAA", "2013J", 1)
        rows = calculate_rolling_features(daily, 1, 10)
        self.assertEqual(rows[-1].clicks_last_7_days, 2)
        self.assertEqual(rows[2].resources_last_7_days, 2)

    def test_personal_baseline_compatibility(self):
        result = calculate_parquet_features(self.dataset, "AAA", "2013J", 1, 1, 10, 2)
        self.assertEqual(len(result.baselines), 10)
        self.assertTrue(result.baselines[-1].baseline_ready)

    def test_limit_changes_display_only(self):
        result = calculate_parquet_features(self.dataset, "AAA", "2013J", 1, 1, 10, 2)
        messages = combined_feature_messages(result)
        displayed = limit_messages(messages, 3)
        self.assertEqual(len(displayed), 3)
        self.assertEqual(len(result.rolling), 10)
        self.assertEqual(len(result.baselines), 10)
        self.assertEqual(displayed, messages[:3])

    def test_invalid_day_range(self):
        with self.assertRaises(ValueError):
            read_daily_activity(self.dataset, "AAA", "2013J", start_day=5, end_day=4)

    def test_missing_dataset_path(self):
        with self.assertRaises(FileNotFoundError):
            read_daily_activity(Path(self.temp.name) / "missing", "AAA", "2013J")

    def test_empty_query_result(self):
        self.assertEqual(read_daily_activity(self.dataset, "AAA", "2013J", 999), [])

    def test_source_parquet_is_not_modified(self):
        before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in self.dataset.rglob("*.parquet")}
        read_daily_activity(self.dataset, "AAA", "2013J", 1)
        after = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in self.dataset.rglob("*.parquet")}
        self.assertEqual(before, after)

    def test_exact_miniature_trusted_parity(self):
        trusted_daily = aggregate_daily_activity(_trusted_events())
        parquet_result = calculate_parquet_features(
            self.dataset, "AAA", "2013J", 1, 1, 10, 2
        )
        self.assertEqual(trusted_daily, parquet_result.daily)
        trusted_rolling = calculate_rolling_features(trusted_daily, 1, 10)
        trusted_baselines = calculate_personal_baselines(trusted_rolling, 2)
        trusted = [
            {**rolling.as_message(), **baseline.as_message()}
            for rolling, baseline in zip(trusted_rolling, trusted_baselines)
        ]
        parquet = combined_feature_messages(parquet_result)
        self.assertEqual(len(trusted), len(parquet))
        for left, right in zip(trusted, parquet):
            _assert_messages_equal(self, left, right)

    def test_real_aaa_student_parity_when_available(self):
        dataset = Path("data/processed/daily_activity")
        raw = Path("data/raw/oulad")
        if not dataset.is_dir() or not (raw / "studentVle.csv").is_file():
            self.skipTest("Real processed Parquet and OULAD CSV files are unavailable")
        parquet_result = calculate_parquet_features(
            dataset, "AAA", "2013J", 28400, -10, 20, 7
        )
        parquet = combined_feature_messages(parquet_result)
        lookup = load_activity_types(raw / "vle.csv")
        events = (
            event for event in iter_daily_events(
                raw / "studentVle.csv", lookup, "AAA", "2013J"
            ) if event.student_id == 28400
        )
        daily = aggregate_daily_activity(events)
        self.assertEqual(daily, parquet_result.daily)
        rolling = calculate_rolling_features(daily, -10, 20)
        baselines = calculate_personal_baselines(rolling, 7)
        trusted = [{**r.as_message(), **b.as_message()} for r, b in zip(rolling, baselines)]
        self.assertEqual(len(trusted), len(parquet))
        for left, right in zip(trusted, parquet):
            _assert_messages_equal(self, left, right)


if __name__ == "__main__":
    unittest.main()
