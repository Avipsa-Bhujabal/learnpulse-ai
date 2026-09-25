"""Tests for unseen-presentation and unseen-module evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb

from learnpulse.ablation_experiment import (
    ABSOLUTE_FEATURE_COLUMNS,
    COMBINED_FEATURE_COLUMNS,
    PERSONAL_CHANGE_FEATURE_COLUMNS,
)
from learnpulse.crosscourse_experiment import (
    FORBIDDEN_COLUMNS,
    MODEL_FEATURE_GROUPS,
    _median_iqr,
    calculate_metrics_safe,
    create_course_holdout_splits,
    create_grouped_reference_splits,
    dataset_shift_rows,
    run_crosscourse_experiment,
    save_crosscourse_outputs,
    validate_feature_schemas,
)
from learnpulse.modeling_table import MODEL_FEATURE_COLUMNS


def rows_fixture():
    rows = []
    units = [("AAA", "2013J"), ("AAA", "2014J"), ("BBB", "2013J"), ("BBB", "2014J"), ("CCC", "2013J"), ("CCC", "2014J")]
    for unit_index, (module, presentation) in enumerate(units):
        students = list(range(1, 7)) + list(range(100 + unit_index * 20, 112 + unit_index * 20))
        for day in (14, 28):
            for position, student in enumerate(students):
                label = int((position + unit_index + day // 14) % 5 == 0)
                row = {"module": module, "presentation": presentation, "student_id": student, "observation_day": day, "future_inactivity": label}
                for feature_index, feature in enumerate(MODEL_FEATURE_COLUMNS):
                    row[feature] = None if feature == "click_z_score" and position % 7 == 0 else float(position + feature_index + label * 3)
                rows.append(row)
    return rows


class CrossCourseExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = rows_fixture()

    def test_exact_feature_schemas_and_no_duplicates(self):
        self.assertEqual(MODEL_FEATURE_GROUPS["A"], ABSOLUTE_FEATURE_COLUMNS)
        self.assertEqual(MODEL_FEATURE_GROUPS["B"], PERSONAL_CHANGE_FEATURE_COLUMNS)
        self.assertEqual(MODEL_FEATURE_GROUPS["C"], COMBINED_FEATURE_COLUMNS)
        self.assertEqual(len(MODEL_FEATURE_GROUPS["C"]), len(set(MODEL_FEATURE_GROUPS["C"])))

    def test_forbidden_observation_target_and_audit_columns_rejected(self):
        self.assertTrue(FORBIDDEN_COLUMNS.isdisjoint(MODEL_FEATURE_GROUPS["C"]))
        bad = {name: list(values) for name, values in MODEL_FEATURE_GROUPS.items()}; bad["A"].append("observation_day")
        with self.assertRaises(ValueError): validate_feature_schemas(bad)

    def test_separate_training_by_observation_day(self):
        splits, _ = create_grouped_reference_splits(self.rows, 14, 3, 42)
        self.assertTrue(all(self.rows[i]["observation_day"] == 14 for split in splits for i in split.train_indices + split.test_indices))

    def test_grouped_folds_keep_students_together_and_are_reusable(self):
        first, _ = create_grouped_reference_splits(self.rows, 14, 3, 42)
        second, _ = create_grouped_reference_splits(self.rows, 14, 3, 42)
        self.assertEqual(first, second)
        for split in first:
            train = {self.rows[i]["student_id"] for i in split.train_indices}; test = {self.rows[i]["student_id"] for i in split.test_indices}
            self.assertFalse(train & test)

    def test_complete_presentation_and_module_holdouts(self):
        presentation = create_course_holdout_splits(self.rows, 14, "presentation", ["course-only"])[0]
        self.assertEqual({(self.rows[i]["module"], self.rows[i]["presentation"]) for i in presentation.test_indices}, {(presentation.held_out_module, presentation.held_out_presentation)})
        module = create_course_holdout_splits(self.rows, 14, "module", ["course-only"])[0]
        self.assertEqual({self.rows[i]["module"] for i in module.test_indices}, {module.held_out_module})
        self.assertNotIn(module.held_out_module, {self.rows[i]["module"] for i in module.train_indices})

    def test_overlap_count_and_strict_removal(self):
        splits = create_course_holdout_splits(self.rows, 14, "presentation", ["course-only", "student-disjoint"], only_module="AAA", only_presentation="2013J")
        course, strict = splits
        self.assertGreater(course.overlapping_students_before_removal, 0)
        self.assertEqual(course.training_students_removed, 0)
        self.assertGreater(strict.training_rows_removed, 0)
        train = {self.rows[i]["student_id"] for i in strict.train_indices}; test = {self.rows[i]["student_id"] for i in strict.test_indices}
        self.assertFalse(train & test)

    def test_one_class_metrics_are_safe_and_confusion_sums(self):
        metrics = calculate_metrics_safe([0, 0], [0, 1], [0.2, 0.8])
        self.assertIsNone(metrics["roc_auc"]); self.assertIsNone(metrics["recall"])
        self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), 2)

    def test_metric_calculations(self):
        metrics = calculate_metrics_safe([0, 0, 1, 1], [0, 1, 0, 1], [.1, .8, .4, .9])
        self.assertEqual(metrics["precision"], .5); self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["specificity"], .5); self.assertEqual(metrics["no_skill_pr_auc"], .5)

    def test_preprocessing_fits_training_only_and_predictions_cover_test(self):
        result = run_crosscourse_experiment(self.rows, [14], ["B"], ["presentation"], ["student-disjoint"], 3, .5, 42, "AAA", "2013J")
        self.assertEqual(result.fitted_models, 1)
        self.assertEqual(len(result.predictions), len([row for row in self.rows if row["module"] == "AAA" and row["presentation"] == "2013J" and row["observation_day"] == 14]))
        self.assertTrue(all(0 <= row["model_output"] <= 1 for row in result.predictions))

    def test_same_holdout_is_used_for_three_models(self):
        result = run_crosscourse_experiment(self.rows, [14], ["A", "B", "C"], ["presentation"], ["course-only"], 3, .5, 42, "AAA", "2013J")
        self.assertEqual({row["fold_or_holdout_id"] for row in result.fold_results}, {"AAA_2013J"})
        self.assertEqual({row["model"] for row in result.fold_results}, {"A", "B", "C"})

    def test_macro_pooled_and_worst_summaries(self):
        result = run_crosscourse_experiment(self.rows, [14], ["A"], ["module"], ["course-only"], 3)
        self.assertEqual({row["summary_type"] for row in result.summary_rows}, {"macro", "pooled"})
        self.assertTrue(result.report["worst_group_results"])

    def test_paired_model_differences(self):
        result = run_crosscourse_experiment(self.rows, [14], ["A", "B", "C"], ["presentation"], ["course-only"], 3, .5, 42, "AAA")
        comparisons = result.report["paired_model_comparisons"]
        self.assertTrue(any(row["comparison"] == "B_minus_A" and row["metric"] == "average_precision" for row in comparisons))

    def test_dataset_shift_and_zero_iqr(self):
        split = create_course_holdout_splits(self.rows, 14, "presentation", ["course-only"], "AAA", "2013J")[0]
        shift = dataset_shift_rows(self.rows, split)
        self.assertEqual(len(shift), len(MODEL_FEATURE_COLUMNS))
        self.assertIn("standardized_median_difference", shift[0])
        self.assertEqual(_median_iqr([2, 2, 2]), (2.0, 0.0))

    def test_deterministic_experiment(self):
        args = (self.rows, [14], ["B"], ["presentation"], ["student-disjoint"], 3, .5, 42, "AAA", "2013J")
        first = run_crosscourse_experiment(*args); second = run_crosscourse_experiment(*args)
        self.assertEqual(first.fold_results, second.fold_results); self.assertEqual(first.predictions, second.predictions)

    def test_atomic_outputs_and_source_not_modified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "source.parquet"
            with duckdb.connect() as connection: connection.execute("COPY (SELECT 1 x) TO ? (FORMAT PARQUET)", [str(source)])
            before = (source.stat().st_size, source.stat().st_mtime_ns)
            result = run_crosscourse_experiment(self.rows, [14], ["B"], ["presentation"], ["course-only"], 3, .5, 42, "AAA", "2013J")
            paths = [root / name for name in ["result.json", "fold.csv", "pred.parquet", "summary.csv", "feature.csv"]]
            save_crosscourse_outputs(result, source, *paths, {})
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(before, (source.stat().st_size, source.stat().st_mtime_ns))

    def test_failed_publication_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "source.parquet"
            with duckdb.connect() as connection: connection.execute("COPY (SELECT 1 x) TO ? (FORMAT PARQUET)", [str(source)])
            targets = [root / name for name in ["result.json", "fold.csv", "pred.parquet", "summary.csv", "feature.csv"]]
            for target in targets: target.write_bytes(b"old")
            result = run_crosscourse_experiment(self.rows, [14], ["B"], ["presentation"], ["course-only"], 3, .5, 42, "AAA", "2013J")
            with patch("learnpulse.crosscourse_experiment._publish", side_effect=RuntimeError("forced")):
                with self.assertRaises(RuntimeError): save_crosscourse_outputs(result, source, *targets, {})
            self.assertTrue(all(target.read_bytes() == b"old" for target in targets))


if __name__ == "__main__": unittest.main()
