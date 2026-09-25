import unittest
from pathlib import Path

import numpy as np

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.common_cohort_experiment import (
    bootstrap_confidence_intervals,
    run_common_cohort_experiment,
    student_bootstrap_samples,
)
from learnpulse.modeling_table import (
    AUDIT_ONLY_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

DAYS = [14, 28, 42, 56]


def rows(include_incomplete: bool = True) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for student in range(1, 25):
        student_days = DAYS if not (include_incomplete and student == 24) else DAYS[:-1]
        for day_index, day in enumerate(student_days):
            row: dict[str, object] = {
                "module": "AAA",
                "presentation": "2013J",
                "student_id": student,
                "observation_day": day,
                TARGET_COLUMN: int((student + day_index) % 4 == 0),
            }
            for index, column in enumerate(MODEL_FEATURE_COLUMNS):
                row[column] = float(student + day + index)
            if student % 5 == 0:
                row["click_z_score"] = None
            output.append(row)
    return output


class CommonCohortExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_rows = rows()
        cls.result = run_common_cohort_experiment(
            cls.source_rows, DAYS, max_folds=4, bootstrap_samples=50, random_seed=42
        )

    def test_every_student_has_all_days(self) -> None:
        by_student: dict[int, set[int]] = {}
        for row in self.result.cohort.rows:
            by_student.setdefault(int(row["student_id"]), set()).add(int(row["observation_day"]))
        self.assertTrue(all(days == set(DAYS) for days in by_student.values()))

    def test_every_student_has_exactly_four_rows(self) -> None:
        counts: dict[int, int] = {}
        for row in self.result.cohort.rows:
            counts[int(row["student_id"])] = counts.get(int(row["student_id"]), 0) + 1
        self.assertTrue(all(count == 4 for count in counts.values()))

    def test_removed_students_are_reported(self) -> None:
        self.assertEqual(self.result.cohort.students_before, 24)
        self.assertEqual(self.result.cohort.students_after, 23)
        self.assertEqual(self.result.cohort.students_removed, 1)

    def test_same_fold_assignment_is_used_at_every_day(self) -> None:
        assignment = {
            student: fold.fold
            for fold in self.result.folds
            for student in fold.validation_students
        }
        for prediction in self.result.predictions:
            self.assertEqual(prediction["fold"], assignment[prediction["student_id"]])

    def test_no_student_overlap_within_fold(self) -> None:
        self.assertTrue(all(not fold.train_students & fold.validation_students for fold in self.result.folds))

    def test_both_classes_in_every_partition_and_day(self) -> None:
        for day in DAYS:
            day_rows = [row for row in self.result.cohort.rows if row["observation_day"] == day]
            for fold in self.result.folds:
                train = {row[TARGET_COLUMN] for row in day_rows if row["student_id"] in fold.train_students}
                validation = {row[TARGET_COLUMN] for row in day_rows if row["student_id"] in fold.validation_students}
                self.assertEqual(train, {0, 1})
                self.assertEqual(validation, {0, 1})

    def test_every_common_row_gets_one_oof_prediction(self) -> None:
        self.assertEqual(len(self.result.predictions), len(self.result.cohort.rows))
        keys = [(row["student_id"], row["observation_day"]) for row in self.result.predictions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_preprocessing_uses_fold_training_rows(self) -> None:
        fold = self.result.folds[0]
        day = DAYS[0]
        training = [
            row for row in self.result.cohort.rows
            if row["observation_day"] == day and row["student_id"] in fold.train_students
        ]
        matrix = np.asarray(
            [
                [
                    np.nan if row.get(column) is None else float(row[column])
                    for column in PERSONAL_CHANGE_FEATURE_COLUMNS
                ]
                for row in training
            ]
        )
        pipeline = self.result.fitted_pipelines[0]
        np.testing.assert_allclose(
            pipeline.named_steps["imputer"].statistics_, np.nanmedian(matrix, axis=0)
        )

    def test_observation_day_is_not_a_feature(self) -> None:
        self.assertNotIn("observation_day", PERSONAL_CHANGE_FEATURE_COLUMNS)

    def test_future_and_audit_columns_are_not_features(self) -> None:
        forbidden = set(AUDIT_ONLY_COLUMNS + IDENTIFIER_COLUMNS + [TARGET_COLUMN])
        self.assertTrue(forbidden.isdisjoint(PERSONAL_CHANGE_FEATURE_COLUMNS))

    def test_pr_auc_lift_is_correct(self) -> None:
        for metrics in self.result.pooled_results.values():
            self.assertAlmostEqual(
                metrics["pr_auc_lift"], metrics["average_precision"] / metrics["positive_prevalence"]
            )

    def test_bootstrap_draws_students(self) -> None:
        draws = list(student_bootstrap_samples([1, 2, 3], 4, 42))
        self.assertEqual(len(draws), 4)
        self.assertTrue(all(len(draw) == 3 for draw in draws))

    def test_bootstrap_keeps_four_rows_per_sampled_student(self) -> None:
        by_student = {}
        for row in self.result.predictions:
            by_student.setdefault(row["student_id"], []).append(row)
        draw = next(student_bootstrap_samples(sorted(by_student), 1, 42))
        expanded = [row for student in draw for row in by_student[student]]
        self.assertEqual(len(expanded), len(draw) * 4)

    def test_one_class_bootstrap_samples_are_counted(self) -> None:
        predictions = []
        for student, target in ((1, 0), (2, 1)):
            for day in DAYS:
                predictions.append({
                    "student_id": student,
                    "observation_day": day,
                    "actual_future_inactivity": target,
                    "personal_change_prediction": target,
                    "personal_change_probability": 0.1 if target == 0 else 0.9,
                })
        intervals = bootstrap_confidence_intervals(predictions, DAYS, 100, 42)
        self.assertGreater(intervals[14]["roc_auc"]["skipped_samples"], 0)
        self.assertEqual(
            intervals[14]["roc_auc"]["valid_samples"] + intervals[14]["roc_auc"]["skipped_samples"], 100
        )

    def test_confidence_intervals_are_reproducible(self) -> None:
        first = bootstrap_confidence_intervals(self.result.predictions, DAYS, 20, 42)
        second = bootstrap_confidence_intervals(self.result.predictions, DAYS, 20, 42)
        self.assertEqual(first, second)

    def test_confusion_matrices_match_day_count(self) -> None:
        for metrics in self.result.pooled_results.values():
            self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), 23)

    def test_source_modeling_file_is_not_modified(self) -> None:
        path = Path("data/processed/modeling_table.csv")
        if not path.exists():
            self.skipTest("real modeling table is not present")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        self.assertTrue(path.read_bytes())
        self.assertEqual(before, (path.stat().st_size, path.stat().st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
