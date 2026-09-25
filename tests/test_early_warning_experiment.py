import unittest
from pathlib import Path

import numpy as np

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.early_warning_experiment import (
    build_trajectories,
    create_day_folds,
    run_early_warning_experiment,
    validate_early_warning_features,
)
from learnpulse.modeling_table import (
    AUDIT_ONLY_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)

DAYS = [14, 28, 42, 56]


def sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for student in range(1, 31):
        for day_index, day in enumerate(DAYS):
            row: dict[str, object] = {
                "module": "AAA",
                "presentation": "2013J",
                "student_id": student,
                "observation_day": day,
                TARGET_COLUMN: int(student % 3 == day_index % 3),
            }
            for index, column in enumerate(MODEL_FEATURE_COLUMNS):
                row[column] = float(student + day + index)
            if student % 5 == 0:
                row["click_z_score"] = None
            rows.append(row)
    return rows


class EarlyWarningExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = sample_rows()
        cls.result = run_early_warning_experiment(cls.rows, DAYS, 5, 42)

    def test_each_observation_day_is_evaluated_separately(self) -> None:
        self.assertEqual(set(self.result.pooled_results), set(DAYS))
        self.assertEqual({row["observation_day"] for row in self.result.fold_results}, set(DAYS))

    def test_observation_day_is_not_a_feature(self) -> None:
        self.assertNotIn("observation_day", PERSONAL_CHANGE_FEATURE_COLUMNS)

    def test_no_student_overlap_within_folds(self) -> None:
        self.assertTrue(all(not fold.train_students & fold.validation_students for fold in self.result.folds))

    def test_consistent_assignments_are_reused_when_valid(self) -> None:
        common = [fold for fold in self.result.folds if fold.uses_common_assignment]
        self.assertTrue(common)
        mappings: dict[int, dict[int, int]] = {}
        for fold in common:
            mappings.setdefault(fold.observation_day, {}).update(
                {student: fold.fold for student in fold.validation_students}
            )
        days = sorted(mappings)
        shared = set.intersection(*(set(mappings[day]) for day in days))
        for student in shared:
            self.assertEqual(len({mappings[day][student] for day in days}), 1)

    def test_both_classes_exist_in_every_partition(self) -> None:
        for fold in self.result.folds:
            train = {self.rows[i][TARGET_COLUMN] for i in fold.train_indices}
            validation = {self.rows[i][TARGET_COLUMN] for i in fold.validation_indices}
            self.assertEqual(train, {0, 1})
            self.assertEqual(validation, {0, 1})

    def test_every_row_has_one_oof_prediction(self) -> None:
        self.assertEqual(len(self.result.predictions), len(self.rows))
        keys = [(row["student_id"], row["observation_day"]) for row in self.result.predictions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_preprocessing_is_fit_on_fold_training_rows(self) -> None:
        fold = self.result.folds[0]
        pipeline = self.result.fitted_pipelines[0]
        matrix = np.asarray(
            [
                [
                    np.nan if self.rows[i].get(column) is None else float(self.rows[i][column])
                    for column in PERSONAL_CHANGE_FEATURE_COLUMNS
                ]
                for i in fold.train_indices
            ]
        )
        np.testing.assert_allclose(
            pipeline.named_steps["imputer"].statistics_, np.nanmedian(matrix, axis=0)
        )

    def test_missing_values_are_handled(self) -> None:
        self.assertTrue(all(row["personal_change_probability"] is not None for row in self.result.predictions))

    def test_probabilities_are_between_zero_and_one(self) -> None:
        self.assertTrue(all(0 <= float(row["personal_change_probability"]) <= 1 for row in self.result.predictions))

    def test_predictions_are_binary(self) -> None:
        self.assertTrue({row["personal_change_prediction"] for row in self.result.predictions} <= {0, 1})

    def test_metrics_are_calculated(self) -> None:
        for day in DAYS:
            self.assertIn("average_precision", self.result.pooled_results[day])
            self.assertIn("brier_score", self.result.pooled_results[day])

    def test_confusion_matrices_sum_to_day_rows(self) -> None:
        for day in DAYS:
            matrix = self.result.pooled_results[day]["confusion_matrix"]
            expected = sum(row["observation_day"] == day for row in self.rows)
            self.assertEqual(sum(sum(part) for part in matrix), expected)

    def test_results_are_reproducible(self) -> None:
        repeated = run_early_warning_experiment(self.rows, DAYS, 5, 42)
        self.assertEqual(self.result.predictions, repeated.predictions)
        self.assertEqual(self.result.fold_results, repeated.fold_results)

    def test_trajectories_use_only_oof_predictions(self) -> None:
        self.assertTrue(all(row["prediction_source"] == "out_of_fold" for row in self.result.predictions))
        with self.assertRaises(ValueError):
            build_trajectories([{**self.result.predictions[0], "prediction_source": "training"}])

    def test_forbidden_columns_are_not_features(self) -> None:
        forbidden = set(AUDIT_ONLY_COLUMNS + IDENTIFIER_COLUMNS + [TARGET_COLUMN])
        self.assertTrue(forbidden.isdisjoint(PERSONAL_CHANGE_FEATURE_COLUMNS))
        validate_early_warning_features()

    def test_requested_days_define_folds(self) -> None:
        folds = create_day_folds(self.rows, [14, 28], 5, 42)
        self.assertEqual({fold.observation_day for fold in folds}, {14, 28})

    def test_original_modeling_file_is_not_modified(self) -> None:
        path = Path("data/processed/modeling_table.csv")
        if not path.exists():
            self.skipTest("real modeling table is not present")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        self.assertTrue(path.read_bytes())
        self.assertEqual(before, (path.stat().st_size, path.stat().st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
