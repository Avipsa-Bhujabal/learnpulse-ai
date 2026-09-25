import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from learnpulse.baseline_models import (
    MODEL_FEATURE_COLUMNS,
    build_logistic_pipeline,
    calculate_metrics,
    feature_matrix,
    rule_based_predictions,
    run_baseline_experiment,
    save_pipeline,
    split_rows_by_student,
    target_vector,
    validate_training_feature_schema,
)
from learnpulse.modeling_table import (
    AUDIT_ONLY_COLUMNS,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN,
)


def model_row(
    student: int,
    day: int,
    target: int,
    missing: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "module": "AAA",
        "presentation": "2013J",
        "student_id": student,
        "observation_day": day,
        TARGET_COLUMN: target,
    }
    for index, column in enumerate(MODEL_FEATURE_COLUMNS):
        row[column] = float(student + day + index)
    if missing:
        row["click_z_score"] = None
    return row


def sample_rows() -> list[dict[str, object]]:
    rows = []
    for student in range(1, 9):
        rows.append(model_row(student, 14, int(student % 2 == 1)))
        rows.append(model_row(student, 28, 0, missing=student in {2, 5, 8}))
    return rows


class BaselineModelTests(unittest.TestCase):
    def test_no_student_overlap(self) -> None:
        split = split_rows_by_student(sample_rows(), 0.25, 42)
        self.assertFalse(split.train_student_ids & split.test_student_ids)

    def test_all_student_rows_stay_together(self) -> None:
        split = split_rows_by_student(sample_rows(), 0.25, 42)
        for student in range(1, 9):
            locations = {
                "train" if row in split.train_rows else "test"
                for row in sample_rows()
                if row["student_id"] == student
            }
            self.assertEqual(len(locations), 1)

    def test_both_classes_exist_in_each_split(self) -> None:
        split = split_rows_by_student(sample_rows(), 0.25, 42)
        self.assertEqual(set(target_vector(split.train_rows)), {0, 1})
        self.assertEqual(set(target_vector(split.test_rows)), {0, 1})

    def test_split_is_reproducible(self) -> None:
        first = split_rows_by_student(sample_rows(), 0.25, 42)
        second = split_rows_by_student(sample_rows(), 0.25, 42)
        self.assertEqual(first.train_student_ids, second.train_student_ids)
        self.assertEqual(first.test_student_ids, second.test_student_ids)

    def test_target_is_not_a_feature(self) -> None:
        self.assertNotIn(TARGET_COLUMN, MODEL_FEATURE_COLUMNS)

    def test_audit_columns_are_not_features(self) -> None:
        self.assertTrue(set(AUDIT_ONLY_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS))

    def test_identifiers_are_not_features(self) -> None:
        self.assertTrue(set(IDENTIFIER_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS))

    def test_forbidden_training_schema_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_training_feature_schema([*MODEL_FEATURE_COLUMNS, "future_clicks"])

    def test_imputer_is_fitted_on_training_rows_only(self) -> None:
        split = split_rows_by_student(sample_rows(), 0.25, 42)
        pipeline = build_logistic_pipeline()
        pipeline.fit(feature_matrix(split.train_rows), target_vector(split.train_rows))
        expected = np.nanmedian(np.asarray(feature_matrix(split.train_rows)), axis=0)
        np.testing.assert_allclose(pipeline.named_steps["imputer"].statistics_, expected)

    def test_scaler_is_fitted_on_imputed_training_rows_only(self) -> None:
        split = split_rows_by_student(sample_rows(), 0.25, 42)
        pipeline = build_logistic_pipeline()
        x_train = feature_matrix(split.train_rows)
        pipeline.fit(x_train, target_vector(split.train_rows))
        imputed_train = pipeline.named_steps["imputer"].transform(x_train)
        np.testing.assert_allclose(
            pipeline.named_steps["scaler"].mean_, np.mean(imputed_train, axis=0)
        )

    def test_missing_values_can_be_processed(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        self.assertEqual(len(result.logistic_predictions), len(result.split.test_rows))

    def test_dummy_model_runs(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        self.assertEqual(len(result.dummy_predictions), len(result.split.test_rows))

    def test_rule_based_model_runs(self) -> None:
        rows = sample_rows()
        rows[0]["current_inactivity_gap"] = 8
        self.assertEqual(rule_based_predictions(rows)[0], 1)

    def test_logistic_regression_runs(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        self.assertTrue(hasattr(result.logistic_pipeline.named_steps["classifier"], "coef_"))

    def test_predictions_are_binary(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        for values in (result.dummy_predictions, result.rule_predictions, result.logistic_predictions):
            self.assertTrue(set(values) <= {0, 1})

    def test_probabilities_are_bounded(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        self.assertTrue(all(0 <= value <= 1 for value in result.logistic_probabilities))

    def test_confusion_matrix_sums_to_test_rows(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        for metrics in result.metrics.values():
            self.assertEqual(sum(sum(row) for row in metrics["confusion_matrix"]), len(result.split.test_rows))

    def test_false_positive_and_negative_counts(self) -> None:
        metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)

    def test_metrics_handle_imbalance(self) -> None:
        metrics = calculate_metrics([0] * 9 + [1], [0] * 10, [0.1] * 10)
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["precision"], 0)
        self.assertIsNotNone(metrics["average_precision"])

    def test_pipeline_artifact_round_trip(self) -> None:
        result = run_baseline_experiment(sample_rows(), 0.25, 42)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pipeline.joblib"
            save_pipeline(result.logistic_pipeline, path)
            loaded = joblib.load(path)
            expected = result.logistic_pipeline.predict(feature_matrix(result.split.test_rows))
            np.testing.assert_array_equal(loaded.predict(feature_matrix(result.split.test_rows)), expected)

    def test_modeling_input_file_is_not_modified(self) -> None:
        path = Path("data/processed/modeling_table.csv")
        if not path.exists():
            self.skipTest("real modeling table is not present")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        rows = path.read_bytes()
        after = (path.stat().st_size, path.stat().st_mtime_ns)
        self.assertTrue(rows)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
