import statistics
import unittest
from pathlib import Path

from learnpulse.ablation_experiment import (
    ABSOLUTE_FEATURE_COLUMNS,
    COMBINED_FEATURE_COLUMNS,
    MODEL_FEATURE_GROUPS,
    PERSONAL_CHANGE_FEATURE_COLUMNS,
    run_ablation_experiment,
    validate_feature_groups,
)
from learnpulse.baseline_models import calculate_metrics
from learnpulse.modeling_table import (
    AUDIT_ONLY_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def rows() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for student in range(1, 21):
        for day in (14, 28, 42, 56):
            row: dict[str, object] = {
                "module": "AAA",
                "presentation": "2013J",
                "student_id": student,
                "observation_day": day,
                TARGET_COLUMN: int(day == 56),
            }
            for index, column in enumerate(MODEL_FEATURE_COLUMNS):
                row[column] = float(student + day + index)
            if student % 4 == 0:
                row["click_z_score"] = None
            result.append(row)
    return result


class AblationExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = rows()
        cls.result = run_ablation_experiment(cls.rows, 5, 42)

    def test_feature_groups_are_exact(self) -> None:
        self.assertEqual(
            ABSOLUTE_FEATURE_COLUMNS,
            MODEL_FEATURE_COLUMNS[:9],
        )
        self.assertEqual(PERSONAL_CHANGE_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS[9:])
        self.assertEqual(COMBINED_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS)

    def test_no_forbidden_columns_in_groups(self) -> None:
        forbidden = set(AUDIT_ONLY_COLUMNS + IDENTIFIER_COLUMNS + [TARGET_COLUMN])
        self.assertTrue(all(forbidden.isdisjoint(group) for group in MODEL_FEATURE_GROUPS.values()))
        validate_feature_groups()

    def test_combined_features_have_no_duplicates(self) -> None:
        self.assertEqual(len(COMBINED_FEATURE_COLUMNS), len(set(COMBINED_FEATURE_COLUMNS)))

    def test_same_folds_are_used_for_all_models(self) -> None:
        for fold in range(1, 6):
            validation_counts = {
                int(row["validation_rows"])
                for row in self.result.fold_results
                if row["fold"] == fold
            }
            self.assertEqual(len(validation_counts), 1)

    def test_no_student_crosses_a_fold(self) -> None:
        for fold in self.result.folds:
            self.assertFalse(fold.train_students & fold.validation_students)

    def test_all_student_rows_stay_together(self) -> None:
        for fold in self.result.folds:
            validation_ids = {self.rows[index]["student_id"] for index in fold.validation_indices}
            for student in validation_ids:
                indices = [i for i, row in enumerate(self.rows) if row["student_id"] == student]
                self.assertTrue(all(index in fold.validation_indices for index in indices))

    def test_every_row_gets_one_oof_prediction(self) -> None:
        self.assertEqual(len(self.result.predictions), len(self.rows))
        keys = [(row["student_id"], row["observation_day"]) for row in self.result.predictions]
        self.assertEqual(len(keys), len(set(keys)))

    def test_both_classes_exist_in_every_validation_fold(self) -> None:
        for fold in self.result.folds:
            targets = {self.rows[index][TARGET_COLUMN] for index in fold.validation_indices}
            self.assertEqual(targets, {0, 1})

    def test_preprocessing_is_fitted_separately_per_fold_and_model(self) -> None:
        pipelines = self.result.fitted_pipelines
        self.assertEqual(len(pipelines), 15)
        self.assertEqual(len({id(pipeline) for pipeline in pipelines}), 15)
        self.assertTrue(all(hasattr(p.named_steps["imputer"], "statistics_") for p in pipelines))

    def test_all_fold_models_converge(self) -> None:
        self.assertTrue(all(row["converged"] for row in self.result.fold_results))
        self.assertTrue(all(int(row["iterations"]) < 1000 for row in self.result.fold_results))

    def test_missing_values_are_handled(self) -> None:
        self.assertEqual(len(self.result.predictions), 80)

    def test_probabilities_are_bounded(self) -> None:
        probability_names = [name for name in self.result.predictions[0] if name.endswith("_probability")]
        for row in self.result.predictions:
            self.assertTrue(all(0 <= float(row[name]) <= 1 for name in probability_names))

    def test_predictions_are_binary(self) -> None:
        prediction_names = [name for name in self.result.predictions[0] if name.endswith("_prediction")]
        for row in self.result.predictions:
            self.assertTrue(all(row[name] in {0, 1} for name in prediction_names))

    def test_fold_metrics_are_correct(self) -> None:
        metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 0, 1], [0.1, 0.8, 0.4, 0.9])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])

    def test_summary_mean_and_standard_deviation_are_correct(self) -> None:
        values = [
            float(row["f1"])
            for row in self.result.fold_results
            if row["model"] == "absolute"
        ]
        summary = self.result.summaries["absolute"]["f1"]
        self.assertAlmostEqual(summary["mean"], statistics.fmean(values))
        self.assertAlmostEqual(summary["standard_deviation"], statistics.stdev(values))

    def test_paired_differences_are_correct(self) -> None:
        paired = self.result.paired_differences["combined_minus_absolute"]["recall"]
        combined = {
            row["fold"]: float(row["recall"])
            for row in self.result.fold_results
            if row["model"] == "combined"
        }
        absolute = {
            row["fold"]: float(row["recall"])
            for row in self.result.fold_results
            if row["model"] == "absolute"
        }
        expected = [combined[fold] - absolute[fold] for fold in range(1, 6)]
        self.assertEqual(paired["fold_differences"], expected)

    def test_confusion_matrices_sum_to_validation_rows(self) -> None:
        for row in self.result.fold_results:
            self.assertEqual(
                sum(sum(part) for part in row["confusion_matrix"]),
                row["validation_rows"],
            )

    def test_experiment_is_reproducible(self) -> None:
        repeated = run_ablation_experiment(self.rows, 5, 42)
        self.assertEqual(self.result.predictions, repeated.predictions)
        self.assertEqual(self.result.fold_results, repeated.fold_results)

    def test_source_modeling_file_is_not_modified(self) -> None:
        path = Path("data/processed/modeling_table.csv")
        if not path.exists():
            self.skipTest("real modeling table is not present")
        before = (path.stat().st_size, path.stat().st_mtime_ns)
        self.assertTrue(path.read_bytes())
        self.assertEqual(before, (path.stat().st_size, path.stat().st_mtime_ns))


if __name__ == "__main__":
    unittest.main()
