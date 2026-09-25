"""Tests for leakage-safe transparent logistic explanations."""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path

from learnpulse.ablation_experiment import PERSONAL_CHANGE_FEATURE_COLUMNS
from learnpulse.explainability import (
    EXPLANATION_DAYS,
    feature_matrix,
    load_fold_assignments,
    run_explainability_experiment,
    select_representative_case,
    summarize_coefficients,
    transformed_feature_names,
)


def _rows():
    rows = []
    for student in range(1, 17):
        fold = (student - 1) % 4 + 1
        for day in EXPLANATION_DAYS:
            positive = int((((student - 1) // 4) + (0 if day == 28 else 1)) % 4 == 0)
            base = float(student + day / 10)
            row = {
                "module": "AAA",
                "presentation": "2013J",
                "student_id": student,
                "observation_day": day,
                "future_inactivity": positive,
            }
            for index, feature in enumerate(PERSONAL_CHANGE_FEATURE_COLUMNS):
                row[feature] = base + index if feature not in {
                    "historical_standard_deviation_clicks", "click_z_score"
                } else (None if student % 3 == 0 else base / (index + 1))
            rows.append(row)
    return rows, {student: (student - 1) % 4 + 1 for student in range(1, 17)}


class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.assignments = _rows()
        cls.result = run_explainability_experiment(
            cls.rows, cls.assignments, [28, 56], random_seed=42, permutation_repeats=2
        )

    def test_fold_assignments_are_reused(self):
        for prediction in self.result.predictions:
            self.assertEqual(prediction["fold"], self.assignments[prediction["student_id"]])

    def test_no_student_is_explained_by_training_model(self):
        self.assertTrue(all(not row["student_was_in_training"] for row in self.result.predictions))

    def test_only_days_28_and_56_are_analyzed(self):
        self.assertEqual({row["observation_day"] for row in self.result.predictions}, {28, 56})
        with self.assertRaises(ValueError):
            run_explainability_experiment(self.rows, self.assignments, [14, 28], permutation_repeats=1)

    def test_forbidden_and_audit_columns_are_excluded(self):
        forbidden = {"student_id", "observation_day", "future_inactivity", "future_clicks", "final_result"}
        self.assertFalse(forbidden & set(PERSONAL_CHANGE_FEATURE_COLUMNS))

    def test_transformed_names_match_coefficients(self):
        for pipeline in self.result.fitted_pipelines.values():
            self.assertEqual(len(transformed_feature_names(pipeline)), pipeline.named_steps["classifier"].coef_.shape[1])

    def test_missing_indicators_have_separate_names(self):
        names = transformed_feature_names(next(iter(self.result.fitted_pipelines.values())))
        self.assertIn("missing_indicator__click_z_score", names)
        self.assertIn("click_z_score", names)

    def test_coefficient_mean_and_standard_deviation(self):
        rows = [
            {"observation_day": 28, "fold": fold, "feature": "x", "standardized_coefficient": value}
            for fold, value in enumerate((1.0, 2.0, 3.0), start=1)
        ]
        summary = summarize_coefficients(rows)[0]
        self.assertEqual(summary["mean_standardized_coefficient"], 2.0)
        self.assertEqual(summary["standard_deviation_across_folds"], 1.0)

    def test_sign_agreement(self):
        rows = [
            {"observation_day": 28, "fold": fold, "feature": "x", "standardized_coefficient": value}
            for fold, value in enumerate((1.0, -1.0), start=1)
        ]
        self.assertFalse(summarize_coefficients(rows)[0]["sign_agreement"])

    def test_permutation_importance_uses_validation_only(self):
        self.assertTrue(all(row["data_partition"] == "validation" for row in self.result.permutation_rows))

    def test_permutation_importance_uses_pr_auc(self):
        self.assertTrue(all(row["scoring"] == "average_precision" for row in self.result.permutation_rows))

    def test_individual_contributions_sum_to_decision(self):
        available = [row for row in self.result.individual_explanations if row["available"]]
        self.assertTrue(available)
        self.assertTrue(all(row["contribution_sum_verification"] if "contribution_sum_verification" in row else row["contribution_sum_verified"] for row in available))
        self.assertTrue(all(abs(row["contribution_sum_difference"]) < 1e-9 for row in available))

    def test_contribution_ordering(self):
        for row in self.result.individual_explanations:
            if not row["available"]:
                continue
            increasing = [item["contribution"] for item in row["top_increasing_contributions"]]
            decreasing = [item["contribution"] for item in row["top_decreasing_contributions"]]
            self.assertEqual(increasing, sorted(increasing, reverse=True))
            self.assertEqual(decreasing, sorted(decreasing))

    def _assert_case(self, actual, predicted):
        predictions = [
            {"student_id": 1, "actual_future_inactivity": actual, "predicted_future_inactivity": predicted, "model_output": 0.6},
            {"student_id": 2, "actual_future_inactivity": 1 - actual, "predicted_future_inactivity": predicted, "model_output": 0.4},
        ]
        chosen = select_representative_case(predictions, actual, predicted)
        self.assertEqual((chosen["actual_future_inactivity"], chosen["predicted_future_inactivity"]), (actual, predicted))

    def test_true_positive_selection(self): self._assert_case(1, 1)
    def test_false_positive_selection(self): self._assert_case(0, 1)
    def test_false_negative_selection(self): self._assert_case(1, 0)
    def test_true_negative_selection(self): self._assert_case(0, 0)

    def test_probabilities_are_bounded(self):
        self.assertTrue(all(0 <= row["model_output"] <= 1 for row in self.result.predictions))

    def test_results_are_reproducible(self):
        repeated = run_explainability_experiment(self.rows, self.assignments, [28, 56], 42, 2)
        self.assertEqual(self.result.predictions, repeated.predictions)
        self.assertEqual(self.result.permutation_rows, repeated.permutation_rows)

    def test_fold_loader_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.csv"
            path.write_text("student_id,fold\n1,1\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_fold_assignments(path)

    def test_missing_values_are_processed(self):
        matrix = feature_matrix(self.rows)
        self.assertTrue(any(math.isnan(value) for row in matrix for value in row))
        self.assertEqual(len(self.result.predictions), len(self.rows))

    def test_original_modeling_table_is_not_modified(self):
        path = Path("data/processed/modeling_table.csv")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        load_fold_assignments(Path("experiments/common_cohort_student_folds.csv"))
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
