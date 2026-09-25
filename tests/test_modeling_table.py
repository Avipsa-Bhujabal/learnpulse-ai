import tempfile
import unittest
from pathlib import Path

from learnpulse.modeling_table import (
    AUDIT_ONLY_COLUMNS,
    IDENTIFIER_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_modeling_table,
    duplicate_key_count,
    load_student_results,
    validate_model_feature_columns,
    write_modeling_csv,
)
from learnpulse.outcomes import FutureInactivityOutcome
from learnpulse.personal_baseline import PersonalBaseline
from learnpulse.rolling_features import RollingStudentActivity


def rolling(
    student: int = 1,
    day: int = 14,
    module: str = "AAA",
    presentation: str = "2013J",
    complete: bool = True,
    clicks: int = 20,
) -> RollingStudentActivity:
    return RollingStudentActivity(
        module, presentation, student, day, complete, clicks, 5, 4, 3, 2, 3, 10, 2, 0
    )


def baseline(
    student: int = 1,
    day: int = 14,
    module: str = "AAA",
    presentation: str = "2013J",
    ready: bool = True,
    average: float | None = 18.5,
) -> PersonalBaseline:
    return PersonalBaseline(
        module=module,
        presentation=presentation,
        student_id=student,
        day=day,
        window_complete=True,
        clicks_last_7_days=20,
        active_days_last_7_days=5,
        days_since_last_activity=0,
        historical_windows_available=8,
        baseline_ready=ready,
        historical_average_clicks=average,
        historical_standard_deviation_clicks=None,
        click_difference_from_personal_average=None if average is None else 20 - average,
        click_percentage_change_from_personal_average=None,
        click_z_score=None,
        historical_average_active_days=4.5,
        active_days_difference_from_personal_average=0.5,
        current_inactivity_gap=0,
        longest_previous_inactivity_gap=2,
    )


def outcome(
    student: int = 1,
    day: int = 14,
    module: str = "AAA",
    presentation: str = "2013J",
    inactive: bool | None = False,
    outcome_eligible: bool = True,
    prediction_eligible: bool = True,
    withdrawn: bool = False,
) -> FutureInactivityOutcome:
    return FutureInactivityOutcome(
        module=module,
        presentation=presentation,
        student_id=student,
        observation_day=day,
        future_window_start_day=day + 1,
        future_window_end_day=day + 14,
        future_window_days=14,
        future_activity_days=None if inactive is None else (0 if inactive else 2),
        future_clicks=None if inactive is None else (0 if inactive else 11),
        future_inactivity=inactive,
        official_withdrawal_day=day if withdrawn else None,
        officially_withdrawn_by_observation_day=withdrawn,
        officially_withdrawn_by_window_end=withdrawn,
        outcome_eligible=outcome_eligible,
        ineligibility_reason=None if outcome_eligible else "future_window_exceeds_course_end",
        prediction_eligible=prediction_eligible,
        prediction_ineligibility_reason=(
            None if prediction_eligible else "already_withdrawn_by_observation_day"
        ),
    )


def build(r=None, b=None, o=None, days=(14,)):
    return build_modeling_table(
        [rolling()] if r is None else r,
        [baseline()] if b is None else b,
        [outcome()] if o is None else o,
        {("AAA", "2013J", 1): "Pass"},
        days,
    )


class ModelingTableTests(unittest.TestCase):
    def test_records_join_on_complete_key(self) -> None:
        row = build().model_ready_rows[0]
        self.assertEqual([row[name] for name in IDENTIFIER_COLUMNS], ["AAA", "2013J", 1, 14])
        self.assertEqual(row["clicks_last_7_days"], 20)
        self.assertEqual(row["historical_average_clicks"], 18.5)

    def test_model_ready_keys_are_unique(self) -> None:
        result = build()
        self.assertEqual(duplicate_key_count(result.model_ready_rows), 0)

    def test_requested_observation_days_are_respected(self) -> None:
        result = build(
            [rolling(day=14), rolling(day=28)],
            [baseline(day=14), baseline(day=28)],
            [outcome(day=14), outcome(day=28)],
            (14, 28),
        )
        self.assertEqual([row["observation_day"] for row in result.model_ready_rows], [14, 28])

    def test_partial_window_is_excluded(self) -> None:
        result = build([rolling(complete=False)])
        self.assertIn("partial_rolling_window", result.excluded_rows[0]["exclusion_reasons"])

    def test_baseline_not_ready_is_excluded(self) -> None:
        result = build(b=[baseline(ready=False)])
        self.assertIn("baseline_not_ready", result.excluded_rows[0]["exclusion_reasons"])

    def test_incomplete_future_window_is_excluded(self) -> None:
        result = build(o=[outcome(inactive=None, outcome_eligible=False, prediction_eligible=False)])
        self.assertEqual(len(result.model_ready_rows), 0)
        self.assertIn("future_window_exceeds_course_end", result.excluded_rows[0]["exclusion_reasons"])

    def test_withdrawn_student_is_excluded(self) -> None:
        result = build(o=[outcome(prediction_eligible=False, withdrawn=True)])
        self.assertIn("withdrawn_by_observation_day", result.excluded_rows[0]["exclusion_reasons"])

    def test_eligible_future_active_row_is_retained(self) -> None:
        self.assertFalse(build(o=[outcome(inactive=False)]).model_ready_rows[0][TARGET_COLUMN])

    def test_eligible_future_inactive_row_is_retained(self) -> None:
        self.assertTrue(build(o=[outcome(inactive=True)]).model_ready_rows[0][TARGET_COLUMN])

    def test_final_result_is_audit_only(self) -> None:
        self.assertIn("final_result", AUDIT_ONLY_COLUMNS)
        self.assertNotIn("final_result", MODEL_FEATURE_COLUMNS)

    def test_withdrawal_columns_are_audit_only(self) -> None:
        names = {name for name in AUDIT_ONLY_COLUMNS if "withdraw" in name}
        self.assertTrue(names)
        self.assertTrue(names.isdisjoint(MODEL_FEATURE_COLUMNS))

    def test_future_activity_fields_are_audit_only(self) -> None:
        self.assertIn("future_activity_days", AUDIT_ONLY_COLUMNS)
        self.assertIn("future_clicks", AUDIT_ONLY_COLUMNS)
        self.assertNotIn("future_clicks", MODEL_FEATURE_COLUMNS)

    def test_target_is_not_a_model_feature(self) -> None:
        self.assertNotIn(TARGET_COLUMN, MODEL_FEATURE_COLUMNS)

    def test_forbidden_model_column_raises(self) -> None:
        with self.assertRaises(ValueError):
            validate_model_feature_columns([*MODEL_FEATURE_COLUMNS, "future_clicks"])

    def test_missing_statistics_remain_null(self) -> None:
        row = build(b=[baseline(average=None)]).model_ready_rows[0]
        self.assertIsNone(row["historical_average_clicks"])
        self.assertIsNone(row["historical_standard_deviation_clicks"])

    def test_modules_and_presentations_remain_separate(self) -> None:
        result = build_modeling_table(
            [rolling(), rolling(module="BBB"), rolling(presentation="2014J")],
            [baseline(), baseline(module="BBB"), baseline(presentation="2014J")],
            [outcome(), outcome(module="BBB"), outcome(presentation="2014J")],
            {},
            [14],
        )
        self.assertEqual(len(result.model_ready_rows), 3)

    def test_students_remain_separate(self) -> None:
        result = build_modeling_table(
            [rolling(1), rolling(2)], [baseline(1), baseline(2)], [outcome(1), outcome(2)], {}, [14]
        )
        self.assertEqual([row["student_id"] for row in result.model_ready_rows], [1, 2])

    def test_duplicate_outcome_keys_are_detected(self) -> None:
        with self.assertRaises(ValueError):
            build(o=[outcome(), outcome()])

    def test_negative_activity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build(r=[rolling(clicks=-1)])

    def test_output_writer_does_not_modify_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "studentInfo.csv"
            text = "code_module,code_presentation,id_student,final_result\nAAA,2013J,1,Pass\n"
            source.write_text(text, encoding="utf-8")
            results = load_student_results(source)
            write_modeling_csv(root / "modeling.csv", build().model_ready_rows)
            self.assertEqual(results[("AAA", "2013J", 1)], "Pass")
            self.assertEqual(source.read_text(encoding="utf-8"), text)

    def test_feature_and_outcome_inputs_are_not_mutated(self) -> None:
        rolling_rows = [rolling()]
        baseline_rows = [baseline()]
        outcome_rows = [outcome()]
        snapshots = (list(rolling_rows), list(baseline_rows), list(outcome_rows))
        build(rolling_rows, baseline_rows, outcome_rows)
        self.assertEqual((rolling_rows, baseline_rows, outcome_rows), snapshots)


if __name__ == "__main__":
    unittest.main()
