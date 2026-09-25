import tempfile
import unittest
from pathlib import Path

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import DailyLearningEvent
from learnpulse.outcomes import (
    calculate_future_inactivity_outcomes,
    final_course_day,
    load_course_lengths,
    load_withdrawal_days,
)


def event(
    day: int,
    clicks: int = 1,
    student_id: int = 1,
    module: str = "AAA",
    presentation: str = "2013J",
    site_id: int | None = None,
) -> DailyLearningEvent:
    return DailyLearningEvent(
        module, presentation, student_id, site_id or day + 1000, day, clicks, "homepage"
    )


def outcomes(
    events: list[DailyLearningEvent],
    observation_days: list[tuple[str, str, int, int]] | None = None,
    length: int | None = 100,
    withdrawals: dict[tuple[str, str, int], int | None] | None = None,
    window: int = 14,
):
    course_lengths = {} if length is None else {("AAA", "2013J"): length}
    return calculate_future_inactivity_outcomes(
        aggregate_daily_activity(events),
        observation_days or [("AAA", "2013J", 1, 10)],
        course_lengths,
        withdrawals,
        window,
    )


class FutureInactivityOutcomeTests(unittest.TestCase):
    def test_observation_day_is_excluded(self) -> None:
        result = outcomes([event(10, 9)], window=3)[0]
        self.assertTrue(result.future_inactivity)
        self.assertEqual(result.future_clicks, 0)

    def test_window_starts_at_t_plus_one(self) -> None:
        result = outcomes([], window=3)[0]
        self.assertEqual(result.future_window_start_day, 11)

    def test_window_ends_at_t_plus_n(self) -> None:
        result = outcomes([], window=14)[0]
        self.assertEqual(result.future_window_end_day, 24)

    def test_observation_activity_does_not_prevent_inactivity(self) -> None:
        self.assertTrue(outcomes([event(10, 500)], window=2)[0].future_inactivity)

    def test_first_future_day_makes_label_false(self) -> None:
        self.assertFalse(outcomes([event(11)], window=3)[0].future_inactivity)

    def test_final_future_day_makes_label_false(self) -> None:
        self.assertFalse(outcomes([event(13)], window=3)[0].future_inactivity)

    def test_activity_after_window_does_not_affect_label(self) -> None:
        self.assertTrue(outcomes([event(14)], window=3)[0].future_inactivity)

    def test_zero_future_activity_is_inactive(self) -> None:
        result = outcomes([], window=3)[0]
        self.assertTrue(result.future_inactivity)
        self.assertEqual(result.future_activity_days, 0)

    def test_future_click_total_is_exact(self) -> None:
        result = outcomes([event(11, 4), event(12, 7), event(20, 99)], window=3)[0]
        self.assertEqual(result.future_clicks, 11)

    def test_future_active_days_are_distinct_days(self) -> None:
        result = outcomes(
            [event(11, 2, site_id=1), event(11, 3, site_id=2), event(12, 4)], window=3
        )[0]
        self.assertEqual(result.future_activity_days, 2)

    def test_complete_course_window_is_eligible(self) -> None:
        result = outcomes([], length=14, window=3)[0]
        self.assertTrue(result.outcome_eligible)

    def test_end_of_course_window_is_ineligible(self) -> None:
        result = outcomes([], length=13, window=3)[0]
        self.assertFalse(result.outcome_eligible)
        self.assertIsNone(result.future_inactivity)
        self.assertEqual(result.ineligibility_reason, "future_window_exceeds_course_end")

    def test_missing_course_metadata_is_safe(self) -> None:
        result = outcomes([], length=None)[0]
        self.assertFalse(result.outcome_eligible)
        self.assertEqual(result.ineligibility_reason, "missing_course_metadata")
        self.assertIsNone(result.future_clicks)

    def test_missing_withdrawal_is_null(self) -> None:
        self.assertIsNone(outcomes([])[0].official_withdrawal_day)

    def test_withdrawal_by_window_end_is_marked(self) -> None:
        result = outcomes([], withdrawals={("AAA", "2013J", 1): 20})[0]
        self.assertTrue(result.officially_withdrawn_by_window_end)

    def test_withdrawal_after_window_end_is_not_marked(self) -> None:
        result = outcomes([], withdrawals={("AAA", "2013J", 1): 25})[0]
        self.assertFalse(result.officially_withdrawn_by_window_end)

    def test_withdrawal_does_not_define_inactivity(self) -> None:
        result = outcomes(
            [event(11)], withdrawals={("AAA", "2013J", 1): 11}
        )[0]
        self.assertTrue(result.officially_withdrawn_by_window_end)
        self.assertFalse(result.future_inactivity)

    def test_withdrawal_before_observation_makes_prediction_ineligible(self) -> None:
        result = outcomes([], withdrawals={("AAA", "2013J", 1): 9})[0]
        self.assertTrue(result.officially_withdrawn_by_observation_day)
        self.assertFalse(result.prediction_eligible)
        self.assertEqual(
            result.prediction_ineligibility_reason,
            "already_withdrawn_by_observation_day",
        )
        self.assertTrue(result.outcome_eligible)

    def test_withdrawal_on_observation_makes_prediction_ineligible(self) -> None:
        result = outcomes([], withdrawals={("AAA", "2013J", 1): 10})[0]
        self.assertTrue(result.officially_withdrawn_by_observation_day)
        self.assertFalse(result.prediction_eligible)

    def test_withdrawal_after_observation_does_not_make_prediction_ineligible(self) -> None:
        result = outcomes([], withdrawals={("AAA", "2013J", 1): 11})[0]
        self.assertFalse(result.officially_withdrawn_by_observation_day)
        self.assertTrue(result.prediction_eligible)

    def test_future_window_withdrawal_remains_separate_from_activity(self) -> None:
        result = outcomes(
            [event(12)], withdrawals={("AAA", "2013J", 1): 11}
        )[0]
        self.assertTrue(result.prediction_eligible)
        self.assertTrue(result.officially_withdrawn_by_window_end)
        self.assertFalse(result.future_inactivity)

    def test_non_withdrawn_complete_row_is_prediction_eligible(self) -> None:
        result = outcomes([])[0]
        self.assertTrue(result.prediction_eligible)
        self.assertIsNone(result.prediction_ineligibility_reason)

    def test_length_268_ends_on_day_267(self) -> None:
        self.assertEqual(final_course_day(268), 267)

    def test_window_ending_on_last_day_is_eligible_and_next_day_is_not(self) -> None:
        exact = outcomes([], [("AAA", "2013J", 1, 253)], length=268, window=14)[0]
        beyond = outcomes([], [("AAA", "2013J", 1, 254)], length=268, window=14)[0]
        self.assertEqual(exact.future_window_end_day, 267)
        self.assertTrue(exact.outcome_eligible)
        self.assertTrue(exact.prediction_eligible)
        self.assertEqual(beyond.future_window_end_day, 268)
        self.assertFalse(beyond.outcome_eligible)
        self.assertFalse(beyond.prediction_eligible)

    def test_students_remain_separate(self) -> None:
        result = outcomes(
            [event(11, student_id=1)],
            [("AAA", "2013J", 1, 10), ("AAA", "2013J", 2, 10)],
        )
        self.assertFalse(result[0].future_inactivity)
        self.assertTrue(result[1].future_inactivity)

    def test_modules_and_presentations_remain_separate(self) -> None:
        result = calculate_future_inactivity_outcomes(
            [
                *aggregate_daily_activity([event(11)]),
                *aggregate_daily_activity([event(11, module="BBB")]),
                *aggregate_daily_activity([event(11, presentation="2014J")]),
            ],
            [
                ("AAA", "2013J", 1, 10),
                ("AAA", "2014J", 1, 10),
                ("BBB", "2013J", 1, 10),
            ],
            {("AAA", "2013J"): 100, ("AAA", "2014J"): 100, ("BBB", "2013J"): 100},
            future_window_days=2,
        )
        self.assertTrue(all(row.future_activity_days == 1 for row in result))

    def test_results_are_chronological(self) -> None:
        result = outcomes(
            [],
            [("AAA", "2013J", 1, 3), ("AAA", "2013J", 1, 1), ("AAA", "2013J", 1, 2)],
        )
        self.assertEqual([row.observation_day for row in result], [1, 2, 3])

    def test_feature_inputs_are_not_mutated(self) -> None:
        daily = aggregate_daily_activity([event(11, 4)])
        before = list(daily)
        calculate_future_inactivity_outcomes(
            daily, [("AAA", "2013J", 1, 10)], {("AAA", "2013J"): 100}
        )
        self.assertEqual(daily, before)

    def test_metadata_loaders_never_modify_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            courses = root / "courses.csv"
            registrations = root / "studentRegistration.csv"
            courses_text = (
                "code_module,code_presentation,module_presentation_length\nAAA,2013J,268\n"
            )
            registrations_text = (
                "code_module,code_presentation,id_student,date_registration,date_unregistration\n"
                "AAA,2013J,1,-10,?\nAAA,2013J,2,-9,40\n"
            )
            courses.write_text(courses_text, encoding="utf-8")
            registrations.write_text(registrations_text, encoding="utf-8")

            self.assertEqual(load_course_lengths(courses)[("AAA", "2013J")], 268)
            loaded = load_withdrawal_days(registrations)
            self.assertIsNone(loaded[("AAA", "2013J", 1)])
            self.assertEqual(loaded[("AAA", "2013J", 2)], 40)
            self.assertEqual(courses.read_text(encoding="utf-8"), courses_text)
            self.assertEqual(registrations.read_text(encoding="utf-8"), registrations_text)


if __name__ == "__main__":
    unittest.main()
