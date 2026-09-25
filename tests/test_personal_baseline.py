import math
import unittest

from learnpulse.personal_baseline import calculate_personal_baselines
from learnpulse.rolling_features import RollingStudentActivity


def rolling_row(
    day: int,
    clicks: int,
    active_days: int = 1,
    inactivity: int | None = 0,
    student_id: int = 1,
    module: str = "AAA",
    presentation: str = "2013J",
    window_complete: bool = True,
) -> RollingStudentActivity:
    return RollingStudentActivity(
        module=module,
        presentation=presentation,
        student_id=student_id,
        day=day,
        window_complete=window_complete,
        clicks_last_7_days=clicks,
        active_days_last_7_days=active_days,
        resources_last_7_days=1,
        activity_types_last_7_days=1,
        homepage_clicks_last_7_days=clicks,
        forum_clicks_last_7_days=0,
        content_clicks_last_7_days=0,
        quiz_clicks_last_7_days=0,
        days_since_last_activity=inactivity,
    )


class PersonalBaselineTests(unittest.TestCase):
    def test_partial_windows_do_not_enter_baseline(self) -> None:
        rows = calculate_personal_baselines(
            [
                rolling_row(1, 100, window_complete=False),
                rolling_row(2, 200, window_complete=False),
                rolling_row(3, 30, window_complete=True),
                rolling_row(4, 40, window_complete=True),
            ]
        )
        self.assertEqual(rows[2].historical_windows_available, 0)
        self.assertEqual(rows[3].historical_windows_available, 1)
        self.assertEqual(rows[3].historical_average_clicks, 30)

    def test_first_complete_window_has_no_complete_history(self) -> None:
        result = calculate_personal_baselines(
            [
                rolling_row(1, 10, window_complete=False),
                rolling_row(2, 20, window_complete=True),
            ]
        )[1]
        self.assertTrue(result.window_complete)
        self.assertEqual(result.historical_windows_available, 0)
        self.assertIsNone(result.historical_average_clicks)

    def test_readiness_counts_only_earlier_complete_windows(self) -> None:
        rows = calculate_personal_baselines(
            [
                rolling_row(1, 1, window_complete=False),
                rolling_row(2, 2, window_complete=True),
                rolling_row(3, 3, window_complete=True),
                rolling_row(4, 4, window_complete=True),
            ],
            minimum_history=2,
        )
        self.assertEqual([row.baseline_ready for row in rows], [False, False, False, True])
        self.assertEqual(rows[-1].historical_windows_available, 2)

    def test_first_row_has_zero_historical_windows(self) -> None:
        first = calculate_personal_baselines([rolling_row(1, 10)])[0]
        self.assertEqual(first.historical_windows_available, 0)
        self.assertIsNone(first.historical_average_clicks)

    def test_current_row_is_not_in_its_own_baseline(self) -> None:
        rows = calculate_personal_baselines([rolling_row(1, 10), rolling_row(2, 30)])
        self.assertEqual(rows[1].historical_average_clicks, 10)

    def test_future_activity_does_not_change_earlier_baseline(self) -> None:
        initial = calculate_personal_baselines([rolling_row(1, 10), rolling_row(2, 20)])
        extended = calculate_personal_baselines(
            [rolling_row(1, 10), rolling_row(2, 20), rolling_row(3, 999)]
        )
        self.assertEqual(initial, extended[:2])

    def test_sudden_decrease_is_negative(self) -> None:
        result = calculate_personal_baselines(
            [rolling_row(1, 100), rolling_row(2, 100), rolling_row(3, 40)]
        )[-1]
        self.assertEqual(result.click_difference_from_personal_average, -60)
        self.assertEqual(result.click_percentage_change_from_personal_average, -60)

    def test_stable_behavior_matches_personal_average(self) -> None:
        result = calculate_personal_baselines(
            [rolling_row(1, 50), rolling_row(2, 50), rolling_row(3, 50)]
        )[-1]
        self.assertEqual(result.click_difference_from_personal_average, 0)
        self.assertEqual(result.active_days_difference_from_personal_average, 0)

    def test_percentage_change_is_null_for_zero_average(self) -> None:
        result = calculate_personal_baselines([rolling_row(1, 0), rolling_row(2, 5)])[-1]
        self.assertIsNone(result.click_percentage_change_from_personal_average)

    def test_z_score_is_null_with_insufficient_standard_deviation(self) -> None:
        result = calculate_personal_baselines([rolling_row(1, 10), rolling_row(2, 20)])[-1]
        self.assertIsNone(result.historical_standard_deviation_clicks)
        self.assertIsNone(result.click_z_score)

    def test_z_score_is_null_when_standard_deviation_is_zero(self) -> None:
        result = calculate_personal_baselines(
            [rolling_row(1, 10), rolling_row(2, 10), rolling_row(3, 20)]
        )[-1]
        self.assertEqual(result.historical_standard_deviation_clicks, 0)
        self.assertIsNone(result.click_z_score)

    def test_sample_standard_deviation_is_correct(self) -> None:
        result = calculate_personal_baselines(
            [rolling_row(1, 10), rolling_row(2, 20), rolling_row(3, 30)]
        )[-1]
        self.assertAlmostEqual(result.historical_standard_deviation_clicks, math.sqrt(50))
        self.assertAlmostEqual(result.click_z_score, 15 / math.sqrt(50))

    def test_baseline_ready_uses_required_number_of_earlier_rows(self) -> None:
        rows = calculate_personal_baselines(
            [rolling_row(1, 1), rolling_row(2, 2), rolling_row(3, 3)], minimum_history=2
        )
        self.assertEqual([row.baseline_ready for row in rows], [False, False, True])

    def test_current_inactivity_gap_uses_current_value(self) -> None:
        result = calculate_personal_baselines([rolling_row(1, 1, inactivity=4)])[0]
        self.assertEqual(result.current_inactivity_gap, 4)

    def test_longest_previous_gap_excludes_current_row(self) -> None:
        rows = calculate_personal_baselines(
            [rolling_row(1, 1, inactivity=2), rolling_row(2, 1, inactivity=9)]
        )
        self.assertIsNone(rows[0].longest_previous_inactivity_gap)
        self.assertEqual(rows[1].longest_previous_inactivity_gap, 2)

    def test_missing_inactivity_values_are_ignored(self) -> None:
        rows = calculate_personal_baselines(
            [rolling_row(1, 1, inactivity=None), rolling_row(2, 1, inactivity=None), rolling_row(3, 1, inactivity=0)]
        )
        self.assertIsNone(rows[2].longest_previous_inactivity_gap)

    def test_students_remain_separate(self) -> None:
        rows = calculate_personal_baselines(
            [rolling_row(1, 10, student_id=1), rolling_row(1, 90, student_id=2)]
        )
        self.assertEqual([row.historical_windows_available for row in rows], [0, 0])

    def test_modules_and_presentations_remain_separate(self) -> None:
        rows = calculate_personal_baselines(
            [
                rolling_row(1, 10, module="AAA", presentation="2013J"),
                rolling_row(1, 20, module="AAA", presentation="2014J"),
                rolling_row(1, 30, module="BBB", presentation="2013J"),
            ]
        )
        self.assertTrue(all(row.historical_windows_available == 0 for row in rows))

    def test_results_are_chronological(self) -> None:
        rows = calculate_personal_baselines(
            [rolling_row(3, 3), rolling_row(1, 1), rolling_row(2, 2)]
        )
        self.assertEqual([row.day for row in rows], [1, 2, 3])

    def test_internal_calculations_are_not_prematurely_rounded(self) -> None:
        result = calculate_personal_baselines(
            [rolling_row(1, 1), rolling_row(2, 1), rolling_row(3, 2), rolling_row(4, 4)]
        )[-1]
        self.assertAlmostEqual(result.historical_average_clicks, 4 / 3)
        self.assertNotEqual(result.historical_average_clicks, round(4 / 3, 4))


if __name__ == "__main__":
    unittest.main()
