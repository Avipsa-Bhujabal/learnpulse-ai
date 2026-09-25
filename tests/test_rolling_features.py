import unittest

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import DailyLearningEvent
from learnpulse.rolling_features import calculate_rolling_features


def make_event(
    day: int,
    site_id: int = 10,
    activity_type: str = "homepage",
    clicks: int = 1,
    student_id: int = 1,
    module: str = "AAA",
    presentation: str = "2013J",
) -> DailyLearningEvent:
    return DailyLearningEvent(
        module, presentation, student_id, site_id, day, clicks, activity_type
    )


def rolling(events: list[DailyLearningEvent], start: int | None = None, end: int | None = None):
    return calculate_rolling_features(aggregate_daily_activity(events), start, end)


class RollingFeatureTests(unittest.TestCase):
    def test_window_complete_uses_requested_timeline_boundary(self) -> None:
        rows = rolling([make_event(1), make_event(10)], 1, 10)
        self.assertEqual(
            [row.window_complete for row in rows],
            [False, False, False, False, False, False, True, True, True, True],
        )

    def test_student_active_every_day(self) -> None:
        rows = rolling([make_event(day, clicks=day) for day in range(1, 9)])
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[6].active_days_last_7_days, 7)
        self.assertEqual(rows[7].clicks_last_7_days, sum(range(2, 9)))
        self.assertTrue(all(row.days_since_last_activity == 0 for row in rows))

    def test_missing_days_are_inactive(self) -> None:
        rows = rolling([make_event(1), make_event(4)], 1, 4)
        self.assertEqual([row.day for row in rows], [1, 2, 3, 4])
        self.assertEqual([row.active_days_last_7_days for row in rows], [1, 1, 1, 2])
        self.assertEqual([row.days_since_last_activity for row in rows], [0, 1, 2, 0])

    def test_return_after_long_inactivity_gap(self) -> None:
        rows = rolling([make_event(1, clicks=5), make_event(12, clicks=8)], 8, 12)
        self.assertEqual(rows[0].clicks_last_7_days, 0)
        self.assertEqual(rows[0].days_since_last_activity, 7)
        self.assertEqual(rows[-1].clicks_last_7_days, 8)
        self.assertEqual(rows[-1].days_since_last_activity, 0)

    def test_seven_day_window_boundaries_are_inclusive(self) -> None:
        rows = rolling([make_event(3, clicks=100), make_event(4, clicks=4), make_event(10, clicks=10)], 10, 10)
        self.assertEqual(rows[0].clicks_last_7_days, 14)
        self.assertEqual(rows[0].active_days_last_7_days, 2)

    def test_future_activity_does_not_affect_earlier_row(self) -> None:
        rows = rolling([make_event(1, clicks=2), make_event(5, clicks=99)], 1, 5)
        self.assertEqual(rows[0].clicks_last_7_days, 2)
        self.assertEqual(rows[3].clicks_last_7_days, 2)

    def test_resources_are_distinct_across_days(self) -> None:
        rows = rolling([make_event(1, site_id=10), make_event(2, site_id=10), make_event(3, site_id=11)])
        self.assertEqual(rows[-1].resources_last_7_days, 2)

    def test_activity_types_are_distinct_across_days(self) -> None:
        rows = rolling(
            [make_event(1, activity_type="homepage"), make_event(2, activity_type="homepage"), make_event(3, activity_type="forumng")]
        )
        self.assertEqual(rows[-1].activity_types_last_7_days, 2)

    def test_students_remain_separate(self) -> None:
        rows = rolling([make_event(1, clicks=2, student_id=1), make_event(1, clicks=9, student_id=2)])
        self.assertEqual([(row.student_id, row.clicks_last_7_days) for row in rows], [(1, 2), (2, 9)])

    def test_modules_and_presentations_remain_separate(self) -> None:
        rows = rolling(
            [
                make_event(1, clicks=2, module="AAA", presentation="2013J"),
                make_event(1, clicks=5, module="BBB", presentation="2013J"),
                make_event(1, clicks=8, module="AAA", presentation="2014J"),
            ]
        )
        self.assertEqual(
            [(row.module, row.presentation, row.clicks_last_7_days) for row in rows],
            [("AAA", "2013J", 2), ("AAA", "2014J", 8), ("BBB", "2013J", 5)],
        )

    def test_explicit_start_before_first_activity_uses_null(self) -> None:
        rows = rolling([make_event(3)], 1, 3)
        self.assertIsNone(rows[0].days_since_last_activity)
        self.assertIsNone(rows[1].days_since_last_activity)
        self.assertEqual(rows[2].days_since_last_activity, 0)
        self.assertEqual(rows[0].active_days_last_7_days, 0)


if __name__ == "__main__":
    unittest.main()
