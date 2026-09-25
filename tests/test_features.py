import unittest

from learnpulse.features import aggregate_daily_activity
from learnpulse.oulad import DailyLearningEvent


def event(site_id: int, clicks: int, activity_type: str, day: int = 4) -> DailyLearningEvent:
    """Create a compact event fixture for aggregation tests."""
    return DailyLearningEvent("AAA", "2013J", 28400, site_id, day, clicks, activity_type)


class DailyActivityAggregationTests(unittest.TestCase):
    def test_combines_multiple_records_for_the_same_student_and_day(self) -> None:
        summaries = aggregate_daily_activity(
            [
                event(10, 3, "homepage"),
                event(11, 5, "forumng"),
                event(11, 2, "forumng"),
                event(12, 7, "oucontent"),
                event(13, 4, "quiz"),
                event(14, 6, "externalquiz"),
                event(15, 1, "resource"),
            ]
        )

        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertEqual(summary.total_clicks, 28)
        self.assertEqual(summary.number_of_resources_visited, 6)
        self.assertEqual(summary.number_of_activity_types_used, 6)
        self.assertEqual(summary.homepage_clicks, 3)
        self.assertEqual(summary.forum_clicks, 7)
        self.assertEqual(summary.content_clicks, 7)
        self.assertEqual(summary.quiz_clicks, 10)

    def test_keeps_days_separate_and_returns_chronological_order(self) -> None:
        summaries = aggregate_daily_activity(
            [event(10, 2, "homepage", day=8), event(11, 3, "forumng", day=2)]
        )

        self.assertEqual([summary.day for summary in summaries], [2, 8])
        self.assertEqual([summary.total_clicks for summary in summaries], [3, 2])


if __name__ == "__main__":
    unittest.main()
