"""Daily feature aggregation for OULAD learning activity."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from learnpulse.oulad import DailyLearningEvent


@dataclass(frozen=True)
class DailyStudentActivity:
    """Aggregated activity for one student on one module-presentation day."""

    module: str
    presentation: str
    student_id: int
    day: int
    total_clicks: int
    number_of_resources_visited: int
    number_of_activity_types_used: int
    homepage_clicks: int
    forum_clicks: int
    content_clicks: int
    quiz_clicks: int
    resource_ids: frozenset[int] = field(repr=False)
    activity_types: frozenset[str] = field(repr=False)

    def as_message(self) -> dict[str, str | int]:
        """Return a JSON-serializable representation of the summary."""
        return {
            "module": self.module,
            "presentation": self.presentation,
            "student_id": self.student_id,
            "day": self.day,
            "total_clicks": self.total_clicks,
            "number_of_resources_visited": self.number_of_resources_visited,
            "number_of_activity_types_used": self.number_of_activity_types_used,
            "homepage_clicks": self.homepage_clicks,
            "forum_clicks": self.forum_clicks,
            "content_clicks": self.content_clicks,
            "quiz_clicks": self.quiz_clicks,
        }


@dataclass
class _Accumulator:
    """Mutable internal state used while combining event records."""

    total_clicks: int
    resource_ids: set[int]
    activity_types: set[str]
    homepage_clicks: int = 0
    forum_clicks: int = 0
    content_clicks: int = 0
    quiz_clicks: int = 0


def aggregate_daily_activity(
    events: Iterable[DailyLearningEvent],
) -> list[DailyStudentActivity]:
    """Combine OULAD records by module, presentation, student, and day.

    The named click categories follow OULAD's activity vocabulary:
    ``homepage`` is homepage activity, ``forumng`` is forum activity,
    ``oucontent`` is course content, and both ``quiz`` and ``externalquiz``
    count as quiz activity. Other activity types still contribute to totals,
    resource counts, and activity-type counts.
    """
    grouped: dict[tuple[str, str, int, int], _Accumulator] = {}

    for event in events:
        key = (event.module, event.presentation, event.student_id, event.day)
        accumulator = grouped.get(key)
        if accumulator is None:
            accumulator = _Accumulator(0, set(), set())
            grouped[key] = accumulator

        accumulator.total_clicks += event.click_count
        accumulator.resource_ids.add(event.site_id)
        accumulator.activity_types.add(event.activity_type)

        if event.activity_type == "homepage":
            accumulator.homepage_clicks += event.click_count
        elif event.activity_type == "forumng":
            accumulator.forum_clicks += event.click_count
        elif event.activity_type == "oucontent":
            accumulator.content_clicks += event.click_count
        elif event.activity_type in {"quiz", "externalquiz"}:
            accumulator.quiz_clicks += event.click_count

    summaries = [
        DailyStudentActivity(
            module=key[0],
            presentation=key[1],
            student_id=key[2],
            day=key[3],
            total_clicks=value.total_clicks,
            number_of_resources_visited=len(value.resource_ids),
            number_of_activity_types_used=len(value.activity_types),
            homepage_clicks=value.homepage_clicks,
            forum_clicks=value.forum_clicks,
            content_clicks=value.content_clicks,
            quiz_clicks=value.quiz_clicks,
            resource_ids=frozenset(value.resource_ids),
            activity_types=frozenset(value.activity_types),
        )
        for key, value in grouped.items()
    ]
    return sorted(
        summaries,
        key=lambda item: (
            item.module,
            item.presentation,
            item.student_id,
            item.day,
        ),
    )
