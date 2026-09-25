"""Rolling seven-course-day features derived from daily OULAD summaries."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from learnpulse.features import DailyStudentActivity


@dataclass(frozen=True)
class RollingStudentActivity:
    """Seven-day activity features for one student on one course day."""

    module: str
    presentation: str
    student_id: int
    day: int
    window_complete: bool
    clicks_last_7_days: int
    active_days_last_7_days: int
    resources_last_7_days: int
    activity_types_last_7_days: int
    homepage_clicks_last_7_days: int
    forum_clicks_last_7_days: int
    content_clicks_last_7_days: int
    quiz_clicks_last_7_days: int
    days_since_last_activity: int | None

    def as_message(self) -> dict[str, str | int | None]:
        """Return a JSON-serializable representation of the rolling row."""
        return asdict(self)


def calculate_rolling_features(
    summaries: Iterable[DailyStudentActivity],
    start_day: int | None = None,
    end_day: int | None = None,
) -> list[RollingStudentActivity]:
    """Calculate trailing seven-day features without looking into the future.

    One row is emitted for every integer course day in each selected group.
    When bounds are omitted, each student's first and last observed activity
    days define its output range. Explicit bounds apply inclusively. The
    analysis-timeline boundary is the output start: a window is complete only
    on or after boundary + 6. Known earlier activity may still preserve the
    existing rolling values and inactivity gap, but it never makes a window
    before that boundary complete.
    """
    if start_day is not None and end_day is not None and start_day > end_day:
        raise ValueError("start_day cannot be greater than end_day")

    groups: dict[tuple[str, str, int], list[DailyStudentActivity]] = defaultdict(list)
    for summary in summaries:
        groups[(summary.module, summary.presentation, summary.student_id)].append(summary)

    results: list[RollingStudentActivity] = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda item: item.day)
        by_day = {item.day: item for item in ordered}
        output_start = start_day if start_day is not None else ordered[0].day
        output_end = end_day if end_day is not None else ordered[-1].day
        if output_start > output_end:
            continue

        # Begin at the first observation when it is earlier than the requested
        # range so days-since-last-activity retains its correct history.
        processing_start = min(output_start, ordered[0].day)
        window: deque[DailyStudentActivity] = deque()
        resource_counts: Counter[int] = Counter()
        activity_type_counts: Counter[str] = Counter()
        last_activity_day: int | None = None

        for day in range(processing_start, output_end + 1):
            cutoff = day - 6
            while window and window[0].day < cutoff:
                expired = window.popleft()
                resource_counts.subtract(expired.resource_ids)
                activity_type_counts.subtract(expired.activity_types)
                resource_counts += Counter()
                activity_type_counts += Counter()

            current = by_day.get(day)
            if current is not None:
                window.append(current)
                resource_counts.update(current.resource_ids)
                activity_type_counts.update(current.activity_types)
                last_activity_day = day

            if day < output_start:
                continue

            results.append(
                RollingStudentActivity(
                    module=key[0],
                    presentation=key[1],
                    student_id=key[2],
                    day=day,
                    window_complete=day >= output_start + 6,
                    clicks_last_7_days=sum(item.total_clicks for item in window),
                    active_days_last_7_days=len(window),
                    resources_last_7_days=len(resource_counts),
                    activity_types_last_7_days=len(activity_type_counts),
                    homepage_clicks_last_7_days=sum(item.homepage_clicks for item in window),
                    forum_clicks_last_7_days=sum(item.forum_clicks for item in window),
                    content_clicks_last_7_days=sum(item.content_clicks for item in window),
                    quiz_clicks_last_7_days=sum(item.quiz_clicks for item in window),
                    days_since_last_activity=(
                        None if last_activity_day is None else day - last_activity_day
                    ),
                )
            )

    return sorted(results, key=lambda item: (item.module, item.presentation, item.student_id, item.day))
