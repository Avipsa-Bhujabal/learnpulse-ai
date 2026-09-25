"""Read real daily interaction records from OULAD."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DailyLearningEvent:
    """A real OULAD student-resource activity summary for one course day."""

    module: str
    presentation: str
    student_id: int
    site_id: int
    day: int
    click_count: int
    activity_type: str

    def as_message(self) -> dict[str, str | int]:
        return asdict(self)


def load_activity_types(vle_path: Path) -> dict[tuple[str, str, int], str]:
    """Connect each site ID to a human-readable activity type."""
    lookup: dict[tuple[str, str, int], str] = {}
    with vle_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            key = (
                row["code_module"],
                row["code_presentation"],
                int(row["id_site"]),
            )
            lookup[key] = row["activity_type"]
    return lookup


def iter_daily_events(
    student_vle_path: Path,
    activity_types: dict[tuple[str, str, int], str],
    module: str,
    presentation: str,
    start_day: int | None = None,
    end_day: int | None = None,
) -> Iterator[DailyLearningEvent]:
    """Yield real records for one module-presentation without loading the CSV."""
    with student_vle_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row["code_module"] != module:
                continue
            if row["code_presentation"] != presentation:
                continue

            day = int(row["date"])
            if start_day is not None and day < start_day:
                continue
            if end_day is not None and day > end_day:
                continue

            site_id = int(row["id_site"])
            key = (module, presentation, site_id)
            yield DailyLearningEvent(
                module=module,
                presentation=presentation,
                student_id=int(row["id_student"]),
                site_id=site_id,
                day=day,
                click_count=int(row["sum_click"]),
                activity_type=activity_types.get(key, "unknown"),
            )


def load_replay_window(
    data_dir: Path,
    module: str,
    presentation: str,
    start_day: int | None = None,
    end_day: int | None = None,
) -> list[DailyLearningEvent]:
    """Load one course window and sort it into honest chronological order."""
    activity_types = load_activity_types(data_dir / "vle.csv")
    events = iter_daily_events(
        data_dir / "studentVle.csv",
        activity_types,
        module,
        presentation,
        start_day,
        end_day,
    )
    return sorted(events, key=lambda event: (event.day, event.student_id, event.site_id))

