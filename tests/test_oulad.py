import tempfile
import unittest
from pathlib import Path

from learnpulse.oulad import iter_daily_events, load_activity_types


class OuladReaderTests(unittest.TestCase):
    def test_reader_joins_activity_type_and_filters_course(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "vle.csv").write_text(
                "id_site,code_module,code_presentation,activity_type,week_from,week_to\n"
                "10,AAA,2013J,quiz,?,?\n",
                encoding="utf-8",
            )
            (root / "studentVle.csv").write_text(
                "code_module,code_presentation,id_student,id_site,date,sum_click\n"
                "AAA,2013J,25,10,3,7\n"
                "BBB,2013J,26,10,3,2\n",
                encoding="utf-8",
            )

            lookup = load_activity_types(root / "vle.csv")
            events = list(
                iter_daily_events(
                    root / "studentVle.csv", lookup, "AAA", "2013J"
                )
            )

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].activity_type, "quiz")
            self.assertEqual(events[0].click_count, 7)


if __name__ == "__main__":
    unittest.main()

