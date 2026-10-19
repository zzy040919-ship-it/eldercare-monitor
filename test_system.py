import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples import pose, sequence
from logic import Config, Detector, features
from storage import Store


class SystemTests(unittest.TestCase):
    def test_scenarios(self):
        for name in ["fall", "fall_with_loss", "bend", "sit", "slow_lie", "already_lying", "short_dip", "occluded_fall", "missing", "zone_entry", "low_activity", "showcase"]:
            with self.subTest(name=name):
                case = sequence(name)
                detector = Detector(Config(**case["config"]))
                events = [e["kind"] for row in case["frames"] for e in detector.step(row["t"], row["points"])["events"]]
                self.assertEqual(events, case["expected_event_kinds"])

    def test_unknown_resets_continuous_timers(self):
        detector = Detector(Config(low_activity_seconds=2))
        events = []
        for i in range(40):
            row = detector.step(i / 10, None if i == 19 else pose())
            events.extend(row["events"])
            if i == 19:
                self.assertEqual(row["state"], "UNKNOWN")
        self.assertFalse(events)

    def test_time_gap_and_invalid_timestamp(self):
        detector = Detector(Config(low_activity_seconds=2))
        detector.step(0, pose())
        result = detector.step(10, pose())
        self.assertEqual(result["features"]["still_seconds"], 0)
        with self.assertRaises(ValueError):
            detector.step(10, pose())
        with self.assertRaises(ValueError):
            detector.step(float("nan"), pose())

    def test_low_visibility_and_bounds(self):
        p = pose()
        p[27][2] = 0.1
        self.assertIsNone(features(p))
        p[27] = [1.2, 0.8, 0.99]
        self.assertIsNone(features(p))
        self.assertIsNone(features(np.zeros((2, 3))))
        self.assertGreater(features(pose(rotation=90))["angle"], 89)

    def test_invalid_configuration(self):
        for kwargs in [{"floor_y": 2}, {"confirm_seconds": 0}, {"zone": [0.8, 0, 0.2, 1]}, {"pre_seconds": 100}]:
            with self.assertRaises(ValueError):
                Config(**kwargs)

    def test_initial_zone_occupancy_and_reentry(self):
        detector = Detector(Config(zone=[0.6, 0.1, 0.9, 0.99]))
        events = []
        for i in range(50):
            x = 0.75 if i < 15 or i >= 30 else 0.3
            events.extend(detector.step(i / 10, pose((x, 0.5)))["events"])
        self.assertEqual([e["kind"] for e in events], ["zone_entry"])

    def test_review_history_and_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.sqlite3"
            store = Store(path)
            session = store.session("demo", "synthetic_landmarks", Config().to_dict())
            event = store.add(session, {"kind": "suspected_fall", "time_s": 5, "onset_s": 3, "evidence": "test"})
            store.review(event, "confirmed", "核实")
            store.review(event, "false_positive", "重新检查片段")
            reopened = Store(path)
            self.assertEqual(reopened.events()[0]["status"], "false_positive")
            with reopened.connect() as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0], 2)
            reopened.export(Path(folder) / "export.json")
            self.assertEqual(len(json.loads((Path(folder) / "export.json").read_text(encoding="utf-8"))["reviews"]), 2)
            with self.assertRaises(ValueError):
                reopened.review("does-not-exist", "confirmed")


if __name__ == "__main__":
    unittest.main()
