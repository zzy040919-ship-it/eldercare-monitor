"""离线集成验证：不打开摄像头，不下载模型；合成数据不代表真实识别率。"""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import cv2
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app import Window, configure_application
from logic import Config
from monitor import ROOT, load_config, run
from storage import Store


class IntegrationTests(unittest.TestCase):
    def test_event_media_and_portable_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = run(ROOT / "examples/showcase.json", output, load_config(ROOT / "examples/showcase_config.json"))
            self.assertTrue(summary["completed"])
            self.assertEqual(summary["frames"], 201)
            self.assertEqual(summary["unknown_frames"], 10)
            rows = Store(output / "events.sqlite3").events()
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertFalse(Path(row["clip"]).is_absolute())
                self.assertTrue((output / row["snapshot"]).is_file())
                info = json.loads(row["clip_info_json"])
                self.assertLessEqual(abs(info["start_s"] - max(0, row["time_s"] - 3)), .11)
                expected_end = min(summary["source_seconds"], row["time_s"] + 3)
                self.assertLessEqual(abs(info["end_s"] - expected_end), .11)
                self.assertEqual(info["post_truncated"], row["time_s"] + 3 > summary["source_seconds"])
                capture = cv2.VideoCapture(str(output / row["clip"]))
                try:
                    self.assertTrue(capture.read()[0])
                    self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 10)
                    self.assertLessEqual(abs(capture.get(cv2.CAP_PROP_FRAME_COUNT) / 10 - (info["end_s"] - info["start_s"])), .11)
                finally:
                    capture.release()

    def test_stop_finalizes_short_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            should_stop = False

            def callback(frame, info):
                nonlocal should_stop
                should_stop = bool(info["event_ids"])

            summary = run(ROOT / "examples/fall.json", folder, Config(), callback, lambda: should_stop)
            self.assertFalse(summary["completed"])
            rows = Store(Path(folder) / "events.sqlite3").events()
            self.assertEqual(len(rows), 1)
            self.assertTrue(json.loads(rows[0]["clip_info_json"])["post_truncated"])
            self.assertTrue((Path(folder) / rows[0]["clip"]).is_file())

    def test_setup_error_is_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                run(Path(folder) / "nonexistent.mp4", folder)
            with Store(Path(folder) / "events.sqlite3").connect() as con:
                summary = json.loads(con.execute("SELECT summary_json FROM sessions").fetchone()[0])
            self.assertFalse(summary["completed"])
            self.assertEqual(summary["frames"], 0)
            self.assertIn("error", summary)

    def test_gui_review_region_and_stop(self):
        application = QApplication.instance() or QApplication([])
        configure_application(application)
        with tempfile.TemporaryDirectory() as folder:
            window = Window(folder)
            window.show()
            application.processEvents()
            try:
                window.choose_demo()
                rect = window.preview.image_rect()
                from PySide6.QtCore import QPoint
                start = QPoint(round(rect.x() + .2 * rect.width()), round(rect.y() + .2 * rect.height()))
                end = QPoint(round(rect.x() + .8 * rect.width()), round(rect.y() + .8 * rect.height()))
                QTest.mousePress(window.preview, Qt.MouseButton.LeftButton, pos=start)
                QTest.mouseMove(window.preview, end)
                QTest.mouseRelease(window.preview, Qt.MouseButton.LeftButton, pos=end)
                for actual, expected in zip(window.config.zone, [.2, .2, .8, .8]):
                    self.assertAlmostEqual(actual, expected, delta=.005)
                window.choose_demo()
                window.realtime.setChecked(False)
                errors = []
                window.error = errors.append
                window.start()
                self.wait_worker(application, window)
                self.assertEqual(errors, [])
                self.assertEqual(window.table.rowCount(), 3)
                self.assertTrue(window.start_button.isEnabled())
                self.assertFalse(window.stop_button.isEnabled())
                window.table.selectRow(0)
                window.note.setText("合成演示复核，不作为真实效果数据")
                window.review("confirmed")
                self.assertEqual(len(window.store.events("confirmed")), 1)
                window.review("false_positive")
                self.assertEqual(len(window.store.events("false_positive")), 1)
                window.filter.setCurrentIndex(window.filter.findData("false_positive"))
                self.assertEqual(window.table.rowCount(), 1)
                window.filter.setCurrentIndex(0)
                window.table.selectRow(0)
                application.processEvents()
                preview = os.environ.get("CARE_TEST_PREVIEW")
                if preview:
                    self.assertTrue(window.grab().save(preview))
                window.realtime.setChecked(True)
                window.start()
                deadline = time.monotonic() + .4
                while time.monotonic() < deadline:
                    application.processEvents()
                    time.sleep(.01)
                window.stop()
                self.wait_worker(application, window)
                self.assertEqual(errors, [])
                with window.store.connect() as con:
                    summaries = [json.loads(row[0]) for row in con.execute("SELECT summary_json FROM sessions")]
                self.assertTrue(any(not summary["completed"] for summary in summaries))
            finally:
                if window.worker and window.worker.isRunning():
                    window.worker.requestInterruption()
                    self.wait_worker(application, window)
                window.close()
                application.processEvents()

    def wait_worker(self, application, window):
        deadline = time.monotonic() + 30
        while window.worker.isRunning() and time.monotonic() < deadline:
            application.processEvents()
            time.sleep(.01)
        self.assertFalse(window.worker.isRunning(), "后台线程未及时结束")
        application.processEvents()


if __name__ == "__main__":
    unittest.main()
