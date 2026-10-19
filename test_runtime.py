"""运行设备与异常输入测试；GPU 路径用测试替身，不宣称本机 GPU 实测。"""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import cv2
import numpy as np

from app import Worker
from examples import pose
from logic import Config, Detector
from monitor import PoseReader, RuntimeConfig, draw, ensure_model, run, runtime_arguments, runtime_from_args


def fake_mediapipe(factory):
    return SimpleNamespace(tasks=SimpleNamespace(
        BaseOptions=Mock(side_effect=lambda **kwargs: kwargs),
        vision=SimpleNamespace(PoseLandmarkerOptions=lambda **kwargs: kwargs,
                               RunningMode=SimpleNamespace(VIDEO="video"),
                               PoseLandmarker=SimpleNamespace(create_from_options=factory))))


class RuntimeTests(unittest.TestCase):
    def test_delegate_auto_fallback_and_forced_gpu(self):
        factory = Mock(side_effect=[RuntimeError("GPU not available"), Mock()])
        mp = fake_mediapipe(factory)
        mp.tasks.BaseOptions.Delegate = SimpleNamespace(CPU=0, GPU=1)
        with patch.dict("sys.modules", {"mediapipe": mp}), patch("monitor.platform.system", return_value="Linux"), patch("monitor.ensure_model", return_value=Path("fake.task")):
            reader = PoseReader(RuntimeConfig())
            self.assertEqual(reader.runtime_info["effective_device"], "cpu")
            self.assertIn("GPU not available", reader.runtime_info["fallback_reason"])
            self.assertEqual(factory.call_count, 2)
            reader.close()
            factory.reset_mock(side_effect=True)
            factory.side_effect = RuntimeError("forced GPU failure")
            with self.assertRaises(RuntimeError):
                PoseReader(RuntimeConfig(device="gpu"))
            self.assertEqual(factory.call_count, 1)
        with patch("monitor.platform.system", return_value="Windows"):
            with self.assertRaises(ValueError):
                PoseReader(RuntimeConfig(device="gpu"))

    def test_config_file_and_cli_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            path.write_text(json.dumps({**Config().to_dict(), "runtime": {"device": "cpu", "max_side": 640, "opencv_threads": 2}}), encoding="utf-8")
            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=Path)
            runtime_arguments(parser)
            runtime = runtime_from_args(parser.parse_args(["--config", str(path), "--max-side", "1280"]))
            self.assertEqual(runtime.device, "cpu")
            self.assertEqual(runtime.max_side, 1280)
            self.assertEqual(runtime.opencv_threads, 2)
        for values in [{"device": "npu"}, {"max_side": 0}, {"opencv_threads": -1}, {"model_variant": "unknown"}]:
            with self.assertRaises(ValueError):
                RuntimeConfig(**values)

    def test_model_cache_corruption_and_atomic_download(self):
        with tempfile.TemporaryDirectory() as directory, patch("monitor.ROOT", Path(directory)):
            folder = Path(directory) / "models"
            folder.mkdir()
            cached = folder / "pose_landmarker_lite.task"
            cached.write_bytes(b"broken cache")
            with self.assertRaises(ValueError):
                ensure_model()
            self.assertEqual(cached.read_bytes(), b"broken cache")
            with patch("monitor.urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaises(OSError):
                    ensure_model("full")
            self.assertFalse((folder / "pose_landmarker_full.task").exists())
            self.assertFalse(list(folder.glob("*.download")))
            with zipfile.ZipFile(folder / "pose_landmarker_full.task", "w") as bundle:
                bundle.writestr("pose.tflite", b"test-bundle-only")
            self.assertEqual(ensure_model("full").name, "pose_landmarker_full.task")

    def test_malformed_landmarks_do_not_crash_rendering(self):
        for points in [[[0, 1]], "broken", [[float("nan"), .5, .9]] * 33, [[.5, .5, 2.0]] * 33]:
            result = Detector().step(0, points)
            self.assertEqual(result["state"], "UNKNOWN")
            output = draw(np.zeros((180, 320, 3), dtype=np.uint8), points, result, Config(), 0)
            self.assertEqual(output.shape, (180, 320, 3))
        points = pose()
        points[0][0] = 1e100
        result = Detector().step(0, points)
        self.assertIsNotNone(result["features"])
        draw(np.zeros((180, 320, 3), dtype=np.uint8), points, result, Config(), 0)

    def test_resize_and_record_actual_device(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "test.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 10, (1280, 720))
            self.assertTrue(writer.isOpened())
            for _ in range(4):
                writer.write(np.zeros((720, 1280, 3), dtype=np.uint8))
            writer.release()
            sizes = []
            fake = Mock()
            fake.runtime_info = {"device": "auto", "effective_device": "cpu", "fallback_reason": "test double"}
            fake.predict.side_effect = lambda frame, t: (sizes.append(frame.shape[:2]) or None, "test no pose")
            original_threads = cv2.getNumThreads()
            with patch("monitor.PoseReader", return_value=fake):
                summary = run(source, folder / "out", runtime=RuntimeConfig(max_side=640, opencv_threads=1))
            self.assertEqual(sizes, [(360, 640)] * 4)
            self.assertEqual(summary["runtime"]["effective_device"], "cpu")
            self.assertEqual(summary["opencv_threads_effective"], 1)
            self.assertEqual(cv2.getNumThreads(), original_threads)

    def test_event_signal_survives_preview_queue_overflow(self):
        worker = Worker("unused", "unused", Config(), False)
        notices = []
        worker.event_created.connect(notices.append)
        for i in range(10):
            worker.on_frame(None, {"t": i, "event_ids": ["event"] if i == 1 else []})
        self.assertEqual(notices, [["event"]])
        self.assertEqual(worker.frames.qsize(), 2)


if __name__ == "__main__":
    unittest.main()
