"""视频/摄像头与合成关键点回放，共用规则和事件记录。"""
import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
from time import perf_counter
import urllib.request
import zipfile

import cv2
import numpy as np

from logic import Config, CONNECTIONS, Detector
from storage import Store

ROOT = Path(__file__).resolve().parent
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"


@dataclass
class RuntimeConfig:
    device: str = "auto"
    model_variant: str = "lite"
    max_side: int = 960
    opencv_threads: int = 0

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "gpu"}:
            raise ValueError("device 必须为 auto/cpu/gpu")
        if self.model_variant not in {"lite", "full", "heavy"}:
            raise ValueError("model_variant 必须为 lite/full/heavy")
        if type(self.max_side) is not int or not 256 <= self.max_side <= 3840:
            raise ValueError("最长边必须为 256～3840 的整数")
        if type(self.opencv_threads) is not int or not 0 <= self.opencv_threads <= 64:
            raise ValueError("OpenCV 线程数必须为 0～64；0 保留库默认值，不控制姿态模型线程")


def runtime_arguments(parser):
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--model-variant", choices=["lite", "full", "heavy"])
    parser.add_argument("--max-side", type=int, help="处理图像最长边；默认 960，调低可减轻负载但可能影响关键点质量")
    parser.add_argument("--opencv-threads", type=int, help="0 为库默认；仅影响 OpenCV，不是 MediaPipe 推理线程")


def runtime_from_args(args):
    values = asdict(load_runtime(args.config)) if hasattr(args, "config") else {}
    values.update({key: getattr(args, key) for key in asdict(RuntimeConfig()) if getattr(args, key) is not None})
    return RuntimeConfig(**values)


def validate_model(path):
    if not zipfile.is_zipfile(path):
        raise ValueError(f"模型不是有效的 task bundle：{path}；请移走损坏文件后重试下载")
    with zipfile.ZipFile(path) as bundle:
        if bundle.testzip() or not any(name.endswith(".tflite") for name in bundle.namelist()):
            raise ValueError("模型包内容不完整")


def ensure_model(variant="lite"):
    RuntimeConfig(model_variant=variant)
    path = ROOT / "models" / f"pose_landmarker_{variant}.task"
    if path.is_file():
        validate_model(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = MODEL_URL.format(variant=variant)
    print(f"Downloading official MediaPipe Pose Landmarker {variant} model...", flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".download", delete=False) as target:
            temporary = Path(target.name)
            with urllib.request.urlopen(url, timeout=45) as response:
                while block := response.read(1024 * 1024):
                    target.write(block)
        validate_model(temporary)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    path.with_suffix(".source.json").write_text(json.dumps({"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")
    return path


class PoseReader:
    def __init__(self, runtime=None):
        runtime = runtime or RuntimeConfig()
        # 当前官方 Python delegate 文档仅承诺 Ubuntu GPU；不把核显/Windows 当作已支持。
        gpu_platform = platform.system() == "Linux"
        if runtime.device == "gpu" and not gpu_platform:
            raise ValueError("当前 MediaPipe Python GPU delegate 官方支持限 Ubuntu；本平台请选 auto 或 cpu")
        import mediapipe as mp
        self.mp = mp
        model = ensure_model(runtime.model_variant)

        def create(device):
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=getattr(mp.tasks.BaseOptions.Delegate, device.upper())),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=2, min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5, min_tracking_confidence=0.5,
            )
            return mp.tasks.vision.PoseLandmarker.create_from_options(options)

        selected = "gpu" if runtime.device != "cpu" and gpu_platform else "cpu"
        reason = "当前平台使用 CPU；GPU delegate 官方支持限 Ubuntu" if runtime.device == "auto" and not gpu_platform else None
        try:
            self.reader = create(selected)
        except RuntimeError as exc:
            if runtime.device != "auto" or selected != "gpu":
                raise
            reason = f"GPU 初始化失败，已回退 CPU：{exc}"
            selected = "cpu"
            self.reader = create(selected)
        self.runtime_info = {**asdict(runtime), "effective_device": selected, "fallback_reason": reason}
        self.last_ms = -1

    def predict(self, frame, t):
        timestamp = max(self.last_ms + 1, round(t * 1000))
        self.last_ms = timestamp
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.reader.detect_for_video(image, timestamp)
        if len(result.pose_landmarks) != 1:
            return None, "检测到多人，无法保持单人身份" if len(result.pose_landmarks) > 1 else "未检出人体"
        points = [[p.x, p.y, min(p.visibility, p.presence)] for p in result.pose_landmarks[0]]
        return points, "关键点可见度不足或人体部分出画"

    def close(self):
        self.reader.close()


def draw(frame, points, result, config, t, synthetic=False):
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    if points is not None and result["features"] is not None:
        p = np.asarray(points, dtype=float).copy()
        p[:, :2] = np.clip(p[:, :2], -1, 2)  # 仅限制绘图坐标，避免畸形非关键点使整型转换溢出。
        for a, b in CONNECTIONS:
            if min(p[a, 2], p[b, 2]) >= config.visibility:
                start = tuple((p[a, :2] * [w, h]).astype(int))
                end = tuple((p[b, :2] * [w, h]).astype(int))
                cv2.line(canvas, start, end, (75, 190, 130), 3, cv2.LINE_AA)
        for x, y, confidence in p:
            if confidence >= config.visibility:
                cv2.circle(canvas, (int(x * w), int(y * h)), 4, (80, 230, 160), -1)
    cv2.line(canvas, (0, int(h * config.floor_y)), (w, int(h * config.floor_y)), (135, 130, 70), 1)
    if config.zone is not None:
        x1, y1, x2, y2 = config.zone
        cv2.rectangle(canvas, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (220, 160, 70), 2)
    color = (80, 200, 120) if result["state"] == "NORMAL" else (0, 160, 255)
    if result["state"] == "ALERTED":
        color = (60, 60, 240)
    cv2.rectangle(canvas, (0, 0), (w, 76), (28, 34, 31), -1)
    prefix = "SYNTHETIC LANDMARK REPLAY | " if synthetic else "POSE MODEL | "
    cv2.putText(canvas, prefix + result["state"] + f" | {t:.1f}s", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    f = result["features"]
    detail = f"angle {f['torso_angle']:.0f} deg | hip y {f['hip_y']:.2f} | still {f['still_seconds']:.1f}s" if f else "UNABLE TO ASSESS - no reliable single-person pose"
    cv2.putText(canvas, detail, (14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 230, 225), 1, cv2.LINE_AA)
    return canvas


class Clips:
    def __init__(self, directory, store, config):
        self.directory, self.store, self.config = directory, store, config
        self.buffer, self.pending = deque(), []

    def frame(self, t, image):
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("事件帧编码失败")
        entry = (t, encoded)
        self.buffer.append(entry)
        while self.buffer and t - self.buffer[0][0] > self.config.pre_seconds:
            self.buffer.popleft()
        for pending in self.pending[:]:
            pending["frames"].append(entry)
            if t >= pending["end"]:
                self.finish(pending, truncated=False)
                self.pending.remove(pending)

    def event(self, event_id, t):
        snapshot = self.directory / f"{event_id}.jpg"
        snapshot.write_bytes(self.buffer[-1][1].tobytes())
        self.store.media(event_id, snapshot=snapshot.resolve().relative_to(self.store.path.parent.resolve()).as_posix())
        self.pending.append({"id": event_id, "frames": list(self.buffer), "end": t + self.config.post_seconds,
                             "desired_start": max(0.0, t - self.config.pre_seconds)})

    def finish(self, pending, truncated):
        frames = pending["frames"]
        path = self.directory / f"{pending['id']}.mp4"
        first = cv2.imdecode(frames[0][1], cv2.IMREAD_COLOR)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (first.shape[1], first.shape[0]))
        if not writer.isOpened():
            self.store.media(pending["id"], clip_info={"error": "视频编码器不可用"})
            return
        times = np.array([entry[0] for entry in frames])
        try:
            # 按源时间重采样为 10 FPS，处理速度不会改变片段播放时长。
            for t in np.arange(times[0], times[-1] + 0.001, 0.1):
                index = max(0, int(np.searchsorted(times, t, side="right")) - 1)
                writer.write(cv2.imdecode(frames[index][1], cv2.IMREAD_COLOR))
        finally:
            writer.release()
        self.store.media(pending["id"], clip=path.resolve().relative_to(self.store.path.parent.resolve()).as_posix(), clip_info={
            "start_s": float(times[0]), "end_s": float(times[-1]), "fps": 10,
            "post_truncated": bool(truncated), "pre_truncated": bool(times[0] > pending["desired_start"] + 0.2)})

    def close(self):
        for pending in self.pending:
            self.finish(pending, truncated=True)
        self.pending.clear()


def load_config(path):
    values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    values.pop("runtime", None)
    return Config(**values)


def load_runtime(path):
    return RuntimeConfig(**json.loads(Path(path).read_text(encoding="utf-8-sig")).get("runtime", {}))


def run(source, output=ROOT / "data", config=None, on_frame=None, stopped=lambda: False, annotated=False, runtime=None):
    config = config or Config()
    runtime = runtime or RuntimeConfig()
    output = Path(output)
    store = Store(output / "events.sqlite3")
    synthetic = not isinstance(source, int) and Path(source).suffix.lower() == ".json"
    demo = json.loads(Path(source).read_text(encoding="utf-8")) if synthetic else None
    if synthetic and (not isinstance(demo, dict) or demo.get("kind") != "synthetic_landmarks"):
        raise ValueError("JSON 输入必须是显式标记的合成关键点序列")
    kind = "synthetic_landmarks" if synthetic else "camera" if isinstance(source, int) else "video"
    session_id = store.session(source, kind, {**config.to_dict(), "runtime": asdict(runtime)})
    directory = output / "sessions" / session_id
    directory.mkdir(parents=True)
    clips = Clips(directory, store, config)
    detector = Detector(config)
    count = unknown = event_count = 0
    processing_seconds = 0.0
    source_t = 0.0
    summary = {"session_id": session_id, "source_kind": kind, "completed": False,
               "runtime": {**asdict(runtime), "effective_device": "synthetic_no_model" if synthetic else None},
               "host": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()}}

    def save_summary():
        summary.update(frames=count, unknown_frames=int(unknown), source_seconds=source_t,
                       event_count=event_count, processing_seconds=processing_seconds,
                       processing_fps=count / max(processing_seconds, 1e-6),
                       timing_scope="逐帧解码、推理、规则、标注、缓冲/事件写盘；不含模型初始化、界面节流、最后片段收尾")
        store.finish(session_id, summary)
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def record_error(exc_type, exc, traceback):
        if exc is not None:
            summary.update(completed=False, error=f"{exc_type.__name__}: {exc}")
        return False

    with ExitStack() as stack:
        stack.callback(save_summary)
        stack.push(record_error)
        stack.callback(clips.close)
        original_threads = cv2.getNumThreads()
        # ponytail: 一个进程只运行一个分析会话；多路并行请分进程，避免全局 CV 线程配置竞争。
        if runtime.opencv_threads:
            cv2.setNumThreads(runtime.opencv_threads)
            stack.callback(cv2.setNumThreads, original_threads)
        summary["opencv_threads_effective"] = cv2.getNumThreads()
        if synthetic:
            iterator = iter(demo["frames"])
            fps = float(demo.get("fps", 10))
            if not math.isfinite(fps) or not 1 <= fps <= 240:
                raise ValueError("合成输入 fps 必须为 1～240 的有限数")
        else:
            capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
            stack.callback(capture.release)
            if not capture.isOpened():
                raise ValueError("无法打开视频或摄像头")
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or not 1 <= fps <= 240:
                if not isinstance(source, int):
                    raise ValueError("无法确定视频帧率；请先转为带正确帧率的 MP4")
                fps = 30.0
            pose = PoseReader(runtime)
            stack.callback(pose.close)
            summary["runtime"] = pose.runtime_info
        stream = stack.enter_context((directory / "trace.jsonl").open("w", encoding="utf-8"))
        writer = None
        camera_start = perf_counter()
        previous_t = -1.0
        try:
            while not stopped():
                started = perf_counter()
                if synthetic:
                    row = next(iterator, None)
                    if row is None:
                        break
                    source_t, points = float(row["t"]), row["points"]
                    frame = np.full((540, 960, 3), (46, 49, 45), dtype=np.uint8)
                    reason = "合成遮挡/缺失关键点"
                else:
                    ok, frame = capture.read()
                    if not ok:
                        if isinstance(source, int):
                            raise RuntimeError("摄像头读取中断；无法继续判断")
                        break
                    if max(frame.shape[:2]) > runtime.max_side:
                        scale = runtime.max_side / max(frame.shape[:2])
                        frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    if isinstance(source, int):
                        source_t = perf_counter() - camera_start
                    else:
                        timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        source_t = timestamp if math.isfinite(timestamp) and timestamp > previous_t else count / fps
                        source_t = max(source_t, previous_t + 1 / fps)
                    points, reason = pose.predict(frame, source_t)
                previous_t = source_t
                result = detector.step(source_t, points, frame.shape[1] / frame.shape[0], reason)
                unknown += result["state"] == "UNKNOWN"
                canvas = draw(frame, points, result, config, source_t, synthetic)
                clips.frame(source_t, canvas)
                event_ids = []
                for event in result["events"]:
                    event_id = store.add(session_id, event)
                    clips.event(event_id, source_t)
                    event_ids.append(event_id)
                    event_count += 1
                if annotated:
                    if writer is None:
                        writer = cv2.VideoWriter(str(directory / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas.shape[1], canvas.shape[0]))
                        if not writer.isOpened():
                            raise RuntimeError("无法创建标注视频")
                        stack.callback(writer.release)
                    writer.write(canvas)
                stream.write(json.dumps({"t": source_t, **result}, ensure_ascii=False) + "\n")
                count += 1
                processing_seconds += perf_counter() - started
                if on_frame:
                    on_frame(canvas, {"t": source_t, **result, "session_id": session_id, "source_kind": kind,
                                      "processing_fps": count / max(processing_seconds, 1e-6), "event_ids": event_ids,
                                      "runtime": summary["runtime"]})
            if count == 0 and not stopped():
                raise ValueError("输入没有可处理的视频帧")
            summary["completed"] = not stopped()
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", help="视频路径、合成 JSON，或 camera:0（主动打开摄像头）")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--annotated", action="store_true")
    parser.add_argument("--download-model", action="store_true")
    runtime_arguments(parser)
    args = parser.parse_args()
    if args.download_model:
        print(ensure_model(runtime_from_args(args).model_variant))
    elif args.source:
        source = int(args.source.split(":", 1)[1]) if args.source.startswith("camera:") else args.source
        print(json.dumps(run(source, args.output, load_config(args.config), annotated=args.annotated, runtime=runtime_from_args(args)), ensure_ascii=False, indent=2))
    else:
        parser.error("请指定输入，或使用 --download-model")
