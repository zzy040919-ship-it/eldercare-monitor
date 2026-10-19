"""用一张全身图片重复成短视频，验证真实模型和处理吞吐；不评估跌倒识别率。"""
import argparse
import json
from pathlib import Path
import platform
import tempfile
from time import perf_counter

import cv2

from monitor import run, runtime_arguments, runtime_from_args


def verify(image_path, runtime=None):
    picture = cv2.imread(str(image_path))
    if picture is None:
        raise ValueError("无法读取图片")
    picture = cv2.resize(picture, (640, round(picture.shape[0] * 640 / picture.shape[1])))
    picture = picture[:picture.shape[0] // 2 * 2, :picture.shape[1] // 2 * 2]
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "static-smoke.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (picture.shape[1], picture.shape[0]))
        if not writer.isOpened():
            raise RuntimeError("视频编码器不可用")
        try:
            for _ in range(90):
                writer.write(picture)
        finally:
            writer.release()
        started = perf_counter()
        summary = run(path, Path(folder) / "results", runtime=runtime)
        elapsed = perf_counter() - started
    if summary["frames"] != 90 or not summary["completed"]:
        raise AssertionError("模型管线未处理完 90 帧")
    return {"test": "静态全身图片重复90帧；真实姿态模型冒烟，不是跌倒识别评估",
            "image_filename": Path(image_path).name, "resolution": [picture.shape[1], picture.shape[0]],
            "source_fps": 30, "platform": platform.platform(), "python": platform.python_version(),
            "delegate": summary["runtime"]["effective_device"], "summary": summary, "wall_seconds_including_initialization": elapsed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--report", type=Path)
    runtime_arguments(parser)
    args = parser.parse_args()
    result = verify(args.image, runtime_from_args(args))
    content = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(content)
    print(content)
