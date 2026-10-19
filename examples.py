"""生成明确标记的合成关键点序列，用于规则测试和界面演示，不经过姿态模型。"""
import json
import math
from pathlib import Path

import numpy as np

from logic import Config

ROOT = Path(__file__).resolve().parent


def pose(hip=(0.42, 0.50), rotation=0, bend=False):
    points = np.tile([hip[0], hip[1], 0.95], (33, 1)).astype(float)
    offsets = {0: (0, -0.31), 11: (-0.035, -0.22), 12: (0.035, -0.22),
               13: (-0.06, -0.12), 14: (0.06, -0.12), 15: (-0.07, -0.01), 16: (0.07, -0.01),
               23: (-0.03, 0), 24: (0.03, 0), 25: (-0.035, 0.17), 26: (0.035, 0.17),
               27: (-0.04, 0.34), 28: (0.04, 0.34)}
    angle = math.radians(rotation)
    for index, (dx, dy) in offsets.items():
        x, y = dx * (16 / 9), dy
        if not bend or index < 23:
            x, y = x * math.cos(angle) + y * math.sin(angle), -x * math.sin(angle) + y * math.cos(angle)
        points[index, :2] = [hip[0] + x / (16 / 9), hip[1] + y]
    return points.round(5).tolist()


def sequence(name, fps=10):
    config = Config()
    duration = 8.0
    expected = []
    if name in {"fall", "fall_with_loss"}:
        expected = ["suspected_fall"]
    elif name == "zone_entry":
        config.zone = [0.65, 0.2, 0.95, 0.95]
        expected = ["zone_entry"]
    elif name == "low_activity":
        config.low_activity_seconds = 4.0
        expected = ["low_activity"]
    elif name == "showcase":
        config.zone = [0.65, 0.2, 0.95, 0.95]
        config.low_activity_seconds = 4.0
        expected = ["suspected_fall", "zone_entry", "low_activity"]
        duration = 20.0
    frames = []
    for i in range(round(duration * fps) + 1):
        t = i / fps
        p = pose()
        if name in {"fall", "fall_with_loss", "occluded_fall"}:
            progress = float(np.clip((t - 2) / 0.6, 0, 1))
            p = pose((0.42, 0.5 + 0.28 * progress), rotation=90 * progress)
            if name == "fall_with_loss" and 5 <= t < 6:
                p = None
            if name == "occluded_fall" and 1.9 <= t < 4.5:
                p = None
        elif name == "bend":
            p = pose(rotation=85 * min(1.0, max(0.0, (t - 2) / 0.6)), bend=True)
        elif name == "sit":
            progress = float(np.clip((t - 2) / 0.8, 0, 1))
            p = pose((0.42, 0.5 + 0.16 * progress))
            for j in (25, 26, 27, 28):
                p[j][1] = min(p[j][1], 0.9)
        elif name == "slow_lie":
            progress = float(np.clip((t - 1) / 5, 0, 1))
            p = pose((0.42, 0.5 + 0.28 * progress), rotation=90 * progress)
        elif name == "already_lying":
            p = pose((0.42, 0.78), rotation=90)
        elif name == "short_dip":
            if 2 <= t < 2.3:
                p = pose((0.42, 0.78), rotation=90)
        elif name == "zone_entry":
            p = pose((0.3 + 0.45 * float(np.clip((t - 2) / 2, 0, 1)), 0.5))
        elif name == "missing":
            p = None
        elif name == "showcase":
            if t < 7:
                progress = float(np.clip((t - 2) / 0.6, 0, 1))
                p = pose((0.42, 0.5 + 0.28 * progress), rotation=90 * progress)
            elif t < 8:
                p = None
            else:
                p = pose((0.42 + 0.33 * float(np.clip((t - 9) / 4, 0, 1)), 0.5))
        frames.append({"t": t, "points": p})
    return {"kind": "synthetic_landmarks", "name": name, "fps": fps,
            "description": "人工构造关键点，仅验证规则与应用流程，不评估 MediaPipe 识别率",
            "config": config.to_dict(), "expected_event_kinds": expected, "frames": frames}


if __name__ == "__main__":
    directory = ROOT / "examples"
    directory.mkdir(exist_ok=True)
    names = ["fall", "fall_with_loss", "bend", "sit", "slow_lie", "already_lying", "short_dip", "occluded_fall", "missing", "zone_entry", "low_activity", "showcase"]
    for name in names:
        with (directory / f"{name}.json").open("x", encoding="utf-8") as stream:
            json.dump(sequence(name), stream, ensure_ascii=False, separators=(",", ":"))
    with (directory / "showcase_config.json").open("x", encoding="utf-8") as stream:
        json.dump(sequence("showcase")["config"], stream, indent=2)
    print(f"Generated {len(names)} synthetic sequences.")
