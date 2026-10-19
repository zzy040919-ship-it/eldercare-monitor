"""单人固定相机的可解释时序规则。输入是关键点，不直接推断医学状态。"""
from collections import deque
from dataclasses import asdict, dataclass, fields
import math

import numpy as np

KEYS = [11, 12, 23, 24, 25, 26, 27, 28]
CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28)]


@dataclass
class Config:
    visibility: float = 0.55
    floor_y: float = 0.65
    horizontal_angle: float = 60.0
    hip_drop: float = 0.12
    down_speed: float = 0.22
    history_seconds: float = 1.2
    confirm_seconds: float = 1.2
    rearm_seconds: float = 1.5
    cooldown_seconds: float = 8.0
    low_activity_seconds: float = 120.0
    motion_threshold: float = 0.025
    zone: list | None = None
    zone_dwell_seconds: float = 0.5
    max_gap_seconds: float = 0.5
    pre_seconds: float = 3.0
    post_seconds: float = 3.0

    def __post_init__(self):
        for field in fields(self):
            if field.name == "zone":
                continue
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} 必须是有限正数")
        if not (0 < self.visibility <= 1 and 0 < self.floor_y < 1 and 0 < self.horizontal_angle < 90):
            raise ValueError("可见度/地面阈值/角度超出范围")
        if self.max_gap_seconds > self.history_seconds or self.pre_seconds > 30 or self.post_seconds > 30:
            raise ValueError("时间间隔不合理，片段前后最多各 30 秒")
        if self.zone is not None:
            if len(self.zone) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in self.zone):
                raise ValueError("区域必须是四个有限数 [x1,y1,x2,y2]")
            x1, y1, x2, y2 = self.zone
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("区域坐标必须归一化且左上小于右下")

    def to_dict(self):
        return asdict(self)


def features(landmarks, aspect=16 / 9, visibility=0.55):
    if landmarks is None:
        return None
    try:
        points = np.asarray(landmarks, dtype=float)
    except (TypeError, ValueError):
        return None
    if points.shape != (33, 3) or not np.isfinite(points).all():
        return None
    if np.any(points[:, 2] < 0) or np.any(points[:, 2] > 1):
        return None
    if np.any(points[KEYS, 2] < visibility) or np.any(points[KEYS, :2] < 0) or np.any(points[KEYS, :2] > 1):
        return None
    shoulder = points[[11, 12], :2].mean(axis=0)
    hip = points[[23, 24], :2].mean(axis=0)
    dx, dy = (shoulder - hip) * [aspect, 1]
    if math.hypot(dx, dy) < 0.04:
        return None
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    return {"hip": hip, "angle": angle, "body": points[KEYS, :2] * [aspect, 1]}


class Detector:
    def __init__(self, config=None):
        self.config = config or Config()
        self.history = deque()
        self.last_t = None
        self.smoothed = None
        self.candidate_since = None
        self.fall_latched = False
        self.last_fall = -math.inf
        self.recovery_since = None
        self.anchor = None
        self.still_since = None
        self.low_latched = False
        self.zone_stable = None
        self.zone_candidate = None
        self.zone_since = None

    def _reset_continuity(self):
        self.history.clear()
        self.smoothed = None
        self.candidate_since = None
        self.recovery_since = None
        self.anchor, self.still_since = None, None
        self.low_latched = False
        self.zone_stable = self.zone_candidate = self.zone_since = None
        # 跌倒事件锁保持；遮挡后仍倒地不能被视为新事件。

    def step(self, t, landmarks, aspect=16 / 9, reason="人体未检出或关键点不可靠"):
        if not math.isfinite(t) or t < 0 or (self.last_t is not None and t <= self.last_t):
            raise ValueError("时间戳必须有限、非负且严格递增")
        previous_t = self.last_t
        self.last_t = t
        gap = previous_t is not None and t - previous_t > self.config.max_gap_seconds
        if gap:
            self._reset_continuity()
        f = features(landmarks, aspect, self.config.visibility)
        if f is None:
            self._reset_continuity()
            return {"state": "UNKNOWN", "reason": reason, "events": [], "features": None}
        dt = min(t - previous_t, self.config.max_gap_seconds) if previous_t is not None else 0.1
        alpha = 1 - math.exp(-dt / 0.12)
        if self.smoothed is not None:
            f = {k: self.smoothed[k] + alpha * (v - self.smoothed[k]) for k, v in f.items()}
        self.smoothed = f
        c, hip, angle = self.config, f["hip"], float(f["angle"])
        while self.history and t - self.history[0][0] > c.history_seconds:
            self.history.popleft()
        upright = angle < 35 and hip[1] < c.floor_y
        lying_low = angle >= c.horizontal_angle and hip[1] >= c.floor_y
        eligible = [(old_t, y) for old_t, y, was_upright in self.history if was_upright and t - old_t >= 0.15]
        drop = max((float(hip[1]) - y for _, y in eligible), default=0.0)
        speed = max(((float(hip[1]) - y) / (t - old_t) for old_t, y in eligible), default=0.0)
        self.history.append((t, float(hip[1]), upright))
        events = []
        if not self.fall_latched and lying_low and drop >= c.hip_drop and speed >= c.down_speed:
            if self.candidate_since is None:
                self.candidate_since = t
        if not lying_low:
            self.candidate_since = None
        if self.candidate_since is not None and t - self.candidate_since >= c.confirm_seconds:
            if not self.fall_latched and t - self.last_fall >= c.cooldown_seconds:
                events.append({"kind": "suspected_fall", "time_s": t, "onset_s": self.candidate_since,
                               "evidence": "近期直立后快速下降，躯干近水平且低位持续达到阈值"})
                self.fall_latched, self.last_fall = True, t
        if self.fall_latched:
            if upright:
                self.recovery_since = t if self.recovery_since is None else self.recovery_since
                if t - self.recovery_since >= c.rearm_seconds and t - self.last_fall >= c.cooldown_seconds:
                    self.fall_latched, self.candidate_since = False, None
            else:
                self.recovery_since = None
        state = "ALERTED" if self.fall_latched else "CANDIDATE" if self.candidate_since is not None else "NORMAL"
        if self.anchor is None or np.max(np.linalg.norm(f["body"] - self.anchor, axis=1)) > c.motion_threshold:
            self.anchor, self.still_since, self.low_latched = f["body"].copy(), t, False
        still_seconds = t - self.still_since
        if state == "NORMAL" and not self.low_latched and still_seconds >= c.low_activity_seconds:
            events.append({"kind": "low_activity", "time_s": t, "onset_s": self.still_since,
                           "evidence": f"持续可靠检测，关键点低活动达到 {still_seconds:.1f} 秒"})
            self.low_latched = True
        if c.zone is not None:
            x1, y1, x2, y2 = c.zone
            inside = bool(x1 <= hip[0] <= x2 and y1 <= hip[1] <= y2)
            if self.zone_candidate != inside:
                self.zone_candidate, self.zone_since = inside, t
            if t - self.zone_since >= c.zone_dwell_seconds and self.zone_stable != inside:
                if self.zone_stable is False and inside:
                    events.append({"kind": "zone_entry", "time_s": t, "onset_s": self.zone_since,
                                   "evidence": "髋部中心由区域外进入区域内，并持续超过防抖时间"})
                self.zone_stable = inside
        serial = {"hip_x": float(hip[0]), "hip_y": float(hip[1]), "torso_angle": angle,
                  "hip_drop": drop, "down_speed": speed, "still_seconds": still_seconds}
        return {"state": state, "reason": "可靠单人姿态", "events": events, "features": serial}
