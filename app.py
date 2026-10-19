"""本地桌面界面：后台分析、区域框选、事件回放与人工复核。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from time import monotonic

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QFontDatabase, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QSpinBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from logic import Config
from monitor import ROOT, RuntimeConfig, load_config, load_runtime, run, runtime_arguments, runtime_from_args
from storage import KINDS, STATUSES, Store

SOURCE_NAMES = {"synthetic_landmarks": "合成关键点", "video": "视频识别", "camera": "摄像头"}
STATE_NAMES = {"UNKNOWN": "无法判断", "NORMAL": "未触发疑似跌倒", "CANDIDATE": "疑似动作观察中", "ALERTED": "疑似跌倒 · 待人工核实"}


def configure_application(application):
    # Qt 的 offscreen 插件可能不枚举系统字体；使用本机字体，不打包字体文件。
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not QFontDatabase.families() and font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.families()
    for family in ["Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC"]:
        if family in families:
            application.setFont(QFont(family, 10))
            break


class Preview(QWidget):
    zone_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.image = QImage()
        self.zone = None
        self.origin = None
        self.editable = True
        self.setMinimumSize(480, 270)
        self.setToolTip("停止分析后，在画面上拖动鼠标设置提醒区域；以髋部中心判断进入")

    def set_frame(self, frame):
        # GUI 不调用 OpenCV；线程数设置只在分析线程启停时发生，避免与 GUI 的 CV 运算并发。
        self.image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_BGR888).copy()
        self.update()

    def image_rect(self):
        if self.image.isNull():
            return QRectF()
        size = self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2, size.width(), size.height())

    def normalized(self, position):
        rect = self.image_rect()
        if rect.isEmpty():
            return None
        return [min(1., max(0., (position.x() - rect.x()) / rect.width())),
                min(1., max(0., (position.y() - rect.y()) / rect.height()))]

    def mousePressEvent(self, event):
        if self.editable and event.button() == Qt.MouseButton.LeftButton and self.image_rect().contains(event.position()):
            self.origin = self.normalized(event.position())

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            end = self.normalized(event.position())
            self.zone = [min(self.origin[0], end[0]), min(self.origin[1], end[1]), max(self.origin[0], end[0]), max(self.origin[1], end[1])]
            self.update()

    def mouseReleaseEvent(self, event):
        if self.origin is not None:
            self.mouseMoveEvent(event)
            self.origin = None
            if self.zone[2] - self.zone[0] < .02 or self.zone[3] - self.zone[1] < .02:
                self.zone = None
            self.zone_changed.emit(self.zone)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202823"))
        rect = self.image_rect()
        if self.image.isNull():
            painter.setPen(QColor("#e0e8e2"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "选择视频，或运行合成演示\n摄像头不会自动打开")
            return
        painter.drawImage(rect, self.image)
        if self.zone:
            x1, y1, x2, y2 = self.zone
            painter.setPen(QPen(QColor("#e8bc63"), 2))
            painter.drawRect(QRectF(rect.x() + x1 * rect.width(), rect.y() + y1 * rect.height(), (x2 - x1) * rect.width(), (y2 - y1) * rect.height()))


class Worker(QThread):
    completed = Signal(object)
    failed = Signal(str)
    event_created = Signal(object)

    def __init__(self, source, output, config, realtime, runtime=None):
        super().__init__()
        self.source, self.output, self.config, self.realtime = source, output, config, realtime
        self.runtime = runtime or RuntimeConfig()
        # 单消费者只显示最新两帧，分析仍逐帧执行，避免慢界面积压内存。
        self.frames = Queue(maxsize=2)
        self.first_t = self.started = None

    def on_frame(self, frame, info):
        if self.started is None:
            self.started, self.first_t = monotonic(), info["t"]
        if self.realtime and not isinstance(self.source, int):
            deadline = self.started + info["t"] - self.first_t
            while monotonic() < deadline and not self.isInterruptionRequested():
                self.msleep(20)
        if info["event_ids"]:
            self.event_created.emit(info["event_ids"])
        try:
            self.frames.put_nowait((frame, info))
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            self.frames.put_nowait((frame, info))

    def run(self):
        try:
            self.completed.emit(run(self.source, self.output, self.config, self.on_frame, self.isInterruptionRequested, runtime=self.runtime))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Window(QMainWindow):
    def __init__(self, output=ROOT / "data", runtime=None):
        super().__init__()
        self.output = Path(output)
        self.store = Store(self.output / "events.sqlite3")
        self.source, self.worker = None, None
        self.config = load_config(ROOT / "config.json")
        self.runtime = runtime or load_runtime(ROOT / "config.json")
        self.rows, self.signature = [], None
        self.setWindowTitle("居家养老异常行为监测 · 辅助提醒原型")
        self.resize(1280, 850)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        title = QLabel("居家养老异常行为监测")
        title.setStyleSheet("font-size: 22px; font-weight: 600; padding: 6px 0;")
        layout.addWidget(title)
        layout.addWidget(QLabel("单人 · 固定相机 · 本地处理　｜　不是医疗器械，不提供无人值守安全保证"))
        toolbar = QHBoxLayout()
        self.video_button = self.button("选择视频", self.choose_video, toolbar)
        self.demo_button = self.button("合成演示（非视频识别）", self.choose_demo, toolbar)
        self.camera_button = self.button("选择摄像头 0", self.choose_camera, toolbar)
        self.runtime_button = self.button("运行设备与性能", self.choose_runtime, toolbar)
        self.start_button = self.button("开始分析", self.start, toolbar)
        self.stop_button = self.button("停止", self.stop, toolbar)
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.realtime = QCheckBox("按源时间预览")
        self.realtime.setChecked(True)
        toolbar.addWidget(self.realtime)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.source_label = QLabel("尚未选择输入")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        splitter = QSplitter()
        self.preview = Preview()
        self.preview.zone_changed.connect(self.set_zone)
        splitter.addWidget(self.preview)
        settings = QWidget()
        form = QFormLayout(settings)
        form.addRow(QLabel("场景参数（开始分析前设置）"))
        self.floor = self.number(.05, .95, .01, self.config.floor_y)
        self.confirm = self.number(.1, 10, .1, self.config.confirm_seconds)
        self.low = self.number(1, 3600, 1, self.config.low_activity_seconds)
        form.addRow("低位髋部 Y 阈值", self.floor)
        form.addRow("疑似姿态持续 / 秒", self.confirm)
        form.addRow("低活动持续 / 秒", self.low)
        help_text = QLabel("Y 为画面高度比例，向下增大。\n这不是物理地面高度。\n\n在左侧画面拖动框选提醒区域。\n以髋部中心进入区域为准。")
        help_text.setWordWrap(True)
        form.addRow(help_text)
        self.zone_label = QLabel("区域提醒：关闭")
        self.zone_label.setWordWrap(True)
        form.addRow(self.zone_label)
        self.clear_zone = QPushButton("清除区域")
        self.clear_zone.clicked.connect(lambda: self.set_zone(None))
        form.addRow(self.clear_zone)
        self.save_config_button = QPushButton("导出当前参数")
        self.save_config_button.clicked.connect(self.save_config)
        form.addRow(self.save_config_button)
        self.load_config_button = QPushButton("载入参数 JSON")
        self.load_config_button.clicked.connect(self.choose_config)
        form.addRow(self.load_config_button)
        self.runtime_label = QLabel()
        self.runtime_label.setWordWrap(True)
        form.addRow(self.runtime_label)
        self.update_runtime_label()
        splitter.addWidget(settings)
        splitter.setSizes([850, 330])
        layout.addWidget(splitter, 3)
        self.state = QLabel("就绪：合成演示不运行姿态模型，不代表真实识别效果。")
        self.state.setWordWrap(True)
        layout.addWidget(self.state)
        event_bar = QHBoxLayout()
        event_bar.addWidget(QLabel("事件记录"))
        self.filter = QComboBox()
        self.filter.addItem("全部状态", None)
        for key, name in STATUSES.items():
            self.filter.addItem(name, key)
        self.filter.currentIndexChanged.connect(lambda: self.refresh_events(force=True))
        event_bar.addWidget(self.filter)
        self.counts = QLabel()
        event_bar.addWidget(self.counts)
        event_bar.addStretch()
        self.button("导出记录 JSON", self.export, event_bar)
        layout.addLayout(event_bar)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["会话开始（本机时间）", "源时间 / 秒", "事件", "状态", "输入类型"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.show_event)
        layout.addWidget(self.table, 2)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(72)
        layout.addWidget(self.details)
        review_bar = QHBoxLayout()
        self.note = QLineEdit()
        self.note.setMaxLength(2000)
        self.note.setPlaceholderText("人工复核备注（合成事件只用于操作演示）")
        review_bar.addWidget(self.note, 1)
        self.review_controls = []
        for status, label in [("confirmed", "确认事件"), ("false_positive", "标记误报"), ("pending", "恢复待确认")]:
            self.review_controls.append(self.button(label, lambda checked=False, s=status: self.review(s), review_bar))
        self.review_controls.append(self.button("回放片段", lambda: self.open_media("clip"), review_bar))
        self.review_controls.append(self.button("查看截图", lambda: self.open_media("snapshot"), review_bar))
        layout.addLayout(review_bar)
        self.lock_controls = [self.video_button, self.demo_button, self.camera_button, self.floor, self.confirm, self.low,
                              self.clear_zone, self.load_config_button, self.save_config_button, self.realtime, self.runtime_button]
        self.refresh_events(force=True)
        self.show_event()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(50)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_events)
        self.refresh_timer.start(1000)

    @staticmethod
    def button(label, callback, layout):
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @staticmethod
    def number(minimum, maximum, step, value):
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setSingleStep(step)
        box.setValue(value)
        return box

    def error(self, message):
        self.state.setText("无法继续判断：" + message)
        QMessageBox.warning(self, "操作未完成", message)

    def update_runtime_label(self):
        variant = self.runtime.model_variant
        cached = "已缓存" if (ROOT / f"models/pose_landmarker_{variant}.task").exists() else "首次使用需下载"
        self.runtime_label.setText(f"设备：{self.runtime.device} · 模型：{variant}（{cached}）\n最长边：{self.runtime.max_side} · OpenCV 线程：{self.runtime.opencv_threads or '库默认'}\n不限定 CPU 品牌/型号；实际设备见运行结果。")

    def choose_runtime(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("运行设备与性能（开始前设置）")
        form = QFormLayout(dialog)
        device, model = QComboBox(), QComboBox()
        device.addItems(["auto", "cpu", "gpu"])
        device.setCurrentText(self.runtime.device)
        model.addItems(["lite", "full", "heavy"])
        model.setCurrentText(self.runtime.model_variant)
        side, threads = QSpinBox(), QSpinBox()
        side.setRange(256, 3840)
        side.setValue(self.runtime.max_side)
        threads.setRange(0, 64)
        threads.setValue(self.runtime.opencv_threads)
        form.addRow("设备", device)
        form.addRow("模型", model)
        form.addRow("处理最长边", side)
        form.addRow("OpenCV 线程（0=库默认）", threads)
        form.addRow(QLabel("auto：可用时尝试 GPU，失败回退并记录原因。\n当前 MediaPipe Python GPU 官方支持限 Ubuntu。\nOpenCV 线程不等于姿态模型线程。降低分辨率可能影响识别。"))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.runtime = RuntimeConfig(device.currentText(), model.currentText(), side.value(), threads.value())
            self.update_runtime_label()

    def set_zone(self, zone):
        self.config.zone = zone
        self.preview.zone = zone
        self.preview.update()
        self.zone_label.setText("区域提醒：" + (", ".join(f"{v:.2f}" for v in zone) if zone else "关闭"))

    def apply_config(self, config):
        self.config = config
        self.floor.setValue(config.floor_y)
        self.confirm.setValue(config.confirm_seconds)
        self.low.setValue(config.low_activity_seconds)
        self.set_zone(config.zone)

    def current_config(self):
        values = self.config.to_dict()
        values.update(floor_y=self.floor.value(), confirm_seconds=self.confirm.value(), low_activity_seconds=self.low.value())
        return Config(**values)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择本地视频", "", "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        if not path:
            return
        capture = cv2.VideoCapture(path)
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            self.error("无法读取视频首帧")
            return
        self.source = path
        self.apply_config(load_config(ROOT / "config.json"))
        self.preview.set_frame(frame)
        self.source_label.setText("视频识别：" + path + "（已恢复默认参数，请校准场景）")
        self.start_button.setEnabled(True)

    def choose_demo(self):
        self.source = str(ROOT / "examples/showcase.json")
        self.apply_config(load_config(ROOT / "examples/showcase_config.json"))
        self.preview.set_frame(np.full((540, 960, 3), 40, dtype=np.uint8))
        self.source_label.setText("合成关键点演示｜20 秒｜跳过姿态模型｜低活动阈值临时设为 4 秒")
        self.start_button.setEnabled(True)

    def choose_camera(self):
        self.source = 0
        self.apply_config(load_config(ROOT / "config.json"))
        self.source_label.setText("摄像头 0：点击“开始分析”才会打开。首次停止后可在画面上校准区域。")
        self.preview.image = QImage()
        self.preview.update()
        self.start_button.setEnabled(True)

    def start(self):
        if self.source is None or (self.worker and self.worker.isRunning()):
            return
        try:
            config = self.current_config()
        except ValueError as exc:
            self.error(str(exc))
            return
        self.worker = Worker(self.source, self.output, config, self.realtime.isChecked(), self.runtime)
        self.worker.completed.connect(self.completed)
        self.worker.failed.connect(self.error)
        self.worker.event_created.connect(self.notify_event)
        self.worker.finished.connect(self.unlocked)
        for control in self.lock_controls:
            control.setEnabled(False)
        self.preview.editable = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.state.setText("正在初始化…真实识别首次使用可能需要下载模型；尚不能判断人体状态。")
        self.worker.start()

    def stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.requestInterruption()
            self.state.setText("正在停止并保存片段；模型初始化或下载期间需等待当前操作返回。")
            self.stop_button.setEnabled(False)

    def notify_event(self, event_ids):
        # 提醒独立于可丢弃的预览帧；快速分析时不会随预览队列一起丢掉。
        QApplication.beep()
        self.refresh_events()

    def unlocked(self):
        for control in self.lock_controls:
            control.setEnabled(True)
        self.preview.editable = True
        self.start_button.setEnabled(self.source is not None)
        self.stop_button.setEnabled(False)
        self.refresh_events(force=True)

    def completed(self, summary):
        self.tick()
        label = "分析完成" if summary["completed"] else "已停止"
        runtime = summary["runtime"]
        self.state.setText(f"{label}｜{summary['frames']} 帧｜{summary['event_count']} 个事件｜无法判断 {summary['unknown_frames']} 帧｜处理吞吐 {summary['processing_fps']:.1f} FPS｜实际设备 {runtime['effective_device']}（不含初始化/预览等待）\n{runtime.get('fallback_reason') or ''}")
        self.update_runtime_label()

    def tick(self):
        if not self.worker:
            return
        latest = None
        while True:
            try:
                latest = self.worker.frames.get_nowait()
            except Empty:
                break
        if latest:
            frame, info = latest
            self.preview.set_frame(frame)
            self.state.setText(f"{SOURCE_NAMES[info['source_kind']]}｜{info['t']:.1f} 秒｜{STATE_NAMES[info['state']]}｜{info['reason']}｜处理 {info['processing_fps']:.1f} FPS｜设备 {info['runtime']['effective_device']}")

    def selected(self):
        index = self.table.currentRow()
        return self.rows[index] if 0 <= index < len(self.rows) else None

    def refresh_events(self, force=False):
        from datetime import datetime
        all_rows = self.store.events()
        counts = {key: sum(row["status"] == key for row in all_rows) for key in STATUSES}
        self.counts.setText("  /  ".join(f"{STATUSES[key]} {value}" for key, value in counts.items()) + "（含合成演示）")
        rows = [row for row in all_rows if not self.filter.currentData() or row["status"] == self.filter.currentData()]
        signature = json.dumps(rows, ensure_ascii=False)
        if not force and signature == self.signature:
            return
        selected = self.selected()
        selected_id = selected["id"] if selected else None
        self.signature, self.rows = signature, rows
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        selected_index = None
        for i, row in enumerate(rows):
            timestamp = datetime.fromisoformat(row["started_utc"]).astimezone().strftime("%m-%d %H:%M:%S")
            values = [timestamp, f"{row['time_s']:.1f}", KINDS[row["kind"]], STATUSES[row["status"]], SOURCE_NAMES[row["source_kind"]]]
            for j, value in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(value))
            if row["id"] == selected_id:
                selected_index = i
        self.table.clearSelection()
        self.table.setCurrentCell(-1, -1)
        if selected_index is not None:
            self.table.selectRow(selected_index)
        self.table.blockSignals(False)
        # 保持当前输入的复核备注，不在片段异步写盘时覆盖。
        if selected_id is None or selected_index is None:
            self.show_event()
        elif force:
            self.show_event()

    def show_event(self):
        row = self.selected()
        for button in self.review_controls:
            button.setEnabled(row is not None)
        self.note.setEnabled(row is not None)
        if row:
            self.note.setText(row["note"])
            self.details.setPlainText(f"{row['evidence']}\n输入：{row['source']}\n片段信息：{row['clip_info_json'] or '尚在收集后续画面'}")
        else:
            self.details.setPlainText("选择事件查看证据、回放片段并人工确认。正常状态不等于安全结论。")
            self.note.clear()

    def review(self, status):
        row = self.selected()
        if row:
            try:
                self.store.review(row["id"], status, self.note.text())
                self.refresh_events(force=True)
            except (ValueError, OSError) as exc:
                self.error(str(exc))

    def open_media(self, field):
        row = self.selected()
        if not row:
            return
        path = self.output / row[field] if row[field] else None
        if path is None or not path.is_file():
            self.error("片段仍在生成、编码失败或文件已移动，请稍后重试并检查片段信息。")
        elif not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            self.error("系统没有打开该文件的关联程序，请安装视频播放器或图片查看器。")

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出全部记录（请选择新文件名）", "events-export.json", "JSON (*.json)")
        if path:
            try:
                self.store.export(path)
                self.state.setText("记录已导出；视频与截图仍保存在本地 data 目录。")
            except (OSError, ValueError) as exc:
                self.error(str(exc))

    def save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出参数（请选择新文件名）", "scene-config.json", "JSON (*.json)")
        if path:
            try:
                with Path(path).open("x", encoding="utf-8") as stream:
                    json.dump({**self.current_config().to_dict(), "runtime": asdict(self.runtime)}, stream, ensure_ascii=False, indent=2)
            except (OSError, ValueError) as exc:
                self.error(str(exc))

    def choose_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "载入场景参数", "", "JSON (*.json)")
        if path:
            try:
                config, runtime = load_config(path), load_runtime(path)
                self.apply_config(config)
                self.runtime = runtime
                self.update_runtime_label()
            except (OSError, ValueError, TypeError) as exc:
                self.error(str(exc))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop()
            self.state.setText("正在保存，请待“已停止”出现后再关闭窗口。")
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    runtime_arguments(parser)
    args = parser.parse_args()
    application = QApplication(sys.argv[:1])
    configure_application(application)
    window = Window(args.data, runtime_from_args(args))
    window.show()
    sys.exit(application.exec())
