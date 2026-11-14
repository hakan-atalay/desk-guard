from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from deskguard.alarms import AlarmSink
from deskguard.config import PERSON, Config
from deskguard.detector import Detector
from deskguard.monitor import SceneMonitor
from deskguard.ui import BLACK, blur_heads, draw_detections, draw_scene, put_text, select_polygon

WINDOW = "DeskGuard"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def open_capture(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return capture


def run(config: Config, source: str = "0", record: str | None = None) -> None:
    capture = open_capture(source)
    live = source.isdigit()
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError("Video source returned no frames")
    height, width = frame.shape[:2]

    detector = Detector(config)
    monitor = SceneMonitor(config.monitor, config.valuable, (width, height))
    alarms = AlarmSink(config.alarms, config.names)
    writer = None
    if record:
        Path(record).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            record, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
        )
    blur = config.privacy.blur_faces
    started = last = time.monotonic()
    fps = 0.0
    frame_index = 0
    cv2.namedWindow(WINDOW)

    try:
        while ok:
            frame_index += 1
            now = time.monotonic() - started if live else frame_index / video_fps
            detections = detector(frame, track=True)
            actors = [d for d in detections if d.is_actor]
            if blur:
                blur_heads(frame, [d.box for d in detections if d.label == PERSON], config.privacy)
            alarms.handle(monitor.update(detections, now), now)

            tick = time.monotonic()
            instant = 1 / max(tick - last, 1e-6)
            fps = 0.9 * fps + 0.1 * instant if fps else instant
            last = tick
            draw_scene(frame, monitor, actors, alarms.banners, config.names, now, fps, blur)
            alarms.flush(frame)
            if writer:
                writer.write(frame)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a") and monitor.armed:
                monitor.disarm()
            elif key == ord("a"):
                monitor.arm(now)
            elif key == ord("b"):
                blur = not blur
            elif key == ord("c"):
                monitor.objects.clear()
            elif key == ord("r"):
                polygon = select_polygon(frame, WINDOW)
                if polygon is not None:
                    monitor.set_roi(polygon)
            ok, frame = capture.read()
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def list_images(source: str | Path) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def detect_images(config: Config, source: str, output_dir: str) -> Path:
    detector = Detector(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for image in list_images(source):
        frame = cv2.imread(str(image))
        if frame is None:
            continue
        detections = detector(frame)
        if config.privacy.blur_faces:
            blur_heads(frame, [d.box for d in detections if d.label == PERSON], config.privacy)
        cv2.imwrite(str(output / image.name), draw_detections(frame, detections, config.names))
        results[image.name] = [
            {
                "label": d.label,
                "confidence": round(d.confidence, 3),
                "box": [round(float(v), 1) for v in d.box],
            }
            for d in detections
        ]
        print(
            f"{image.name}: {', '.join(sorted({config.names[d.label] for d in detections})) or '-'}"
        )
    report = output / "detections.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def collect(output_dir: str, camera: int = 0, interval: float = 1.0) -> int:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    capture = open_capture(str(camera))
    saved = len(list(output.glob("*.jpg")))
    auto, last = False, 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            key = cv2.waitKey(1) & 0xFF
            now = time.monotonic()
            if key == ord(" ") or (auto and now - last >= interval):
                cv2.imwrite(str(output / f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"), frame)
                saved, last = saved + 1, now
            if key == ord("a"):
                auto = not auto
            if key == ord("q"):
                break
            mode = "açık" if auto else "kapalı"
            status = f"kaydedilen {saved}  |  otomatik {mode}  |  boşluk: kaydet  a: oto  q: çık"
            preview = frame.copy()
            put_text(preview, status, 0, 0, 16, (0, 255, 0), BLACK)
            cv2.imshow(WINDOW, preview)
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return saved
