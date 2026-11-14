from __future__ import annotations

from functools import cache
from importlib import resources

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from deskguard.alarms import Banner
from deskguard.config import PrivacySettings
from deskguard.monitor import Detection, ObjectState, SceneMonitor

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ZONE = (255, 200, 0)
ACTOR = (220, 220, 220)
FIXTURE = (200, 140, 60)
OWNER = (0, 200, 0)
TOUCHED = (0, 140, 255)
STATE_COLORS = {
    ObjectState.CANDIDATE: (170, 170, 170),
    ObjectState.STATIONARY: (0, 215, 255),
    ObjectState.ABANDONED: (0, 0, 255),
    ObjectState.REMOVED: (255, 0, 255),
}
STATE_TEXT = {
    ObjectState.CANDIDATE: "yeni",
    ObjectState.ABANDONED: "terk edildi",
    ObjectState.REMOVED: "alındı",
}
KEYS_HELP = "r: alan   a: kur/kapat   b: bulanık   c: sıfırla   q: çıkış"
ZONE_HELP = "Sol tık: nokta ekle   Sağ tık: geri al   Enter: onayla   Esc: iptal"


@cache
def font(size: int) -> ImageFont.FreeTypeFont:
    path = resources.files("deskguard").joinpath("DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size)


def text_size(text: str, size: int) -> tuple[int, int]:
    _, _, right, bottom = font(size).getbbox(text)
    return right + 6, bottom + 6


def put_text(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    size: int = 16,
    color: tuple[int, int, int] = WHITE,
    background: tuple[int, int, int] | None = None,
) -> int:
    width, height = text_size(text, size)
    x, y = max(0, int(x)), max(0, int(y))
    x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
    if x2 <= x or y2 <= y:
        return 0
    if background is not None:
        frame[y:y2, x:x2] = background
    patch = Image.fromarray(cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(patch).text((3, 3), text, font=font(size), fill=color[::-1])
    frame[y:y2, x:x2] = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
    return width


def label_box(frame: np.ndarray, box: np.ndarray, text: str, color, thickness: int = 2) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    width, height = text_size(text, 15)
    top = y1 - height if y1 >= height else y2
    left = min(x1, frame.shape[1] - width)
    put_text(frame, text, left, top, 15, BLACK if sum(color) > 400 else WHITE, color)


def blur_heads(frame: np.ndarray, persons: list[np.ndarray], settings: PrivacySettings) -> None:
    kernel = settings.kernel_size | 1
    height, width = frame.shape[:2]
    for box in persons:
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        bottom = y1 + int((y2 - y1) * settings.head_ratio)
        if bottom > y1 and x2 > x1:
            frame[y1:bottom, x1:x2] = cv2.GaussianBlur(frame[y1:bottom, x1:x2], (kernel, kernel), 0)


def draw_scene(
    frame: np.ndarray,
    monitor: SceneMonitor,
    actors: list[Detection],
    banners: list[Banner],
    names: dict[str, str],
    now: float,
    fps: float,
    blur: bool,
) -> None:
    height, width = frame.shape[:2]
    armed = monitor.is_active(now)
    cv2.polylines(frame, [monitor.roi.astype(np.int32)], True, ZONE, 2)
    for actor in actors:
        text = f"{names.get(actor.label, actor.label)} {actor.confidence:.2f}"
        label_box(frame, actor.box, text, ACTOR, 1)
    for obj in monitor.objects.values():
        text = f"#{obj.object_id} {names.get(obj.label, obj.label)}"
        limit = monitor.settings.abandon_seconds
        if not obj.valuable:
            color = FIXTURE
        elif obj.state is ObjectState.STATIONARY and obj.last_attended == now and armed:
            color = TOUCHED
            text += " dokunuluyor"
        elif obj.state is ObjectState.STATIONARY and obj.last_attended == now:
            color = OWNER
            text += " sahibi yanında"
        elif obj.state is ObjectState.STATIONARY:
            color = STATE_COLORS[obj.state]
            text += f" sahipsiz {now - obj.last_attended:.0f}/{limit:.0f} sn"
        else:
            color = STATE_COLORS[obj.state]
            text += f" {STATE_TEXT[obj.state]}"
        alert = obj.state in (ObjectState.ABANDONED, ObjectState.REMOVED)
        label_box(frame, obj.box, text, color, 3 if alert else 2)

    top = 30
    for banner in banners:
        _, line = text_size(banner.text, 20)
        cv2.rectangle(
            frame,
            (0, top),
            (width, top + line + 4),
            (0, 0, 190) if banner.is_alarm else (0, 130, 0),
            -1,
        )
        put_text(frame, banner.text, 8, top + 2, 20)
        top += line + 8

    remaining = monitor.arming_remaining(now)
    if not monitor.armed:
        status, color = "DEVRE DIŞI", (0, 255, 0)
    elif remaining > 0:
        status, color = f"KURULUYOR {remaining:.0f} sn - alandan çıkın", (0, 200, 255)
    else:
        status, color = "KURULU", (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (width, 28), BLACK, -1)
    left = put_text(frame, status, 4, 2, 16, color)
    put_text(
        frame,
        f"|  FPS {fps:.0f}  |  bulanıklaştırma {'açık' if blur else 'kapalı'}",
        4 + left,
        2,
        16,
    )
    help_width, help_height = text_size(KEYS_HELP, 14)
    cv2.rectangle(frame, (0, height - help_height), (width, height), BLACK, -1)
    put_text(frame, KEYS_HELP, (width - help_width) // 2, height - help_height, 14)


def draw_detections(
    frame: np.ndarray, detections: list[Detection], names: dict[str, str]
) -> np.ndarray:
    canvas = frame.copy()
    for d in detections:
        text = f"{names.get(d.label, d.label)} {d.confidence:.2f}"
        label_box(canvas, d.box, text, ACTOR if d.is_actor else (0, 215, 255))
    return canvas


def select_polygon(frame: np.ndarray, window: str) -> np.ndarray | None:
    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            canvas = frame.copy()
            if points:
                cv2.polylines(canvas, [np.array(points, np.int32)], len(points) > 2, ZONE, 2)
                for point in points:
                    cv2.circle(canvas, point, 4, ZONE, -1)
            put_text(canvas, ZONE_HELP, 4, 4, 16, WHITE, BLACK)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key in (10, 13) and len(points) >= 3:
                return np.array(points, dtype=np.float32)
            if key == 27:
                return None
    finally:
        cv2.setMouseCallback(window, lambda *_: None)
