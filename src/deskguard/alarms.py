from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from deskguard.config import AlarmSettings
from deskguard.monitor import Event, EventKind

MESSAGES = {
    EventKind.ABANDONED: "TERK EDİLDİ: {name} #{object_id}",
    EventKind.REMOVED: "ALINDI: {name} #{object_id}",
    EventKind.RECLAIMED: "SAHİBİ DÖNDÜ: {name} #{object_id}",
    EventKind.INTRUSION: "İZİNSİZ GİRİŞ: korunan alana girildi",
}


def describe(event: Event, names: dict[str, str]) -> str:
    name = names.get(event.label or "", event.label)
    return MESSAGES[event.kind].format(name=name, object_id=event.object_id)


@dataclass
class Banner:
    text: str
    expires_at: float
    is_alarm: bool


class AlarmSink:
    def __init__(self, settings: AlarmSettings, names: dict[str, str]) -> None:
        self.settings = settings
        self.names = names
        self.output_dir = Path(settings.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "events.csv"
        self.banners: list[Banner] = []
        self._last_fired: dict[str, float] = {}
        self._pending: list[Event] = []

    def handle(self, events: list[Event], now: float) -> None:
        for event in events:
            if now - self._last_fired.get(event.key, -1e9) < self.settings.cooldown_seconds:
                continue
            self._last_fired[event.key] = now
            text = describe(event, self.names)
            self.banners.append(Banner(text, now + self.settings.banner_seconds, event.is_alarm))
            self._pending.append(event)
            print(text)
            if event.is_alarm and self.settings.beep:
                print("\a", end="", flush=True)
        self.banners = [b for b in self.banners if b.expires_at > now]

    def flush(self, frame: np.ndarray) -> None:
        if not self._pending:
            return
        is_new = not self.log_path.exists()
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(["time", "video_time", "event", "label", "object_id", "snapshot"])
            for event in self._pending:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                parts = [stamp, event.kind.value, event.label, event.object_id]
                snapshot = "_".join(str(p) for p in parts if p is not None) + ".jpg"
                if event.is_alarm:
                    cv2.imwrite(str(self.output_dir / snapshot), frame)
                writer.writerow(
                    [
                        stamp,
                        f"{event.timestamp:.2f}",
                        event.kind.value,
                        event.label or "",
                        "" if event.object_id is None else event.object_id,
                        snapshot if event.is_alarm else "",
                    ]
                )
        self._pending.clear()
