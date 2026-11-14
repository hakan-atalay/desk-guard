from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from deskguard.config import ACTORS, MonitorSettings


def box_center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32)


def box_area(box: np.ndarray) -> float:
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = box_area(a) + box_area(b) - w * h
    return float(w * h / union) if union > 0 else 0.0


def expand_box(box: np.ndarray, ratio: float) -> np.ndarray:
    dx, dy = (box[2] - box[0]) * ratio, (box[3] - box[1]) * ratio
    return np.array([box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy], dtype=np.float32)


def overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = float(point[0]), float(point[1])
    inside = False
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


class ObjectState(str, Enum):
    CANDIDATE = "candidate"
    STATIONARY = "stationary"
    ABANDONED = "abandoned"
    REMOVED = "removed"


class EventKind(str, Enum):
    ABANDONED = "abandoned"
    REMOVED = "removed"
    RECLAIMED = "reclaimed"
    INTRUSION = "intrusion"


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    label: str
    confidence: float
    track_id: int | None = None

    @property
    def is_actor(self) -> bool:
        return self.label in ACTORS


@dataclass(frozen=True)
class Event:
    kind: EventKind
    timestamp: float
    object_id: int | None = None
    label: str | None = None

    @property
    def is_alarm(self) -> bool:
        return self.kind is not EventKind.RECLAIMED

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.object_id}"


@dataclass
class TrackedObject:
    object_id: int
    label: str
    valuable: bool
    box: np.ndarray
    track_id: int | None
    last_seen: float
    anchor: np.ndarray
    still_since: float
    last_attended: float
    state: ObjectState = ObjectState.CANDIDATE
    missing_since: float | None = None

    def missing_for(self, now: float) -> float:
        return 0.0 if self.missing_since is None else now - self.missing_since


class SceneMonitor:
    def __init__(
        self, settings: MonitorSettings, valuable: dict[str, bool], frame_size: tuple[int, int]
    ) -> None:
        width, height = frame_size
        self.settings = settings
        self.valuable = dict(valuable)
        self.move_tolerance = settings.move_ratio * float(np.hypot(width, height))
        self.roi = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        self.objects: dict[int, TrackedObject] = {}
        self._next_id = 1
        self._armed_at: float | None = None
        self._intruder = False
        self._clear_since: float | None = None

    @property
    def armed(self) -> bool:
        return self._armed_at is not None

    def arm(self, now: float) -> None:
        self._armed_at = now
        self._intruder = False
        self._clear_since = None

    def disarm(self) -> None:
        self._armed_at = None

    def arming_remaining(self, now: float) -> float:
        if self._armed_at is None:
            return 0.0
        return max(0.0, self.settings.arm_delay_seconds - (now - self._armed_at))

    def is_active(self, now: float) -> bool:
        return self.armed and self.arming_remaining(now) == 0.0

    def set_roi(self, polygon: np.ndarray) -> None:
        self.roi = np.asarray(polygon, dtype=np.float32)
        self.objects.clear()

    def update(self, detections: list[Detection], now: float) -> list[Event]:
        actors = [d.box for d in detections if d.is_actor]
        items = [
            d
            for d in detections
            if d.label in self.valuable and point_in_polygon(box_center(d.box), self.roi)
        ]
        seen, events = self._associate(items, now)
        events += self._attend(actors, now)
        events += self._advance(seen, actors, now)
        if self._intrusion(actors, now):
            events.append(Event(EventKind.INTRUSION, now))
        return events

    def _intrusion(self, actors: list[np.ndarray], now: float) -> bool:
        if not self.is_active(now):
            return False
        if any(point_in_polygon(box_center(a), self.roi) for a in actors):
            self._clear_since = None
            new = not self._intruder
            self._intruder = True
            return new
        if self._clear_since is None:
            self._clear_since = now
        if now - self._clear_since >= self.settings.intrusion_reset_seconds:
            self._intruder = False
        return False

    def _associate(self, items: list[Detection], now: float) -> tuple[set[int], list[Event]]:
        seen: set[int] = set()
        events: list[Event] = []
        for detection in sorted(items, key=lambda d: d.confidence, reverse=True):
            obj = self._match(detection, seen)
            if obj is not None and self._taken(obj, detection, now):
                obj.state, obj.missing_since = ObjectState.REMOVED, now
                events.append(Event(EventKind.REMOVED, now, obj.object_id, obj.label))
                seen.add(obj.object_id)
                obj = None
            if obj is None:
                obj = TrackedObject(
                    object_id=self._next_id,
                    label=detection.label,
                    valuable=self.valuable[detection.label],
                    box=detection.box,
                    track_id=detection.track_id,
                    last_seen=now,
                    anchor=box_center(detection.box),
                    still_since=now,
                    last_attended=now,
                )
                self.objects[obj.object_id] = obj
                self._next_id += 1
            else:
                obj.box, obj.track_id, obj.last_seen, obj.missing_since = (
                    detection.box,
                    detection.track_id,
                    now,
                    None,
                )
                center = box_center(detection.box)
                if float(np.linalg.norm(center - obj.anchor)) > self.move_tolerance:
                    obj.anchor, obj.still_since, obj.state = center, now, ObjectState.CANDIDATE
            seen.add(obj.object_id)
        return seen, events

    def _taken(self, obj: TrackedObject, detection: Detection, now: float) -> bool:
        settled = obj.state in (ObjectState.STATIONARY, ObjectState.ABANDONED)
        moved = float(np.linalg.norm(box_center(detection.box) - obj.anchor))
        return settled and obj.valuable and self.is_active(now) and moved > 2 * self.move_tolerance

    def _match(self, detection: Detection, seen: set[int]) -> TrackedObject | None:
        candidates = [
            o
            for o in self.objects.values()
            if o.label == detection.label
            and o.object_id not in seen
            and o.state is not ObjectState.REMOVED
        ]
        for obj in candidates:
            if detection.track_id is not None and obj.track_id == detection.track_id:
                return obj
        best, best_score = None, self.settings.match_iou
        for obj in candidates:
            score = iou(detection.box, obj.box)
            if score > best_score:
                best, best_score = obj, score
        return best

    def _attend(self, actors: list[np.ndarray], now: float) -> list[Event]:
        events = []
        zones = [expand_box(a, self.settings.owner_margin) for a in actors]
        for obj in self.objects.values():
            if obj.state is ObjectState.REMOVED or not any(overlaps(z, obj.box) for z in zones):
                continue
            obj.last_attended = now
            if obj.state is ObjectState.ABANDONED:
                obj.state = ObjectState.STATIONARY
                events.append(Event(EventKind.RECLAIMED, now, obj.object_id, obj.label))
        return events

    def _advance(self, seen: set[int], actors: list[np.ndarray], now: float) -> list[Event]:
        s = self.settings
        events = []
        for object_id, obj in list(self.objects.items()):
            if object_id in seen:
                if obj.state is ObjectState.CANDIDATE and now - obj.still_since >= s.settle_seconds:
                    obj.state = ObjectState.STATIONARY
                if (
                    obj.state is ObjectState.STATIONARY
                    and obj.valuable
                    and now - obj.last_attended >= s.abandon_seconds
                ):
                    obj.state = ObjectState.ABANDONED
                    events.append(Event(EventKind.ABANDONED, now, obj.object_id, obj.label))
            elif obj.state is ObjectState.CANDIDATE:
                if now - obj.last_seen > s.forget_seconds:
                    del self.objects[object_id]
            elif obj.state is ObjectState.REMOVED:
                if obj.missing_for(now) > s.removal_seconds + s.removed_display_seconds:
                    del self.objects[object_id]
            elif not self.is_active(now) and any(overlaps(a, obj.box) for a in actors):
                obj.missing_since = None
            else:
                if obj.missing_since is None:
                    obj.missing_since = now
                if obj.missing_for(now) >= s.removal_seconds:
                    obj.state = ObjectState.REMOVED
                    if obj.valuable and self.is_active(now):
                        events.append(Event(EventKind.REMOVED, now, obj.object_id, obj.label))
        return events
