from __future__ import annotations

from pathlib import Path

import numpy as np

from deskguard.config import Config, ModelSettings
from deskguard.monitor import Detection, box_area, iou


def is_open_vocabulary(weights: str) -> bool:
    return "yoloe" in Path(weights).name.lower()


def filter_detections(
    detections: list[Detection], frame_area: float, settings: ModelSettings
) -> list[Detection]:
    actors = [d for d in detections if d.is_actor and d.confidence >= settings.actor_confidence]
    items = [
        d
        for d in detections
        if not d.is_actor
        and box_area(d.box) <= settings.max_box_ratio * frame_area
        and all(iou(d.box, a.box) < settings.actor_overlap_iou for a in actors)
    ]
    kept: list[Detection] = []
    duplicate_iou = settings.duplicate_iou
    detections = actors + items
    for d in sorted(detections, key=lambda d: d.confidence, reverse=True):
        if all(k.label != d.label or iou(k.box, d.box) < duplicate_iou for k in kept):
            kept.append(d)
    return kept


class Detector:
    def __init__(self, config: Config) -> None:
        self.settings = config.model
        weights = self.settings.weights
        if is_open_vocabulary(weights):
            from ultralytics import YOLOE

            self.model = YOLOE(weights)
            prompts, labels = config.vocabulary()
            self.model.set_classes(prompts, self.model.get_text_pe(prompts))
            self.mapping = dict(enumerate(labels))
        else:
            from ultralytics import YOLO

            self.model = YOLO(weights)
            self.mapping = {int(i): n for i, n in self.model.names.items() if n in config.labels}
        self.labels = set(self.mapping.values())

    def __call__(
        self, frame: np.ndarray, track: bool = False, confidence: float | None = None
    ) -> list[Detection]:
        options = {
            "conf": self.settings.confidence if confidence is None else confidence,
            "imgsz": self.settings.image_size,
            "verbose": False,
        }
        if self.settings.device:
            options["device"] = self.settings.device
        if track:
            result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **options)[0]
        else:
            result = self.model.predict(frame, **options)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        detections = [
            Detection(box, self.mapping[cls], float(score), tid)
            for box, cls, score, tid in zip(
                boxes.xyxy.cpu().numpy().astype(np.float32),
                boxes.cls.int().tolist(),
                boxes.conf.tolist(),
                ids,
                strict=True,
            )
            if cls in self.mapping
        ]
        height, width = frame.shape[:2]
        return filter_detections(detections, width * height, self.settings)
