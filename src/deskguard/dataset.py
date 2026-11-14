from __future__ import annotations

import random
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from deskguard.app import IMAGE_SUFFIXES, list_images
from deskguard.config import Config
from deskguard.detector import Detector, filter_detections
from deskguard.monitor import Detection, iou
from deskguard.ui import draw_detections

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def to_yolo_lines(
    detections: list[Detection], class_ids: dict[str, int], width: int, height: int
) -> list[str]:
    lines = []
    for d in detections:
        x1, y1 = max(0.0, float(d.box[0])), max(0.0, float(d.box[1]))
        x2, y2 = min(float(width), float(d.box[2])), min(float(height), float(d.box[3]))
        if x2 > x1 and y2 > y1:
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            w, h = (x2 - x1) / width, (y2 - y1) / height
            lines.append(f"{class_ids[d.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def split_items(
    items: list[Path], val_ratio: float, test_ratio: float, seed: int = 0
) -> dict[str, list[Path]]:
    shuffled = sorted(items)
    random.Random(seed).shuffle(shuffled)
    test = round(len(shuffled) * test_ratio)
    val = max(1, round(len(shuffled) * val_ratio)) if len(shuffled) - test > 1 else 0
    return {
        "test": shuffled[:test],
        "val": shuffled[test : test + val],
        "train": shuffled[test + val :],
    }


def write_data_yaml(root: Path, labels: list[str]) -> Path:
    data = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(labels)),
    }
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            (root / kind / split).mkdir(parents=True, exist_ok=True)
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), "utf-8")
    return data_yaml


def relabel(detections: list[Detection], label: str, config: Config) -> list[Detection]:
    items = [
        Detection(d.box, label, d.confidence, d.track_id) for d in detections if not d.is_actor
    ]
    actors = [d for d in detections if d.is_actor]
    return filter_detections(actors + items, float("inf"), config.model)


def autolabel(
    config: Config,
    images_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.15,
    confidence: float = 0.35,
    only: str | None = None,
) -> Path:
    images = list_images(images_dir)
    if not images:
        raise FileNotFoundError(f"No images in {images_dir}")
    detector = Detector(config)
    class_ids = {label: i for i, label in enumerate(config.labels)}
    root = Path(output_dir)
    data_yaml = write_data_yaml(root, config.labels)
    (root / "preview").mkdir(exist_ok=True)
    for split, members in split_items(images, val_ratio, test_ratio).items():
        for image in members:
            frame = cv2.imread(str(image))
            if frame is None:
                continue
            detections = detector(frame, confidence=confidence)
            if only:
                detections = relabel(detections, only, config)
            lines = to_yolo_lines(detections, class_ids, frame.shape[1], frame.shape[0])
            shutil.copy2(image, root / "images" / split / image.name)
            (root / "labels" / split / f"{image.stem}.txt").write_text("\n".join(lines), "utf-8")
            preview = draw_detections(frame, detections, config.names)
            cv2.imwrite(str(root / "preview" / image.name), preview)
    return data_yaml


@dataclass(frozen=True)
class Sample:
    image: Path
    boxes: np.ndarray
    labels: tuple[str, ...]


def load_split(data_yaml: str, split: str) -> list[Sample]:
    spec = yaml.safe_load(Path(data_yaml).read_text("utf-8"))
    root = Path(spec.get("path") or Path(data_yaml).parent)
    images_dir = root / spec[split]
    labels_dir = root / Path(spec[split].replace("images", "labels", 1))
    names = spec["names"]
    names = (
        dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    )
    samples = []
    for image in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        height, width = cv2.imread(str(image)).shape[:2]
        boxes, labels = [], []
        label_file = labels_dir / f"{image.stem}.txt"
        for line in label_file.read_text("utf-8").splitlines() if label_file.exists() else []:
            cls, cx, cy, w, h = line.split()[:5]
            cx, cy, w, h = (
                float(cx) * width,
                float(cy) * height,
                float(w) * width,
                float(h) * height,
            )
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            labels.append(names[int(cls)])
        samples.append(Sample(image, np.array(boxes, np.float32).reshape(-1, 4), tuple(labels)))
    return samples


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(
        np.mean(np.where(index < recall.size, envelope[np.minimum(index, recall.size - 1)], 0.0))
    )


def class_metrics(
    pairs: list[tuple[list[Detection], np.ndarray]], operating_confidence: float
) -> tuple[float, float, float, float]:
    total = sum(len(truth) for _, truth in pairs)
    aps, precision_op, recall_op = [], 0.0, 0.0
    for threshold in IOU_THRESHOLDS:
        scores, hits = [], []
        for predictions, truth in pairs:
            used: set[int] = set()
            for p in sorted(predictions, key=lambda p: p.confidence, reverse=True):
                overlaps = [(iou(p.box, t), i) for i, t in enumerate(truth) if i not in used]
                best = max(overlaps, default=(0.0, -1))
                hit = best[0] >= threshold
                if hit:
                    used.add(best[1])
                scores.append(p.confidence)
                hits.append(hit)
        if not total or not scores:
            aps.append(0.0)
            continue
        order = np.argsort(-np.array(scores), kind="stable")
        tp = np.cumsum(np.array(hits, float)[order])
        recall = tp / total
        precision = tp / np.arange(1, len(tp) + 1)
        aps.append(average_precision(recall, precision))
        kept = np.nonzero(np.array(scores)[order] >= operating_confidence)[0]
        if threshold == IOU_THRESHOLDS[0] and kept.size:
            precision_op, recall_op = float(precision[kept[-1]]), float(recall[kept[-1]])
    return aps[0], float(np.mean(aps)), precision_op, recall_op


def evaluate(
    config: Config, data_yaml: str, split: str, weights_list: list[str], output_dir: str
) -> str:
    samples = load_split(data_yaml, split)
    detectors = {
        Path(w).stem: Detector(replace(config, model=replace(config.model, weights=w)))
        for w in weights_list
    }
    labels = sorted(
        {label for s in samples for label in s.labels}.intersection(
            *(d.labels for d in detectors.values())
        )
    )
    summary = ["| Model | FPS | mAP50 | mAP50-95 |", "| --- | ---: | ---: | ---: |"]
    details = [
        "| Model | Class | n | AP50 | Precision | Recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, detector in detectors.items():
        predictions, timings = [], []
        for sample in samples:
            frame = cv2.imread(str(sample.image))
            started = time.perf_counter()
            predictions.append(detector(frame, confidence=0.01))
            timings.append(time.perf_counter() - started)
        rows = []
        for label in labels:
            pairs = [
                (
                    [p for p in preds if p.label == label],
                    s.boxes[[i for i, x in enumerate(s.labels) if x == label]],
                )
                for s, preds in zip(samples, predictions, strict=True)
            ]
            ap50, ap, precision, recall = class_metrics(pairs, config.model.confidence)
            count = sum(len(t) for _, t in pairs)
            rows.append((ap50, ap))
            details.append(
                f"| {name} | {label} | {count} | {ap50:.3f} | {precision:.3f} | {recall:.3f} |"
            )
        fps = 1 / float(np.mean(timings[1:] or timings))
        map50, map_all = (
            (float(np.mean(v)) for v in zip(*rows, strict=True)) if rows else (0.0, 0.0)
        )
        summary.append(f"| {name} | {fps:.1f} | {map50:.3f} | {map_all:.3f} |")
    report = "\n".join(summary + [""] + details) + "\n"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.md").write_text(report, "utf-8")
    return report


WEB_DATASETS = [
    ("yolo-object-detection-custom-dataset", "study-desk-items"),
    ("obj-detection-kd5zb", "left-off-objects-1"),
    ("mangoesclassifierdetector", "lost-found-items"),
]


def web_label(name: str) -> str | None:
    name = name.lower()
    if "phone" in name and "charger" not in name:
        return "phone"
    if "key" in name and "keyboard" not in name:
        return "keys"
    if "wallet" in name:
        return "wallet"
    if "glasses" in name:
        return "glasses"
    if any(word in name for word in ("bottle", "can", "cup", "mug")):
        return "drink"
    return None


def download_web_data(api_key: str, raw_dir: str = "data/web_raw") -> list[Path]:
    from roboflow import Roboflow

    client = Roboflow(api_key=api_key)
    folders = []
    for workspace, project_id in WEB_DATASETS:
        project = client.workspace(workspace).project(project_id)
        for number in range(20, 0, -1):
            try:
                version = project.version(number)
            except Exception:
                continue
            target = Path(raw_dir) / project_id
            version.download("yolov8", location=str(target), overwrite=True)
            folders.append(target)
            break
    return folders


def merge_web_data(
    config: Config,
    raw_dir: str = "data/web_raw",
    output_dir: str = "data/train",
    detector: Detector | None = None,
) -> Path:
    root = Path(output_dir)
    data_yaml = write_data_yaml(root, config.labels)
    class_ids = {label: i for i, label in enumerate(config.labels)}
    counts = dict.fromkeys(config.labels, 0)
    images = 0
    for spec_path in sorted(Path(raw_dir).glob("*/data.yaml")):
        names = yaml.safe_load(spec_path.read_text("utf-8"))["names"]
        names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
        mapping = {i: web_label(n) for i, n in enumerate(names)}
        for label_file in sorted(spec_path.parent.glob("*/labels/*.txt")):
            split = "val" if label_file.parent.parent.name in ("valid", "test") else "train"
            lines = []
            for line in label_file.read_text("utf-8").splitlines():
                parts = line.split()
                target = mapping.get(int(parts[0])) if len(parts) == 5 else None
                if target:
                    lines.append(" ".join([str(class_ids[target]), *parts[1:]]))
                    counts[target] += 1
            images_dir = label_file.parent.parent / "images"
            image = next(images_dir.glob(f"{label_file.stem}.*"), None)
            if not lines or image is None:
                continue
            if detector is not None:
                frame = cv2.imread(str(image))
                actors = [d for d in detector(frame) if d.is_actor]
                lines += to_yolo_lines(actors, class_ids, frame.shape[1], frame.shape[0])
                for actor in actors:
                    counts[actor.label] += 1
            stem = f"{spec_path.parent.name}_{label_file.stem}"
            shutil.copy2(image, root / "images" / split / f"{stem}{image.suffix}")
            (root / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), "utf-8")
            images += 1
    print(f"{images} images: {counts}")
    return data_yaml
