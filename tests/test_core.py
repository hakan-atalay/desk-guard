import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from deskguard.alarms import AlarmSink, describe
from deskguard.config import AlarmSettings, ModelSettings, load_config, parse_config
from deskguard.dataset import class_metrics, evaluate, split_items, to_yolo_lines
from deskguard.detector import filter_detections, is_open_vocabulary
from deskguard.monitor import Detection, Event, EventKind, iou, point_in_polygon
from deskguard.ui import put_text


def det(box, label="keys", conf=0.9):
    return Detection(np.array(box, dtype=np.float32), label, conf)


def test_default_config():
    config = load_config()
    prompts, labels = config.vocabulary()
    assert config.labels[:2] == ["person", "hand"]
    assert labels[prompts.index("tea glass")] == "drink"
    assert config.valuable["phone"] and not config.valuable["ashtray"]
    assert config.names["keys"] == "Anahtar"


def test_reserved_labels_rejected():
    with pytest.raises(ValueError):
        parse_config({"targets": [{"label": "hand"}]})


def test_geometry():
    assert iou(np.float32([0, 0, 10, 10]), np.float32([5, 0, 15, 10])) == pytest.approx(1 / 3)
    concave = np.float32([[0, 0], [10, 0], [10, 10], [5, 5], [0, 10]])
    assert point_in_polygon(np.float32([2, 2]), concave)
    assert not point_in_polygon(np.float32([5, 8]), concave)


def test_filter_detections():
    detections = [
        det([0, 0, 640, 300], "glasses_case", 0.4),
        det([280, 380, 420, 460], "keys", 0.8),
        det([282, 381, 420, 460], "keys", 0.6),
        det([0, 0, 640, 480], "person", 0.9),
        det([50, 50, 120, 140], "hand", 0.7),
        det([52, 48, 121, 138], "drink", 0.5),
        det([500, 400, 560, 470], "hand", 0.2),
    ]
    kept = filter_detections(detections, 640 * 480, ModelSettings())
    assert [(d.label, d.confidence) for d in kept] == [
        ("person", 0.9),
        ("keys", 0.8),
        ("hand", 0.7),
    ]
    assert is_open_vocabulary("yoloe-26s-seg.pt") and not is_open_vocabulary("runs/best.pt")


def test_alarm_log_and_messages(tmp_path):
    names = load_config().names
    sink = AlarmSink(AlarmSettings(output_dir=str(tmp_path), beep=False), names)
    removed = Event(EventKind.REMOVED, 12.5, 3, "keys")
    assert describe(removed, names) == "ALINDI: Anahtar #3"
    sink.handle([removed, Event(EventKind.INTRUSION, 12.5)], 12.5)
    sink.handle([removed], 13.0)
    sink.flush(np.zeros((20, 20, 3), dtype=np.uint8))
    rows = list(csv.reader((tmp_path / "events.csv").open(encoding="utf-8")))
    assert [r[2] for r in rows[1:]] == ["removed", "intrusion"]
    assert rows[1][5].endswith("_removed_keys_3.jpg") and rows[2][5].endswith("_intrusion.jpg")
    assert (tmp_path / rows[1][5]).exists()


def test_turkish_text_drawn_at_border():
    frame = np.zeros((40, 120, 3), dtype=np.uint8)
    assert put_text(frame, "İZİNSİZ GİRİŞ çok uzun", 100, 30) > 0
    assert frame.any()


def test_yolo_lines_and_split():
    (line,) = to_yolo_lines([det([-10, 0, 50, 100])], {"keys": 2}, 100, 200)
    assert [float(v) for v in line.split()] == pytest.approx([2, 0.25, 0.25, 0.5, 0.5])
    splits = split_items([Path(f"{i}.jpg") for i in range(20)], 0.2, 0.15, seed=1)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [13, 4, 3]


def test_metrics():
    truth = np.float32([[0, 0, 10, 10]])
    assert class_metrics([([det([0, 0, 10, 10])], truth)], 0.25) == pytest.approx((1, 1, 1, 1))
    pairs = [([det([0, 0, 10, 10]), det([80, 80, 90, 90], conf=0.8)], truth), ([], truth)]
    ap50, _, precision, recall = class_metrics(pairs, 0.25)
    assert 0 < ap50 < 1 and precision == 0.5 and recall == 0.5


class FakeDetector:
    def __init__(self, config):
        self.labels = {"person", "keys"}

    def __call__(self, frame, track=False, confidence=None):
        return [det([10, 10, 30, 30])]


def test_evaluate_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("deskguard.dataset.Detector", FakeDetector)
    for folder in ("images/test", "labels/test"):
        (tmp_path / folder).mkdir(parents=True)
    for name in ("a", "b"):
        cv2.imwrite(str(tmp_path / f"images/test/{name}.jpg"), np.zeros((100, 100, 3), np.uint8))
        (tmp_path / f"labels/test/{name}.txt").write_text("1 0.2 0.2 0.2 0.2\n", "utf-8")
    data = tmp_path / "data.yaml"
    names = {0: "person", 1: "keys"}
    data.write_text(yaml.safe_dump({"path": str(tmp_path), "test": "images/test", "names": names}))
    report = evaluate(load_config(), str(data), "test", ["best.pt"], str(tmp_path / "eval"))
    assert "| best |" in report and "| best | keys | 2 | 1.000 |" in report


def test_web_labels_are_mapped():
    from deskguard.dataset import web_label

    expected = {
        "Mobile phone": "phone",
        "Electronic-devices-phone-charger": None,
        "Car-key-Toyota": "keys",
        "Keyboard": None,
        "Wallet-brown": "wallet",
        "Sunglasses": "glasses",
        "Water bottle": "drink",
        "metal_can": "drink",
        "Laptop": None,
    }
    assert {name: web_label(name) for name in expected} == expected


def test_merge_web_data(tmp_path):
    from deskguard.dataset import merge_web_data

    export = tmp_path / "raw" / "desk-items"
    names = ["Laptop", "Keys", "Mobile phone"]
    for split in ("train", "valid"):
        (export / split / "images").mkdir(parents=True)
        (export / split / "labels").mkdir(parents=True)
        cv2.imwrite(str(export / split / "images" / "a.jpg"), np.zeros((10, 10, 3), np.uint8))
        cv2.imwrite(str(export / split / "images" / "b.jpg"), np.zeros((10, 10, 3), np.uint8))
        (export / split / "labels" / "a.txt").write_text("1 0.5 0.5 0.2 0.2\n2 0.3 0.3 0.1 0.1")
        (export / split / "labels" / "b.txt").write_text("0 0.5 0.5 0.9 0.9")
    (export / "data.yaml").write_text(yaml.safe_dump({"names": names}))

    class Hands:
        def __call__(self, frame, track=False, confidence=None):
            return [det([0, 0, 5, 5], "hand", 0.9), det([5, 5, 10, 10], "keys", 0.9)]

    config = load_config()
    out = tmp_path / "train"
    data_yaml = merge_web_data(config, str(tmp_path / "raw"), str(out), Hands())
    assert sorted(p.name for p in (out / "images").rglob("*.jpg")) == ["desk-items_a.jpg"] * 2
    assert (out / "labels/val/desk-items_a.txt").read_text().splitlines() == [
        "3 0.5 0.5 0.2 0.2",
        "2 0.3 0.3 0.1 0.1",
        "1 0.250000 0.250000 0.500000 0.500000",
    ]
    assert yaml.safe_load(data_yaml.read_text())["names"] == dict(enumerate(config.labels))


def test_relabel_single_item_session():
    from deskguard.dataset import relabel

    detections = [
        det([0, 0, 50, 30], "phone", 0.8),
        det([1, 0, 50, 30], "keys", 0.4),
        det([60, 60, 90, 90], "hand", 0.7),
    ]
    kept = relabel(detections, "wallet", load_config())
    assert [(d.label, d.confidence) for d in kept] == [("wallet", 0.8), ("hand", 0.7)]
