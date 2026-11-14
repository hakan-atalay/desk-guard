from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import yaml

PERSON = "person"
HAND = "hand"
ACTORS = frozenset({PERSON, HAND})


@dataclass(frozen=True)
class Target:
    label: str
    name: str
    prompts: tuple[str, ...]
    valuable: bool = True


@dataclass(frozen=True)
class ModelSettings:
    weights: str = "yoloe-26s-seg.pt"
    confidence: float = 0.25
    actor_confidence: float = 0.4
    image_size: int = 640
    device: str | None = None
    duplicate_iou: float = 0.6
    max_box_ratio: float = 0.25
    actor_overlap_iou: float = 0.5


@dataclass(frozen=True)
class MonitorSettings:
    settle_seconds: float = 2.0
    abandon_seconds: float = 10.0
    removal_seconds: float = 3.0
    forget_seconds: float = 2.0
    removed_display_seconds: float = 5.0
    owner_margin: float = 0.2
    move_ratio: float = 0.03
    match_iou: float = 0.3
    arm_delay_seconds: float = 5.0
    intrusion_reset_seconds: float = 3.0


@dataclass(frozen=True)
class AlarmSettings:
    output_dir: str = "runs/alarms"
    cooldown_seconds: float = 5.0
    banner_seconds: float = 3.0
    beep: bool = True


@dataclass(frozen=True)
class PrivacySettings:
    blur_faces: bool = True
    head_ratio: float = 0.4
    kernel_size: int = 51


@dataclass(frozen=True)
class Config:
    model: ModelSettings = field(default_factory=ModelSettings)
    monitor: MonitorSettings = field(default_factory=MonitorSettings)
    alarms: AlarmSettings = field(default_factory=AlarmSettings)
    privacy: PrivacySettings = field(default_factory=PrivacySettings)
    person_prompts: tuple[str, ...] = (PERSON,)
    hand_prompts: tuple[str, ...] = (HAND,)
    targets: tuple[Target, ...] = ()

    @property
    def labels(self) -> list[str]:
        return [PERSON, HAND, *(t.label for t in self.targets)]

    @property
    def names(self) -> dict[str, str]:
        return {PERSON: "kişi", HAND: "el", **{t.label: t.name for t in self.targets}}

    @property
    def valuable(self) -> dict[str, bool]:
        return {t.label: t.valuable for t in self.targets}

    def vocabulary(self) -> tuple[list[str], list[str]]:
        prompts = [*self.person_prompts, *self.hand_prompts]
        labels = [PERSON] * len(self.person_prompts) + [HAND] * len(self.hand_prompts)
        for target in self.targets:
            prompts.extend(target.prompts)
            labels.extend([target.label] * len(target.prompts))
        return prompts, labels


def parse_config(raw: dict) -> Config:
    targets = tuple(
        Target(
            label=item["label"],
            name=item.get("name", item["label"]),
            prompts=tuple(item.get("prompts") or [item["label"]]),
            valuable=item.get("valuable", True),
        )
        for item in raw.get("targets") or []
    )
    labels = [t.label for t in targets]
    if len(labels) != len(set(labels)) or ACTORS.intersection(labels):
        raise ValueError("Target labels must be unique and must not be 'person' or 'hand'")
    return Config(
        model=ModelSettings(**raw.get("model", {})),
        monitor=MonitorSettings(**raw.get("monitor", {})),
        alarms=AlarmSettings(**raw.get("alarms", {})),
        privacy=PrivacySettings(**raw.get("privacy", {})),
        person_prompts=tuple(raw.get("person_prompts") or [PERSON]),
        hand_prompts=tuple(raw.get("hand_prompts") or [HAND]),
        targets=targets,
    )


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        text = resources.files("deskguard").joinpath("default.yaml").read_text("utf-8")
    else:
        text = Path(path).read_text("utf-8")
    return parse_config(yaml.safe_load(text) or {})
