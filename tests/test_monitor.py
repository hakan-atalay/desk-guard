import numpy as np
import pytest

from deskguard.config import MonitorSettings
from deskguard.monitor import Detection, EventKind, ObjectState, SceneMonitor

SETTINGS = MonitorSettings(
    settle_seconds=1.0,
    abandon_seconds=5.0,
    removal_seconds=2.0,
    forget_seconds=1.0,
    removed_display_seconds=1.0,
    arm_delay_seconds=0.0,
    intrusion_reset_seconds=1.0,
)
VALUABLE = {"phone": True, "ashtray": False}
PHONE_BOX = np.array([300, 300, 360, 400], dtype=np.float32)
NEAR = np.array([250, 50, 450, 380], dtype=np.float32)
FAR = np.array([800, 50, 900, 300], dtype=np.float32)


def phone(track_id=1, box=PHONE_BOX):
    return Detection(box, "phone", 0.9, track_id)


def person(box=NEAR):
    return Detection(box, "person", 0.9, 99)


def hand(box):
    return Detection(box, "hand", 0.8, 98)


@pytest.fixture
def monitor():
    return SceneMonitor(SETTINGS, VALUABLE, (1000, 600))


def play(monitor, frames, start=0.0, step=0.5):
    events, t = [], start
    for detections in frames:
        events += monitor.update(detections, t)
        t += step
    return [e.kind for e in events], t


def test_abandoned_after_owner_leaves(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    assert monitor.objects[1].state is ObjectState.STATIONARY
    kinds, _ = play(monitor, [[phone()]] * 12, start=t)
    assert kinds == [EventKind.ABANDONED]


def test_owner_or_hand_nearby_prevents_abandon(monitor):
    hand_box = np.array([310, 320, 380, 420], dtype=np.float32)
    kinds, _ = play(monitor, [[phone(), person()]] * 15 + [[phone(), hand(hand_box)]] * 15)
    assert kinds == []


def test_far_person_is_not_owner(monitor):
    kinds, _ = play(monitor, [[phone(), person(FAR)]] * 14)
    assert EventKind.ABANDONED in kinds


def test_reclaimed_when_owner_returns(monitor):
    _, t = play(monitor, [[phone()]] * 14)
    kinds, _ = play(monitor, [[phone(), person()]], start=t)
    assert kinds == [EventKind.RECLAIMED]


def test_removed_only_when_armed(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    kinds, _ = play(monitor, [[]] * 6, start=t)
    assert kinds == []
    monitor.objects.clear()
    _, t = play(monitor, [[phone(), person()]] * 4, start=t + 3)
    monitor.arm(t)
    kinds, _ = play(monitor, [[]] * 6, start=t)
    assert kinds == [EventKind.REMOVED]


def test_occlusion_is_not_removal_while_disarmed(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    play(monitor, [[person()]] * 10, start=t)
    assert monitor.objects[1].state is ObjectState.STATIONARY


def test_person_over_item_cannot_hide_theft_while_armed(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    monitor.arm(t)
    kinds, _ = play(monitor, [[person()]] * 10, start=t)
    assert kinds == [EventKind.INTRUSION, EventKind.REMOVED]


def test_fixture_never_alarms(monitor):
    ashtray = Detection(np.array([500, 300, 560, 340], dtype=np.float32), "ashtray", 0.9, 5)
    monitor.arm(0.0)
    kinds, t = play(monitor, [[ashtray]] * 20)
    more, _ = play(monitor, [[]] * 10, start=t)
    assert kinds + more == []


def test_track_id_switch_keeps_identity(monitor):
    play(monitor, [[phone(track_id=1)]] * 3)
    play(monitor, [[phone(track_id=7)]] * 3, start=1.5)
    assert len(monitor.objects) == 1


def test_moving_item_becomes_candidate(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    play(monitor, [[phone(box=PHONE_BOX + np.float32([200, 0, 200, 0])), person()]], start=t)
    assert monitor.objects[1].state is ObjectState.CANDIDATE


def test_items_outside_zone_ignored(monitor):
    monitor.set_roi(np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32))
    play(monitor, [[phone()]] * 4)
    assert monitor.objects == {}


def test_intrusion_once_per_entry(monitor):
    kinds, t = play(monitor, [[person()]] * 3)
    assert kinds == []
    monitor.arm(t)
    kinds, t = play(monitor, [[person()]] * 10 + [[]] + [[person()]] * 3, start=t)
    assert kinds == [EventKind.INTRUSION]
    _, t = play(monitor, [[]] * 4, start=t)
    kinds, _ = play(monitor, [[person()]], start=t)
    assert kinds == [EventKind.INTRUSION]


def test_exit_delay():
    monitor = SceneMonitor(MonitorSettings(arm_delay_seconds=5.0), VALUABLE, (1000, 600))
    monitor.arm(0.0)
    assert monitor.arming_remaining(1.0) == 4.0
    kinds, _ = play(monitor, [[person()]] * 8)
    assert kinds == []
    kinds, _ = play(monitor, [[]] * 8 + [[person()]], start=4.0)
    assert kinds == [EventKind.INTRUSION]


def test_item_picked_up_while_armed_raises_removed(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    monitor.arm(t)
    in_hand = PHONE_BOX + np.float32([0, -150, 0, -150])
    kinds, t = play(monitor, [[phone(), person()], [phone(box=in_hand), person()]], start=t)
    assert kinds == [EventKind.INTRUSION, EventKind.REMOVED]
    kinds, _ = play(monitor, [[person()]] * 2 + [[]] * 10, start=t)
    assert EventKind.REMOVED not in kinds


def test_item_moved_while_disarmed_is_not_removed(monitor):
    _, t = play(monitor, [[phone(), person()]] * 4)
    moved = PHONE_BOX + np.float32([0, -150, 0, -150])
    kinds, _ = play(monitor, [[phone(box=moved), person()]], start=t)
    assert kinds == []
    assert monitor.objects[1].state is ObjectState.CANDIDATE
