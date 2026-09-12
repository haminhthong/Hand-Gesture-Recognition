"""Tests cho DynamicGestureFSM (chuỗi cử chỉ động On/Off)."""

import numpy as np

from hand_gesture_controller.dynamic_gesture import DynamicGestureFSM
from hand_gesture_controller.schemas import HandObservation


def create_on_off_obs(stage: str, timestamp: float) -> HandObservation:
    """Tạo observation giả lập cho từng giai đoạn của On/Off."""
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.5, 0.8, 0.0]   # Wrist
    coords[9] = [0.5, 0.5, 0.0]   # Middle MCP (palm_size = 0.3)
    palm_size = 0.3

    # Thumb: tip tại (0.45, 0.38), ip tại (0.43, 0.55) -> thumb extended
    coords[3] = [0.43, 0.55, 0.0]
    coords[4] = [0.45, 0.38, 0.0]

    # Index: duỗi thẳng lên trên (mcp=0.55, pip=0.45, tip=0.15)
    coords[5] = [0.45, 0.55, 0.0]
    coords[6] = [0.45, 0.45, 0.0]
    coords[8] = [0.45, 0.15, 0.0]

    # Ring & Pinky: gập chặt về phía lòng bàn tay (mcp=0.55, pip=0.60, tip=0.70)
    coords[13] = [0.55, 0.55, 0.0]
    coords[14] = [0.55, 0.60, 0.0]
    coords[16] = [0.55, 0.70, 0.0]

    coords[17] = [0.60, 0.55, 0.0]
    coords[18] = [0.60, 0.60, 0.0]
    coords[20] = [0.60, 0.70, 0.0]

    if stage == "start":
        # Middle duỗi và chụm vào ngón cái: pip tại (0.50, 0.45), tip tại (0.46, 0.38)
        # d_tip (0.42) > d_pip (0.35) -> middle_up = True
        # d12_4 = 0.01 / 0.3 = 0.033 < 0.35
        coords[10] = [0.50, 0.45, 0.0]
        coords[12] = [0.46, 0.38, 0.0]
    elif stage == "end":
        # Middle gập lại: pip tại (0.50, 0.60), tip tại (0.50, 0.70)
        # d_tip (0.10) < d_pip (0.20) -> middle_up = False
        # d12_4 = hypot(0.05, 0.32) / 0.3 = 1.08 >= 0.35
        coords[10] = [0.50, 0.60, 0.0]
        coords[12] = [0.50, 0.70, 0.0]

    return HandObservation(
        landmarks=coords,
        handedness="Right",
        handedness_score=1.0,
        timestamp=timestamp,
        frame_width=640,
        frame_height=480,
        palm_size=palm_size,
        hand_center=(0.5, 0.5),
    )


def test_on_off_valid_sequence():
    fsm = DynamicGestureFSM(timeout_seconds=1.5)

    # Frame 1: Tư thế bắt đầu tại t=10.0s
    obs_start = create_on_off_obs("start", timestamp=10.0)
    res1 = fsm.update(obs_start)
    assert res1 == "Still"
    assert fsm.current_state == "START_ONOFF"

    # Frame 2: Hoàn tất chuyển đổi tại t=10.4s (< 1.5s)
    obs_end = create_on_off_obs("end", timestamp=10.4)
    res2 = fsm.update(obs_end)
    assert res2 == "On/Off"
    assert fsm.current_state == "IDLE"


def test_on_off_timeout_resets():
    fsm = DynamicGestureFSM(timeout_seconds=1.5)

    # Bắt đầu tại t=10.0s
    obs_start = create_on_off_obs("start", timestamp=10.0)
    fsm.update(obs_start)
    assert fsm.current_state == "START_ONOFF"

    # Chuyển đổi quá trễ tại t=11.8s (+1.8s > 1.5s timeout)
    obs_end = create_on_off_obs("end", timestamp=11.8)
    res = fsm.update(obs_end)
    assert res == "Still"
    assert fsm.current_state == "IDLE"


def test_missing_hand_resets_fsm():
    fsm = DynamicGestureFSM()

    obs_start = create_on_off_obs("start", timestamp=1.0)
    fsm.update(obs_start)
    assert fsm.current_state == "START_ONOFF"

    # Mất dấu bàn tay (observation = None)
    res = fsm.update(None)
    assert res == "Still"
    assert fsm.current_state == "IDLE"
