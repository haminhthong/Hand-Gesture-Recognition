"""Integration test kiểm tra toàn bộ luồng pipeline từ Observation đến Canvas Action."""

import numpy as np

from hand_gesture_controller.canvas import DraggableObject, DraggableObjectManager
from hand_gesture_controller.dynamic_gesture import DynamicGestureFSM
from hand_gesture_controller.event_mapper import GestureEvent, GestureEventMapper
from hand_gesture_controller.rule_baseline import RuleStaticBaseline
from hand_gesture_controller.schemas import HandObservation
from hand_gesture_controller.stabilizer import GestureStabilizer


def make_test_obs(timestamp: float) -> HandObservation:
    """Tạo observation cử chỉ Select chuẩn xác."""
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.5, 0.8, 0.0]  # Wrist
    coords[9] = [0.5, 0.5, 0.0]  # Middle MCP
    palm_size = 0.3

    # Thumb: duỗi
    coords[3] = [0.45, 0.65, 0.0]
    coords[4] = [0.42, 0.38, 0.0]

    # Index: duỗi
    coords[5] = [0.45, 0.55, 0.0]
    coords[6] = [0.45, 0.45, 0.0]
    coords[8] = [0.45, 0.35, 0.0]  # Chụm vào gần ngón cái (khoảng cách 0.03 < 0.60 * palm_size)

    # Middle, Ring, Pinky: gập chặt
    for i, base_x in enumerate([0.50, 0.55, 0.60]):
        base_idx = 9 + i * 4
        coords[base_idx] = [base_x, 0.55, 0.0]
        coords[base_idx + 1] = [base_x, 0.65, 0.0]
        coords[base_idx + 3] = [base_x, 0.75, 0.0]

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


def test_end_to_end_pipeline_integration():
    # 1. Pipeline modules
    rule_engine = RuleStaticBaseline()
    dynamic_fsm = DynamicGestureFSM()
    stabilizer = GestureStabilizer(activation_dwell_ms=50.0, release_dwell_ms=50.0)
    event_mapper = GestureEventMapper()
    canvas = DraggableObjectManager()
    canvas.visible = True

    obj = DraggableObject(x=100, y=100, width=80, height=80)
    canvas.add_object(obj)

    # 2. Simulate 4 frames of Select gesture
    timestamps = [1.0, 1.03, 1.06, 1.09]
    events = []

    for t in timestamps:
        obs = make_test_obs(timestamp=t)
        # Static recognition
        pred = rule_engine.predict_observation(obs)
        # Dynamic gesture
        motion = dynamic_fsm.update(obs)
        # Temporal stabilization
        stable = stabilizer.update(pred, timestamp=t)
        # Event mapping
        ev = event_mapper.map_gesture_to_event(stable.label, motion, timestamp=t)
        events.append(ev)
        # Canvas action (con trỏ tại (120, 120))
        canvas.update_event(obs.landmarks, ev, 640, 480, cursor_pos=(120, 120))

    # Frame 1 & 2: Dwell time accumulating (< 50ms)
    # Frame 3 (60ms): Stabilized to Select -> START_DRAG!
    assert GestureEvent.START_DRAG in events
    # Frame 4 (90ms): DRAG
    assert GestureEvent.DRAG in events
    assert canvas.active_object == obj
