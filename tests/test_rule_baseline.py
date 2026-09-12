"""Tests xác thực các luật hình học của RuleStaticBaseline."""

import numpy as np

from hand_gesture_controller.rule_baseline import RuleStaticBaseline
from hand_gesture_controller.schemas import HandObservation


def make_hand(
    fingers_extended=(False, False, False, False, False),
    thumb_index_dist: float = 0.5,
    middle_index_dist: float = 0.5,
) -> HandObservation:
    """Tạo mock HandObservation để kiểm thử hình học."""
    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.5, 0.8, 0.0]  # Wrist
    coords[9] = [0.5, 0.5, 0.0]  # Middle MCP (palm_size = 0.3)

    # Đặt khớp MCP
    mcps = [(1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
    tips = [4, 8, 12, 16, 20]

    for i, (is_ext) in enumerate(fingers_extended):
        tip_idx = tips[i]
        if i == 0:
            # Thumb: nếu duỗi thì xa cổ tay hơn khớp ip (3)
            coords[3] = [0.4, 0.7, 0.0]
            coords[4] = [0.3, 0.6, 0.0] if is_ext else [0.45, 0.75, 0.0]
        else:
            base_idx = mcps[i][0]
            pip_idx = mcps[i][1]
            coords[base_idx] = [0.3 + i * 0.1, 0.6, 0.0]
            if is_ext:
                coords[pip_idx] = [0.3 + i * 0.1, 0.4, 0.0]
                coords[tip_idx] = [0.3 + i * 0.1, 0.2, 0.0]
            else:
                coords[pip_idx] = [0.3 + i * 0.1, 0.65, 0.0]
                coords[tip_idx] = [0.3 + i * 0.1, 0.75, 0.0]

    # Điều chỉnh khoảng cách thumb-index và middle-index theo tham số
    palm_size = 0.3
    if fingers_extended[0] and fingers_extended[1]:
        coords[4] = coords[8] + np.array([thumb_index_dist * palm_size, 0.0, 0.0], dtype=np.float32)
    if fingers_extended[2]:  # Nếu middle duỗi
        coords[12] = coords[8] + np.array([middle_index_dist * palm_size, 0.0, 0.0], dtype=np.float32)

    return HandObservation(
        landmarks=coords,
        handedness="Right",
        handedness_score=1.0,
        timestamp=1.0,
        frame_width=640,
        frame_height=480,
        palm_size=palm_size,
        hand_center=(0.5, 0.5),
    )


def test_fist_rule():
    rule_engine = RuleStaticBaseline()
    obs = make_hand(fingers_extended=(False, False, False, False, False))
    pred = rule_engine.predict_observation(obs)
    assert pred.label == "Fist"
    assert pred.confidence >= 0.90


def test_stop_rule():
    rule_engine = RuleStaticBaseline()
    obs = make_hand(fingers_extended=(True, True, True, True, True), thumb_index_dist=0.8, middle_index_dist=0.5)
    pred = rule_engine.predict_observation(obs)
    assert pred.label == "Stop"
    assert pred.confidence >= 0.90


def test_peace_rule():
    rule_engine = RuleStaticBaseline()
    # Chỉ trỏ và giữa duỗi, cái gập
    obs = make_hand(fingers_extended=(False, True, True, False, False), thumb_index_dist=0.7, middle_index_dist=0.5)
    pred = rule_engine.predict_observation(obs)
    assert pred.label == "Peace"
    assert pred.confidence >= 0.85


def test_select_rule():
    rule_engine = RuleStaticBaseline()
    # Pinch cái - trỏ (thumb_index_dist < 0.60), ngón giữa tách ra (> 0.4), tổng ngón duỗi = 2
    obs = make_hand(fingers_extended=(True, True, False, False, False), thumb_index_dist=0.25, middle_index_dist=0.6)
    pred = rule_engine.predict_observation(obs)
    assert pred.label == "Select"


def test_options_rule():
    rule_engine = RuleStaticBaseline()
    # Cả 3 ngón (cái, trỏ, giữa) đều chụm sát nhau
    obs = make_hand(fingers_extended=(True, True, True, False, False), thumb_index_dist=0.25, middle_index_dist=0.25)
    pred = rule_engine.predict_observation(obs)
    assert pred.label == "Options"
