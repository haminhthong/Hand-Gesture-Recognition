"""Unit tests kiểm tra Máy trạng thái hữu hạn (FSM) cho các cử chỉ động (Wave, SOS Demo Gesture, On/Off).

Bao gồm kiểm tra trình tự hợp lệ, timeout, và reset trạng thái khi mất dấu bàn tay.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hand_gesture_controller.gesture_detector import GestureDetector


def make_pt(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def make_hand(center_x: float = 0.5, fingers_up: int = 5):
    pts = [make_pt(center_x, 0.8) for _ in range(21)]
    pts[0] = make_pt(center_x, 0.9)
    pts[9] = make_pt(center_x, 0.6)

    # 4 ngón chính: index, middle, ring, pinky
    if fingers_up == 0:
        ext = (False, False, False, False)
    elif fingers_up >= 4:
        ext = (True, True, True, True)
    else:
        ext = (fingers_up >= 1, fingers_up >= 2, fingers_up >= 3, False)

    for i, (m, p, t) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))):
        pts[m] = make_pt(center_x, 0.65)
        if ext[i]:
            pts[p] = make_pt(center_x, 0.45)
            pts[t] = make_pt(center_x, 0.20)
        else:
            pts[p] = make_pt(center_x, 0.55)
            pts[t] = make_pt(center_x, 0.70)

    # Thumb: Pinky MCP tại center_x + 0.1, Thumb IP tại center_x - 0.1 (d = 0.2)
    pts[17] = make_pt(center_x + 0.1, 0.65)
    pts[3] = make_pt(center_x - 0.1, 0.65)
    if fingers_up == 5:
        # Ngón cái vươn xa khỏi Pinky MCP -> is_thumb_up = True
        pts[4] = make_pt(center_x - 0.3, 0.65)
    else:
        # Ngón cái gập về phía Pinky MCP -> is_thumb_up = False
        pts[4] = make_pt(center_x + 0.05, 0.65)

    return SimpleNamespace(landmark=pts)



def test_wave_direction_changes():
    """Vẫy tay qua lại đủ số lần quy định (wave_direction_changes=3) sẽ xác nhận cử chỉ Wave."""
    detector = GestureDetector()

    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=10.0):
        # Frame 1: Bắt đầu vẫy tại x=0.5 (open hand 5 fingers)
        detector.detect_motion_gesture(make_hand(0.5, 5))
        assert detector.gestureTracking == "Start Wave"

    # Frame 2: Di chuyển sang phải -> Move Right
    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=10.1):
        detector.detect_motion_gesture(make_hand(0.65, 5))

    # Frame 3: Đổi hướng sang trái -> Move Left (1st change)
    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=10.2):
        detector.detect_motion_gesture(make_hand(0.40, 5))

    # Frame 4: Đổi hướng sang phải -> Move Right (2nd change)
    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=10.3):
        detector.detect_motion_gesture(make_hand(0.65, 5))

    # Frame 5: Đổi hướng sang trái -> Move Left (3rd change -> Wave confirmed)
    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=10.4):
        (gesture, _), _ = detector.detect_motion_gesture(make_hand(0.35, 5))
        assert gesture == "Wave"


def test_sos_valid_sequence():
    """SOS Demo Gesture: Xòe 4 ngón (ngón cái gập) rồi nắm đấm (0 ngón) trong khoảng thời gian quy định."""
    detector = GestureDetector()

    # Step 1: 4 ngón duỗi, ngón cái gập -> "Start SOS"
    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=100.0):
        # Tạo tay 4 ngón: ngón cái gập
        hand_4_fingers = make_hand(0.5, 4)
        hand_4_fingers.landmark[4] = make_pt(0.5, 0.7)  # Thumb folded
        detector.detect_motion_gesture(hand_4_fingers)
        assert detector.gestureTracking == "Start SOS"

    # Step 2: Nắm đấm trong thời gian cho phép (< 1.5s timeout) -> Xác nhận SOS
    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=100.8):
        hand_fist = make_hand(0.5, 0)
        (gesture, _), tracking = detector.detect_motion_gesture(hand_fist)
        assert gesture == "SOS"
        assert tracking == ""


def test_sos_timeout_rejects_sequence():
    """Nếu nắm đấm quá muộn (> 1.5s), chuỗi cử chỉ SOS bị hủy bỏ."""
    detector = GestureDetector()

    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=50.0):
        hand_4 = make_hand(0.5, 4)
        hand_4.landmark[4] = make_pt(0.5, 0.7)
        detector.detect_motion_gesture(hand_4)
        assert detector.gestureTracking == "Start SOS"

    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=52.0):  # +2.0s > 1.5s
        hand_fist = make_hand(0.5, 0)
        (gesture, _), tracking = detector.detect_motion_gesture(hand_fist)
        assert gesture == "Still"
        assert tracking == ""


def test_missing_hand_resets_fsm():
    """Kiểm tra reset() xóa toàn bộ trạng thái đang theo dõi của FSM."""
    detector = GestureDetector()

    with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=1.0):
        detector.detect_motion_gesture(make_hand(0.5, 5))
        assert detector.gestureTracking == "Start Wave"

    detector.reset()
    assert detector.gestureTracking == ""
    assert detector.wave_count == 0
    assert detector.prev_center is None
