"""Unit tests xác thực tính chuẩn xác của các khớp MCP và bảng ánh xạ FINGER_LANDMARKS.

Đảm bảo không bao giờ nhầm lẫn khớp MCP (gốc ngón) với DIP (áp đầu ngón).
"""

from types import SimpleNamespace

from hand_gesture_controller.gesture_detector import GestureDetector
from hand_gesture_controller.gesture_features import FINGER_LANDMARKS


def make_pt(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def test_finger_landmarks_constants():
    """Kiểm tra mapping FINGER_LANDMARKS khớp chính xác tiêu chuẩn MediaPipe Hands."""
    assert FINGER_LANDMARKS["thumb"] == {"tip": 4, "ip": 3, "mcp": 2, "cmc": 1}
    assert FINGER_LANDMARKS["index"] == {"tip": 8, "pip": 6, "mcp": 5}
    assert FINGER_LANDMARKS["middle"] == {"tip": 12, "pip": 10, "mcp": 9}
    assert FINGER_LANDMARKS["ring"] == {"tip": 16, "pip": 14, "mcp": 13}
    assert FINGER_LANDMARKS["pinky"] == {"tip": 20, "pip": 18, "mcp": 17}


def test_index_finger_uses_mcp_5():
    """Xác nhận ngón trỏ sử dụng MCP ID = 5 (không phải DIP ID = 7)."""
    detector = GestureDetector()
    # Tạo bàn tay với ngón trỏ duỗi thẳng qua MCP=5, PIP=6, TIP=8
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)   # Wrist
    pts[5] = make_pt(0.5, 0.7)   # Index MCP (ID 5)
    pts[6] = make_pt(0.5, 0.5)   # Index PIP (ID 6)
    pts[7] = make_pt(0.8, 0.8)   # Index DIP (ID 7) - đặt lệch vị trí để nếu nhầm sang DIP sẽ sai góc
    pts[8] = make_pt(0.5, 0.2)   # Index TIP (ID 8)

    # Nếu is_finger_up dùng MCP=5, ngón trỏ thẳng 180 độ -> True
    # Nếu dùng DIP=7, góc TIP(8)->PIP(6)->DIP(7) sẽ gãy và trả về False
    assert detector.is_finger_up(pts, tip_id=8, joint_id=6) is True
    assert detector.is_finger_up(pts, tip_id=8, joint_id=6, mcp_id=5) is True


def test_middle_finger_uses_mcp_9():
    """Xác nhận ngón giữa sử dụng MCP ID = 9 (không phải DIP ID = 11)."""
    detector = GestureDetector()
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)
    pts[9] = make_pt(0.5, 0.7)   # Middle MCP (ID 9)
    pts[10] = make_pt(0.5, 0.5)  # Middle PIP (ID 10)
    pts[11] = make_pt(0.8, 0.8)  # Middle DIP (ID 11) - đặt lệch
    pts[12] = make_pt(0.5, 0.2)  # Middle TIP (ID 12)

    assert detector.is_finger_up(pts, tip_id=12, joint_id=10) is True
    assert detector.is_finger_up(pts, tip_id=12, joint_id=10, mcp_id=9) is True


def test_ring_finger_uses_mcp_13():
    """Xác nhận ngón áp út sử dụng MCP ID = 13 (không phải DIP ID = 15)."""
    detector = GestureDetector()
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)
    pts[13] = make_pt(0.5, 0.7)  # Ring MCP (ID 13)
    pts[14] = make_pt(0.5, 0.5)  # Ring PIP (ID 14)
    pts[15] = make_pt(0.8, 0.8)  # Ring DIP (ID 15) - đặt lệch
    pts[16] = make_pt(0.5, 0.2)  # Ring TIP (ID 16)

    assert detector.is_finger_up(pts, tip_id=16, joint_id=14) is True
    assert detector.is_finger_up(pts, tip_id=16, joint_id=14, mcp_id=13) is True


def test_pinky_finger_uses_mcp_17():
    """Xác nhận ngón út sử dụng MCP ID = 17 (không phải DIP ID = 19)."""
    detector = GestureDetector()
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)
    pts[17] = make_pt(0.5, 0.7)  # Pinky MCP (ID 17)
    pts[18] = make_pt(0.5, 0.5)  # Pinky PIP (ID 18)
    pts[19] = make_pt(0.8, 0.8)  # Pinky DIP (ID 19) - đặt lệch
    pts[20] = make_pt(0.5, 0.2)  # Pinky TIP (ID 20)

    assert detector.is_finger_up(pts, tip_id=20, joint_id=18) is True
    assert detector.is_finger_up(pts, tip_id=20, joint_id=18, mcp_id=17) is True


def test_fist_geometry():
    """Xác nhận tư thế Fist (nắm đấm) có 0 ngón duỗi."""
    detector = GestureDetector()
    # Tất cả ngón gập: TIP ở gần cổ tay hơn PIP
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)  # Wrist
    for tip, pip, mcp in ((8, 6, 5), (12, 10, 9), (16, 14, 13), (20, 18, 17)):
        pts[mcp] = make_pt(0.5, 0.6)
        pts[pip] = make_pt(0.5, 0.5)
        pts[tip] = make_pt(0.5, 0.7)  # TIP gập xuống dưới PIP

    # Thumb gập sát Pinky MCP (17)
    pts[17] = make_pt(0.6, 0.6)
    pts[3] = make_pt(0.40, 0.62)
    pts[4] = make_pt(0.58, 0.62)

    states = detector._finger_states(pts)
    assert states == (False, False, False, False, False)
    res = detector.detect_static_gesture_result(SimpleNamespace(landmark=pts))
    assert res.label == "Fist"


def test_stop_geometry():
    """Xác nhận tư thế Stop (xòe cả 5 ngón)."""
    detector = GestureDetector()
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)  # Wrist
    for tip, pip, mcp, x in (
        (8, 6, 5, 0.4),
        (12, 10, 9, 0.5),
        (16, 14, 13, 0.6),
        (20, 18, 17, 0.7),
    ):
        pts[mcp] = make_pt(x, 0.65)
        pts[pip] = make_pt(x, 0.45)
        pts[tip] = make_pt(x, 0.20)

    # Thumb extended
    pts[17] = make_pt(0.7, 0.65)
    pts[3] = make_pt(0.3, 0.6)
    pts[4] = make_pt(0.2, 0.5)

    states = detector._finger_states(pts)
    assert states == (True, True, True, True, True)
    res = detector.detect_static_gesture_result(SimpleNamespace(landmark=pts))
    assert res.label == "Stop"


def test_peace_geometry():
    """Xác nhận tư thế Peace (chỉ trỏ và giữa duỗi)."""
    detector = GestureDetector()
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)
    # Index & Middle extended
    for tip, pip, mcp, x in ((8, 6, 5, 0.45), (12, 10, 9, 0.55)):
        pts[mcp] = make_pt(x, 0.65)
        pts[pip] = make_pt(x, 0.45)
        pts[tip] = make_pt(x, 0.20)

    # Ring & Pinky folded
    for tip, pip, mcp, x in ((16, 14, 13, 0.65), (20, 18, 17, 0.75)):
        pts[mcp] = make_pt(x, 0.65)
        pts[pip] = make_pt(x, 0.55)
        pts[tip] = make_pt(x, 0.70)

    # Thumb folded (gập sát Pinky MCP 17)
    pts[17] = make_pt(0.75, 0.65)
    pts[3] = make_pt(0.45, 0.65)
    pts[4] = make_pt(0.70, 0.65)

    states = detector._finger_states(pts)
    assert states == (False, True, True, False, False)
    res = detector.detect_static_gesture_result(SimpleNamespace(landmark=pts))
    assert res.label == "Peace"

