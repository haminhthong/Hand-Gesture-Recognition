"""Unit tests kiểm tra độ ưu tiên (Precedence Table) và giải quyết tính nhập nhằng (Pairwise Ambiguity).

Các cặp cử chỉ cần kiểm tra tính phi nhập nhằng:
- Select vs Options (chụm 2 ngón vs chụm 3 ngón)
- Select vs OK (chụm 2 ngón xòe 0 ngón còn lại vs xòe >= 3 ngón còn lại)
- Stop vs Wave start
- Fist vs Stop
"""

from types import SimpleNamespace

from hand_gesture_controller.gesture_detector import GestureDetector


def make_pt(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def make_hand(
    wrist_y: float = 0.9,
    middle_mcp_y: float = 0.5,
    thumb_tip_pos=(0.4, 0.4),
    index_tip_pos=(0.4, 0.4),
    middle_tip_pos=(0.6, 0.2),
    fingers_extended=(True, True, False, False, False),
):
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, wrist_y)       # Wrist
    pts[9] = make_pt(0.5, middle_mcp_y)  # Middle MCP (palm_size = 0.4)

    pts[4] = make_pt(thumb_tip_pos[0], thumb_tip_pos[1])
    pts[8] = make_pt(index_tip_pos[0], index_tip_pos[1])
    pts[12] = make_pt(middle_tip_pos[0], middle_tip_pos[1])

    # Thiết lập các ngón duỗi hay gập
    finger_tips_pips_mcps = [
        (8, 6, 5, 0.45),
        (12, 10, 9, 0.55),
        (16, 14, 13, 0.65),
        (20, 18, 17, 0.75),
    ]

    for i, (tip, pip, mcp, x) in enumerate(finger_tips_pips_mcps):
        is_ext = fingers_extended[i + 1]  # Bỏ qua ngón cái
        pts[mcp] = make_pt(x, 0.65)
        if is_ext:
            pts[pip] = make_pt(x, 0.45)
            # Giữ nguyên tip đã set nếu là index/middle
            if tip not in (8, 12):
                pts[tip] = make_pt(x, 0.20)
        else:
            pts[pip] = make_pt(x, 0.55)
            pts[tip] = make_pt(x, 0.70)

    # Ngón cái
    thumb_ext = fingers_extended[0]
    pts[17] = make_pt(0.75, 0.65)
    pts[3] = make_pt(0.45, 0.60)
    if not thumb_ext:
        pts[4] = make_pt(pts[3].x, pts[3].y + 0.05)  # Gập gần cổ tay hơn

    return SimpleNamespace(landmark=pts)


def test_select_vs_options_ambiguity():
    """Khi ngón cái và trỏ chụm lại:
    - Nếu ngón giữa cũng chụm gần trỏ -> Options
    - Nếu ngón giữa cách xa trỏ -> Select
    """
    detector = GestureDetector()

    # Case 1: Chụm 3 ngón (Thumb, Index, Middle đều ở (0.5, 0.4))
    hand_options = SimpleNamespace(landmark=[make_pt(0.5, 0.8)] * 21)
    hand_options.landmark[0] = make_pt(0.5, 0.9)
    hand_options.landmark[9] = make_pt(0.5, 0.5)  # Palm size = 0.4
    hand_options.landmark[4] = make_pt(0.5, 0.4)  # Thumb
    hand_options.landmark[8] = make_pt(0.51, 0.4) # Index (d=0.01 / 0.4 = 0.025 < 0.60)
    hand_options.landmark[12] = make_pt(0.52, 0.4) # Middle (d=0.01 / 0.4 = 0.025 < 0.35)

    res_opt = detector.detect_static_gesture_result(hand_options)
    assert res_opt.label == "Options"
    assert res_opt.rule_score > 0.80

    # Case 2: Chụm 2 ngón (Thumb & Index chụm, Middle ở xa (0.5, 0.1))
    # fingers_up = 2 (index, middle, thumb gập chụm về phía trỏ)
    hand_select = SimpleNamespace(landmark=[make_pt(0.5, 0.8)] * 21)
    hand_select.landmark[0] = make_pt(0.5, 0.9)
    hand_select.landmark[9] = make_pt(0.5, 0.5)  # Palm size = 0.4
    hand_select.landmark[17] = make_pt(0.7, 0.6) # Pinky MCP
    hand_select.landmark[3] = make_pt(0.3, 0.5)  # Thumb IP (xa 17: dist = 0.41)
    hand_select.landmark[4] = make_pt(0.5, 0.4)  # Thumb tip (gần 17 hơn 3: dist = 0.28 -> is_thumb_up=False)
    hand_select.landmark[8] = make_pt(0.51, 0.4) # Index tip (pinch với thumb)
    hand_select.landmark[12] = make_pt(0.5, 0.1) # Middle xa trỏ (d = 0.30 / 0.4 = 0.75 > 0.4)

    # Index: MCP 5, PIP 6
    hand_select.landmark[5] = make_pt(0.51, 0.6)
    hand_select.landmark[6] = make_pt(0.51, 0.5)

    # Middle: MCP 9 (set above), PIP 10
    hand_select.landmark[10] = make_pt(0.5, 0.3)

    # Ring & pinky folded
    for m, p, t in ((13, 14, 16), (17, 18, 20)):
        hand_select.landmark[m] = make_pt(0.6, 0.6)
        hand_select.landmark[p] = make_pt(0.6, 0.55)
        hand_select.landmark[t] = make_pt(0.6, 0.7)

    res_sel = detector.detect_static_gesture_result(hand_select)
    assert res_sel.label == "Select"
    assert res_sel.rule_score > 0.80


def test_fist_highest_precedence():
    """Khi tất cả các ngón đều gập (fingers_up == 0), Fist luôn thắng mọi điều kiện khác."""
    detector = GestureDetector()
    # Nắm đấm: ngón cái vô tình ở gần ngón trỏ nhưng tất cả ngón đều gập
    pts = [make_pt(0.5, 0.8) for _ in range(21)]
    pts[0] = make_pt(0.5, 0.9)
    pts[9] = make_pt(0.5, 0.5)

    # All fingers folded
    for m, p, t in ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)):
        pts[m] = make_pt(0.5, 0.6)
        pts[p] = make_pt(0.5, 0.5)
        pts[t] = make_pt(0.5, 0.7)

    # Thumb folded sát 17
    pts[17] = make_pt(0.6, 0.6)
    pts[3] = make_pt(0.4, 0.6)
    pts[4] = make_pt(0.58, 0.6)

    res = detector.detect_static_gesture_result(SimpleNamespace(landmark=pts))
    assert res.label == "Fist"

