"""Tests cho GestureEventMapper: Rising edge, Drag sequence, Cooldown và Hand-loss Failsafe."""

from hand_gesture_controller.event_mapper import GestureEvent, GestureEventMapper


def test_drag_sequence():
    mapper = GestureEventMapper()

    # Khung hình 1: Bắt đầu Select -> START_DRAG
    ev1 = mapper.map_gesture_to_event("Select", timestamp=1.0)
    assert ev1 == GestureEvent.START_DRAG
    assert mapper.is_dragging is True

    # Khung hình 2 & 3: Tiếp tục duy trì Select -> DRAG
    ev2 = mapper.map_gesture_to_event("Select", timestamp=1.03)
    assert ev2 == GestureEvent.DRAG

    ev3 = mapper.map_gesture_to_event("Select", timestamp=1.06)
    assert ev3 == GestureEvent.DRAG

    # Khung hình 4: Nhả tay (chuyển sang NoAction hoặc Fist) -> STOP_DRAG
    ev4 = mapper.map_gesture_to_event("NoAction", timestamp=1.10)
    assert ev4 == GestureEvent.STOP_DRAG
    assert mapper.is_dragging is False


def test_hand_loss_failsafe_while_dragging():
    mapper = GestureEventMapper()

    # Đang kéo vật thể
    mapper.map_gesture_to_event("Select", timestamp=1.0)
    assert mapper.is_dragging is True

    # Camera bị mất dấu bàn tay (hand lost) -> Failsafe kích hoạt STOP_DRAG
    fail_event = mapper.handle_hand_loss(timestamp=1.05)
    assert fail_event == GestureEvent.STOP_DRAG
    assert mapper.is_dragging is False


def test_rising_edge_actions_and_cooldown():
    mapper = GestureEventMapper(cooldowns={"change_color": 0.5, "delete_object": 0.5})

    # Frame 1: Options mới xuất hiện -> Kích hoạt CHANGE_COLOR
    ev1 = mapper.map_gesture_to_event("Options", timestamp=10.0)
    assert ev1 == GestureEvent.CHANGE_COLOR

    # Frame 2: Vẫn giữ Options (không phải rising edge) -> NONE (không đổi màu liên tục)
    ev2 = mapper.map_gesture_to_event("Options", timestamp=10.03)
    assert ev2 == GestureEvent.NONE

    # Frame 3: Nhả về NoAction rồi xuất hiện lại Options trong 0.2s (< 0.5s cooldown) -> NONE
    mapper.map_gesture_to_event("NoAction", timestamp=10.10)
    ev3 = mapper.map_gesture_to_event("Options", timestamp=10.20)
    assert ev3 == GestureEvent.NONE

    # Frame 4: Sau 0.6s (> 0.5s cooldown) -> CHANGE_COLOR được phép phát sinh lại!
    mapper.map_gesture_to_event("NoAction", timestamp=10.50)
    ev4 = mapper.map_gesture_to_event("Options", timestamp=10.65)
    assert ev4 == GestureEvent.CHANGE_COLOR


def test_motion_toggle_canvas():
    mapper = GestureEventMapper()

    ev = mapper.map_gesture_to_event("NoAction", motion_label="On/Off", timestamp=5.0)
    assert ev == GestureEvent.TOGGLE_CANVAS

    # Frame tiếp theo vẫn báo On/Off nhưng không lặp
    ev_repeat = mapper.map_gesture_to_event("NoAction", motion_label="On/Off", timestamp=5.03)
    assert ev_repeat == GestureEvent.NONE
