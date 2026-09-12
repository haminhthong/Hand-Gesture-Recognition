"""Tests cho canvas: DraggableObject, DraggableObjectManager, ShapeMenu và CursorFilter."""

import numpy as np

from hand_gesture_controller.canvas import (
    CircleObject,
    CursorFilter,
    DraggableObject,
    DraggableObjectManager,
    ShapeMenu,
    StarObject,
    TriangleObject,
)
from hand_gesture_controller.schemas import GestureEvent


def test_cursor_filter_smoothing():
    cursor_filter = CursorFilter(tau=0.08)

    # Điểm đầu tiên
    x0, y0 = cursor_filter.filter(100.0, 200.0, timestamp=1.0)
    assert (x0, y0) == (100, 200)

    # Bước nhảy đột ngột tới (200, 300) sau 0.02s: bộ lọc EMA làm mượt
    x1, y1 = cursor_filter.filter(200.0, 300.0, timestamp=1.02)
    assert 100 < x1 < 200
    assert 200 < y1 < 300


def test_draggable_object_collision_and_drag():
    obj = DraggableObject(x=100, y=100, width=50, height=50)

    # Va chạm
    assert obj.is_point_inside(120, 120) is True
    assert obj.is_point_inside(80, 80) is False

    # Bắt đầu kéo thả
    assert obj.start_drag(110, 115) is True
    assert obj.is_dragging is True

    # Cập nhật vị trí con trỏ mới tới (150, 160)
    obj.update_position(150, 160, frame_width=640, frame_height=480)
    # x = px - offset_x = 150 - (110 - 100) = 140
    # y = py - offset_y = 160 - (115 - 100) = 145
    assert obj.x == 140
    assert obj.y == 145

    obj.stop_drag()
    assert obj.is_dragging is False


def test_shape_objects_collision():
    circle = CircleObject(x=100, y=100, radius=30)
    # Tâm tại (130, 130), bán kính 30
    assert circle.is_point_inside(130, 130) is True
    assert circle.is_point_inside(170, 170) is False

    triangle = TriangleObject(x=100, y=100, size=60)
    assert triangle.is_point_inside(130, 140) is True

    star = StarObject(x=100, y=100, size=60)
    assert star.is_point_inside(130, 130) is True


def test_manager_events_and_actions():
    mgr = DraggableObjectManager()
    mgr.visible = True

    obj = DraggableObject(x=50, y=50, width=40, height=40, color=(0, 255, 0))
    mgr.add_object(obj)

    # 1. START_DRAG tại vị trí (60, 60)
    mgr.update_event(
        hand_landmarks=np.zeros((21, 3)),
        event=GestureEvent.START_DRAG,
        frame_width=640,
        frame_height=480,
        cursor_pos=(60, 60),
    )
    assert mgr.active_object == obj

    # 2. DRAG tới vị trí (120, 120)
    mgr.update_event(
        hand_landmarks=np.zeros((21, 3)),
        event=GestureEvent.DRAG,
        frame_width=640,
        frame_height=480,
        cursor_pos=(120, 120),
    )
    assert obj.x == 110
    assert obj.y == 110

    # 3. STOP_DRAG
    mgr.update_event(
        hand_landmarks=np.zeros((21, 3)),
        event=GestureEvent.STOP_DRAG,
        frame_width=640,
        frame_height=480,
        cursor_pos=(120, 120),
    )
    assert mgr.active_object is None

    # 4. CHANGE_COLOR
    old_color = obj.color
    mgr.update_event(
        hand_landmarks=np.zeros((21, 3)),
        event=GestureEvent.CHANGE_COLOR,
        frame_width=640,
        frame_height=480,
        cursor_pos=(120, 120),
    )
    assert obj.color != old_color

    # 5. DELETE_OBJECT
    mgr.update_event(
        hand_landmarks=np.zeros((21, 3)),
        event=GestureEvent.DELETE_OBJECT,
        frame_width=640,
        frame_height=480,
        cursor_pos=(120, 120),
    )
    assert obj not in mgr.objects


def test_shape_menu():
    menu = ShapeMenu()
    mgr = DraggableObjectManager()
    mgr.visible = True

    # Kích hoạt mở menu bằng OPEN_MENU
    menu.update(
        hand_landmarks=np.zeros((21, 3)),
        static_gesture="Peace",
        w=640,
        h=480,
        mgr=mgr,
        event=GestureEvent.OPEN_MENU,
        cursor_pos=(100, 100),
    )
    assert menu.is_open is True
