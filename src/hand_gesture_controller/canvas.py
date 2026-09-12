"""Mô-đun quản lý canvas tương tác 2D, các vật thể kéo thả và menu tạo hình."""

import math
import random
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import GestureEvent, HandObservation


class CursorFilter:
    """Bộ lọc thông thấp / Exponential Moving Average cho tọa độ con trỏ."""

    def __init__(self, tau: float = 0.08) -> None:
        self.tau = tau
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.prev_time: Optional[float] = None

    def reset(self) -> None:
        self.prev_x = None
        self.prev_y = None
        self.prev_time = None

    def filter(self, x: float, y: float, timestamp: float) -> Tuple[int, int]:
        if self.prev_x is None or self.prev_y is None or self.prev_time is None:
            self.prev_x, self.prev_y, self.prev_time = x, y, timestamp
            return int(round(x)), int(round(y))

        dt = max(1e-4, timestamp - self.prev_time)
        alpha = 1.0 - math.exp(-dt / max(self.tau, 1e-4))
        self.prev_x += alpha * (x - self.prev_x)
        self.prev_y += alpha * (y - self.prev_y)
        self.prev_time = timestamp
        return int(round(self.prev_x)), int(round(self.prev_y))


class DraggableObject:
    """Lớp cơ sở biểu diễn vật thể hình chữ nhật có thể kéo thả trên màn hình."""

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        name: str = "Object",
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.name = name
        self.is_dragging = False
        self.offset_x = 0
        self.offset_y = 0

    def is_point_inside(self, px: int, py: int) -> bool:
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )

    def start_drag(self, px: int, py: int) -> bool:
        if self.is_point_inside(px, py):
            self.is_dragging = True
            self.offset_x = px - self.x
            self.offset_y = py - self.y
            return True
        return False

    def update_position(
        self,
        px: int,
        py: int,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        if self.is_dragging:
            self.x = px - self.offset_x
            self.y = py - self.offset_y
            if frame_width is not None:
                self.x = max(0, min(self.x, max(0, frame_width - self.width)))
            if frame_height is not None:
                self.y = max(0, min(self.y, max(0, frame_height - self.height)))

    def stop_drag(self) -> None:
        self.is_dragging = False

    def change_color(self) -> None:
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )

    def draw(self, frame: np.ndarray) -> None:
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            self.color,
            -1,
        )
        border_color = (255, 255, 255) if self.is_dragging else (0, 0, 0)
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            border_color,
            2,
        )

    def create_full_size(self, cx: int, cy: int) -> "DraggableObject":
        return DraggableObject(cx, cy, 100, 80, self.color, self.name)


class RectangleObject(DraggableObject):
    pass


class CircleObject(DraggableObject):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 40,
        color: Tuple[int, int, int] = (255, 0, 0),
        name: str = "Circle",
    ) -> None:
        super().__init__(x, y, radius * 2, radius * 2, color, name)
        self.radius = radius

    def is_point_inside(self, px: int, py: int) -> bool:
        cx, cy = self.x + self.radius, self.y + self.radius
        return math.hypot(px - cx, py - cy) <= self.radius

    def draw(self, frame: np.ndarray) -> None:
        cx, cy = self.x + self.radius, self.y + self.radius
        cv2.circle(frame, (cx, cy), self.radius, self.color, -1)
        border_color = (255, 255, 255) if self.is_dragging else (0, 0, 0)
        cv2.circle(frame, (cx, cy), self.radius, border_color, 2)

    def create_full_size(self, cx: int, cy: int) -> "CircleObject":
        return CircleObject(cx, cy, 50, self.color, self.name)


class TriangleObject(DraggableObject):
    def __init__(
        self,
        x: int,
        y: int,
        size: int = 80,
        color: Tuple[int, int, int] = (0, 0, 255),
        name: str = "Triangle",
    ) -> None:
        super().__init__(x, y, size, size, color, name)
        self.size = size

    def _get_pts(self) -> np.ndarray:
        p1 = [self.x + self.size // 2, self.y]
        p2 = [self.x, self.y + self.size]
        p3 = [self.x + self.size, self.y + self.size]
        return np.array([p1, p2, p3], np.int32)

    def is_point_inside(self, px: int, py: int) -> bool:
        pts = self._get_pts()
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        pts = self._get_pts().reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], self.color)
        border_color = (255, 255, 255) if self.is_dragging else (0, 0, 0)
        cv2.polylines(frame, [pts], True, border_color, 2)

    def create_full_size(self, cx: int, cy: int) -> "TriangleObject":
        return TriangleObject(cx, cy, 100, self.color, self.name)


class StarObject(DraggableObject):
    def __init__(
        self,
        x: int,
        y: int,
        size: int = 80,
        color: Tuple[int, int, int] = (0, 255, 255),
        name: str = "Star",
    ) -> None:
        super().__init__(x, y, size, size, color, name)
        self.size = size

    def _get_pts(self) -> np.ndarray:
        cx, cy = self.x + self.size // 2, self.y + self.size // 2
        r_outer = self.size // 2
        r_inner = r_outer // 2
        pts = []
        for i in range(10):
            r = r_outer if i % 2 == 0 else r_inner
            angle = i * math.pi / 5.0 - math.pi / 2.0
            pts.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
        return np.array(pts, np.int32)

    def is_point_inside(self, px: int, py: int) -> bool:
        pts = self._get_pts()
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        pts = self._get_pts().reshape((-1, 1, 2))
        cv2.fillPoly(frame, [pts], self.color)
        border_color = (255, 255, 255) if self.is_dragging else (0, 0, 0)
        cv2.polylines(frame, [pts], True, border_color, 2)

    def create_full_size(self, cx: int, cy: int) -> "StarObject":
        return StarObject(cx, cy, 100, self.color, self.name)


class DraggableObjectManager:
    """Quản lý danh sách các vật thể kéo thả 2D và trạng thái tương tác trên Canvas."""

    def __init__(self, cursor_tau: float = 0.08) -> None:
        self.objects: List[DraggableObject] = []
        self.active_object: Optional[DraggableObject] = None
        self.visible: bool = False
        self.cursor_filter = CursorFilter(tau=cursor_tau)

    def add_object(self, obj: DraggableObject) -> None:
        self.objects.append(obj)

    def toggle_visibility(self) -> None:
        self.visible = not self.visible
        if not self.visible and self.active_object is not None:
            self.active_object.stop_drag()
            self.active_object = None

    def get_cursor_position(
        self,
        observation: HandObservation,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Lấy tọa độ con trỏ (điểm giữa ngón cái và ngón trỏ) đã qua lọc mượt."""
        w = frame_width or observation.frame_width
        h = frame_height or observation.frame_height
        coords = observation.landmarks
        raw_x = (coords[4, 0] + coords[8, 0]) / 2.0 * w
        raw_y = (coords[4, 1] + coords[8, 1]) / 2.0 * h
        return self.cursor_filter.filter(raw_x, raw_y, observation.timestamp)

    def update_event(
        self,
        hand_landmarks: Any,
        event: GestureEvent,
        frame_width: int,
        frame_height: int,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Cập nhật trạng thái vật thể từ GestureEvent."""
        if hand_landmarks is None:
            self.cursor_filter.reset()
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None
            return

        if event == GestureEvent.TOGGLE_CANVAS:
            self.toggle_visibility()
            return

        if not self.visible:
            return

        if cursor_pos is not None:
            mid_x, mid_y = cursor_pos
        elif isinstance(hand_landmarks, np.ndarray):
            raw_x = (hand_landmarks[4, 0] + hand_landmarks[8, 0]) / 2.0 * frame_width
            raw_y = (hand_landmarks[4, 1] + hand_landmarks[8, 1]) / 2.0 * frame_height
            mid_x, mid_y = int(round(raw_x)), int(round(raw_y))
        else:
            return

        if event == GestureEvent.DELETE_OBJECT:
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    self.objects.remove(obj)
                    if self.active_object == obj:
                        self.active_object = None
                    break

        elif event == GestureEvent.CHANGE_COLOR:
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    obj.change_color()
                    break

        elif event == GestureEvent.START_DRAG:
            if self.active_object is None:
                for obj in reversed(self.objects):
                    if obj.start_drag(mid_x, mid_y):
                        self.active_object = obj
                        self.objects.remove(obj)
                        self.objects.append(obj)
                        break

        elif event == GestureEvent.DRAG:
            if self.active_object is not None:
                self.active_object.update_position(
                    mid_x, mid_y, frame_width, frame_height
                )

        elif event == GestureEvent.STOP_DRAG:
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None

    def draw_all(self, frame: np.ndarray) -> None:
        if self.visible:
            for obj in self.objects:
                obj.draw(frame)

    def draw_cursor(
        self,
        frame: np.ndarray,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Vẽ con trỏ điều khiển tròn với viền nổi bật."""
        if cursor_pos is not None:
            mid_x, mid_y = cursor_pos
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 12, (255, 255, 255), 2)


class ShapeMenu:
    """Trình đơn chọn hình để tạo mới vật thể kéo thả trên canvas."""

    def __init__(self) -> None:
        self.is_open: bool = False
        self.shapes: List[DraggableObject] = [
            DraggableObject(0, 0, 60, 50, (100, 200, 100), "Rectangle"),
            CircleObject(0, 0, 30, (100, 100, 200), "Circle"),
            TriangleObject(0, 0, 60, (200, 100, 100), "Triangle"),
            StarObject(0, 0, 60, (200, 200, 100), "Star"),
        ]
        self.prev_select: bool = False

    @staticmethod
    def _menu_button_geometry(frame_height: int) -> Tuple[int, int, int, int]:
        return 10, frame_height - 70, 100, 60

    def _layout_shapes(self, frame_height: int) -> Tuple[int, int, int, int]:
        btn_x, btn_y, btn_w, _ = self._menu_button_geometry(frame_height)
        menu_w = 4 * 70 + 20
        menu_h = 80
        menu_x = btn_x + btn_w + 10
        menu_y = frame_height - 90

        for i, shape in enumerate(self.shapes):
            shape.x = menu_x + 10 + i * 70
            shape.y = menu_y + 15

        return menu_x, menu_y, menu_w, menu_h

    def handle_click(self, px: int, py: int) -> Optional[int]:
        if not self.is_open:
            return None
        for i, shape in enumerate(self.shapes):
            if shape.is_point_inside(px, py):
                self.is_open = False
                return i
        return None

    def update(
        self,
        hand_landmarks: Any,
        static_gesture: str,
        w: int,
        h: int,
        mgr: DraggableObjectManager,
        event: GestureEvent = GestureEvent.NONE,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        if hand_landmarks is None or not mgr.visible:
            self.prev_select = False
            self.is_open = False
            return

        if cursor_pos is not None:
            mx, my = cursor_pos
        elif isinstance(hand_landmarks, HandObservation):
            mx, my = mgr.get_cursor_position(hand_landmarks, w, h)
        elif isinstance(hand_landmarks, np.ndarray) and hand_landmarks.shape == (21, 3):
            mx = int(round((hand_landmarks[4, 0] + hand_landmarks[8, 0]) * 0.5 * w))
            my = int(round((hand_landmarks[4, 1] + hand_landmarks[8, 1]) * 0.5 * h))
        else:
            self.prev_select = False
            return

        if event == GestureEvent.OPEN_MENU:
            self.is_open = not self.is_open
            self.prev_select = False
            return

        if self.is_open:
            self._layout_shapes(h)

        is_select = static_gesture == "Select"
        if is_select and not self.prev_select:
            btn_x, btn_y, btn_w, btn_h = self._menu_button_geometry(h)
            if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                self.is_open = not self.is_open
            else:
                idx = self.handle_click(mx, my)
                if idx is not None:
                    cx, cy = w // 2 - 50, h // 2 - 50
                    new_obj = self.shapes[idx].create_full_size(cx, cy)
                    mgr.add_object(new_obj)

        self.prev_select = is_select

    def draw(self, frame: np.ndarray, h: int) -> None:
        btn_x, btn_y, btn_w, btn_h = self._menu_button_geometry(h)
        cv2.rectangle(
            frame,
            (btn_x, btn_y),
            (btn_x + btn_w, btn_y + btn_h),
            (50, 50, 50),
            -1,
        )
        cv2.rectangle(
            frame,
            (btn_x, btn_y),
            (btn_x + btn_w, btn_y + btn_h),
            (200, 200, 200),
            2,
        )
        cv2.putText(
            frame,
            "MENU",
            (btn_x + 15, btn_y + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        if self.is_open:
            menu_x, menu_y, menu_w, menu_h = self._layout_shapes(h)
            cv2.rectangle(
                frame,
                (menu_x, menu_y),
                (menu_x + menu_w, menu_y + menu_h),
                (30, 30, 30),
                -1,
            )
            cv2.rectangle(
                frame,
                (menu_x, menu_y),
                (menu_x + menu_w, menu_y + menu_h),
                (150, 150, 150),
                2,
            )
            for shape in self.shapes:
                shape.draw(frame)
