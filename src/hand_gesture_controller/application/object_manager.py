"""Mô-đun quản lý các vật thể kéo thả 2D (DraggableObjectManager) và tương tác canvas."""

import math
import random
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

from ..schemas import GestureEvent, HandObservation
from ..temporal.cursor_filter import CursorFilter


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

    def __init__(self, smooth_alpha: float = 0.4, cursor_tau: float = 0.08) -> None:
        self.objects: List[DraggableObject] = []
        self.active_object: Optional[DraggableObject] = None
        self.prev_gesture: Optional[str] = None
        self.visible: bool = False
        self.prev_motion_gesture: Optional[str] = None

        self.smooth_alpha: float = smooth_alpha
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None

        self.cursor_filter = CursorFilter(tau=cursor_tau)

    def add_object(self, obj: DraggableObject) -> None:
        self.objects.append(obj)

    def toggle_visibility(self) -> None:
        self.visible = not self.visible
        if not self.visible and self.active_object is not None:
            self.active_object.stop_drag()
            self.active_object = None

    def get_thumb_index_midpoint(
        self,
        hand_landmarks: Any,
        frame_width: int,
        frame_height: int,
        smooth: bool = True,
    ) -> Tuple[int, int]:
        """Tính tọa độ điểm giữa ngón cái và ngón trỏ kèm lọc mượt."""
        lm = hand_landmarks.landmark
        thumb_tip = lm[4]
        index_tip = lm[8]

        raw_x = (thumb_tip.x + index_tip.x) / 2.0 * frame_width
        raw_y = (thumb_tip.y + index_tip.y) / 2.0 * frame_height

        if smooth:
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x, self.smooth_y = raw_x, raw_y
            else:
                self.smooth_x = (
                    self.smooth_alpha * raw_x + (1 - self.smooth_alpha) * self.smooth_x
                )
                self.smooth_y = (
                    self.smooth_alpha * raw_y + (1 - self.smooth_alpha) * self.smooth_y
                )
            return int(round(self.smooth_x)), int(round(self.smooth_y))

        return int(round(raw_x)), int(round(raw_y))

    def get_cursor_position(
        self,
        observation: HandObservation,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """Lấy tọa độ con trỏ đã qua lọc mượt CursorFilter từ HandObservation."""
        w = frame_width or observation.frame_width
        h = frame_height or observation.frame_height
        coords = observation.landmarks
        # Điểm giữa ngón cái và trỏ
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
        """Cập nhật trạng thái vật thể bằng GestureEvent (chuẩn kiến trúc HCI)."""
        if hand_landmarks is None:
            self.smooth_x = None
            self.smooth_y = None
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
        elif hasattr(hand_landmarks, "landmark"):
            mid_x, mid_y = self.get_thumb_index_midpoint(
                hand_landmarks, frame_width, frame_height
            )
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

    def update(
        self,
        hand_landmarks: Any,
        static_gesture: str,
        motion_gesture: str,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Phương thức tương thích ngược cập nhật trực tiếp từ nhãn cử chỉ."""
        if hand_landmarks is None or not hasattr(hand_landmarks, "landmark"):
            self.smooth_x = None
            self.smooth_y = None
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None
            return

        if motion_gesture == "On/Off" and self.prev_motion_gesture != "On/Off":
            self.toggle_visibility()

        self.prev_motion_gesture = motion_gesture

        if not self.visible:
            return

        mid_x, mid_y = self.get_thumb_index_midpoint(
            hand_landmarks, frame_width, frame_height
        )

        if static_gesture == "Stop" and self.prev_gesture != "Stop":
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    self.objects.remove(obj)
                    if self.active_object == obj:
                        self.active_object = None
                    break

        if static_gesture == "Options" and self.prev_gesture != "Options":
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    obj.change_color()
                    break

        if static_gesture == "Select":
            if self.active_object is None:
                for obj in reversed(self.objects):
                    if obj.start_drag(mid_x, mid_y):
                        self.active_object = obj
                        self.objects.remove(obj)
                        self.objects.append(obj)
                        break
            else:
                self.active_object.update_position(
                    mid_x, mid_y, frame_width, frame_height
                )
        else:
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None

        self.prev_gesture = static_gesture

    def draw_all(self, frame: np.ndarray) -> None:
        if self.visible:
            for obj in self.objects:
                obj.draw(frame)

    def draw_cursor(
        self,
        frame: np.ndarray,
        hand_landmarks: Any,
        frame_width: int,
        frame_height: int,
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Vẽ con trỏ điều khiển tròn với viền nổi bật."""
        if cursor_pos is not None:
            mid_x, mid_y = cursor_pos
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 12, (255, 255, 255), 2)
        elif hand_landmarks is not None and hasattr(hand_landmarks, "landmark"):
            mid_x, mid_y = self.get_thumb_index_midpoint(
                hand_landmarks, frame_width, frame_height
            )
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 12, (255, 255, 255), 2)
