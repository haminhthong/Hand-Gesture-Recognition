"""Mô-đun Trình đơn tạo hình (ShapeMenu) chọn hình học để đưa lên canvas."""

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ..schemas import GestureEvent, HandObservation
from .object_manager import (
    CircleObject,
    DraggableObject,
    DraggableObjectManager,
    StarObject,
    TriangleObject,
)


class ShapeMenu:
    """Trình đơn chọn hình để tạo mới vật thể kéo thả trên canvas."""

    def __init__(self) -> None:
        """Khởi tạo ShapeMenu."""
        self.is_open: bool = False
        self.shapes: List[DraggableObject] = [
            DraggableObject(0, 0, 60, 50, (100, 200, 100), "Rectangle"),
            CircleObject(0, 0, 30, (100, 100, 200), "Circle"),
            TriangleObject(0, 0, 60, (200, 100, 100), "Triangle"),
            StarObject(0, 0, 60, (200, 200, 100), "Star"),
        ]
        self.prev: bool = False

    @staticmethod
    def _menu_button_geometry(frame_height: int) -> Tuple[int, int, int, int]:
        return 10, frame_height - 70, 100, 60

    def _layout_shapes(self, frame_height: int) -> Tuple[int, int, int, int]:
        """Đặt vị trí hitbox của các hình mẫu trước khi xử lý click hoặc vẽ."""
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
        """Xử lý nhấp chọn mục trong menu."""
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
        """Cập nhật trạng thái mở menu và tạo mới vật thể."""
        if hand_landmarks is None or not mgr.visible:
            self.prev = False
            self.is_open = False
            return

        if cursor_pos is not None:
            mx, my = cursor_pos
        elif isinstance(hand_landmarks, HandObservation):
            mx, my = mgr.get_cursor_position(hand_landmarks, w, h)
        elif hasattr(hand_landmarks, "landmark"):
            mx, my = mgr.get_thumb_index_midpoint(hand_landmarks, w, h)
        elif isinstance(hand_landmarks, np.ndarray) and hand_landmarks.shape == (21, 3):
            mx = int(round((hand_landmarks[4, 0] + hand_landmarks[8, 0]) * 0.5 * w))
            my = int(round((hand_landmarks[4, 1] + hand_landmarks[8, 1]) * 0.5 * h))
        else:
            self.prev = False
            return

        if event == GestureEvent.OPEN_MENU:
            self.is_open = not self.is_open
            self.prev = False
            return

        if self.is_open:
            self._layout_shapes(h)

        is_select = static_gesture == "Select"

        if is_select and not self.prev:
            # Vùng nút bấm mở menu góc dưới trái
            btn_x, btn_y, btn_w, btn_h = self._menu_button_geometry(h)
            if btn_x <= mx <= btn_x + btn_w and btn_y <= my <= btn_y + btn_h:
                self.is_open = not self.is_open
            else:
                idx = self.handle_click(mx, my)
                if idx is not None:
                    cx, cy = w // 2 - 50, h // 2 - 50
                    new_obj = self.shapes[idx].create_full_size(cx, cy)
                    mgr.add_object(new_obj)

        self.prev = is_select

    def draw(self, frame: np.ndarray, h: int) -> None:
        """Vẽ nút menu và danh sách các hình mẫu lên khung hình."""
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

            for i, shape in enumerate(self.shapes):
                shape.draw(frame)
