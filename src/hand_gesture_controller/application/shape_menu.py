"""Mô-đun Trình đơn tạo hình (ShapeMenu) chọn hình học để đưa lên canvas."""

from typing import Any, List, Optional
import cv2
import numpy as np

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
    ) -> None:
        """Cập nhật trạng thái mở menu và tạo mới vật thể."""
        if not hand_landmarks or not hasattr(hand_landmarks, "landmark") or not mgr.visible:
            self.prev = False
            self.is_open = False
            return

        mx, my = mgr.get_thumb_index_midpoint(hand_landmarks, w, h)
        is_select = static_gesture == "Select"

        if is_select and not self.prev:
            # Vùng nút bấm mở menu góc dưới trái
            if 10 <= mx <= 110 and h - 70 <= my <= h - 10:
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
        btn_x, btn_y, btn_w, btn_h = 10, h - 70, 100, 60
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
            menu_w = 4 * 70 + 20
            menu_h = 80
            menu_x = btn_x + btn_w + 10
            menu_y = h - 90

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
                item_x = menu_x + 10 + i * 70
                item_y = menu_y + 15
                shape.x = item_x
                shape.y = item_y
                shape.draw(frame)
