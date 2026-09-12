"""Mô-đun tiền xử lý chuẩn hóa 21 điểm mốc MediaPipe thành vector đặc trưng 63D."""

from typing import Any, Optional

import numpy as np

from .schemas import HandObservation


class LandmarkPreprocessor:
    """Bộ tiền xử lý chuẩn hóa 21 điểm mốc 3D MediaPipe thành vector đặc trưng 63 chiều.

    Quy trình xử lý hình học:
    1. Dịch chuyển cổ tay (index 0) về gốc tọa độ (0, 0, 0).
    2. Chuẩn hóa tỉ lệ theo khoảng cách từ cổ tay tới khớp giữa ngón giữa (index 9).
    3. Phản chiếu trục X cho bàn tay trái để đồng nhất về hệ tọa độ bàn tay phải.
    4. Tùy chọn chuẩn hóa góc xoay trong mặt phẳng sao cho trục bàn tay hướng lên trên (-Y).
    5. Trải phẳng thành vector đặc trưng 63 chiều float32.
    """

    def __init__(
        self,
        mirror_left_hand: bool = True,
        normalize_rotation: bool = False,
    ) -> None:
        self.mirror_left_hand = mirror_left_hand
        self.normalize_rotation = normalize_rotation

    def transform_observation(self, observation: HandObservation) -> np.ndarray:
        """Chuẩn hóa trực tiếp từ đối tượng HandObservation sang vector 63D."""
        return self.transform(observation.landmarks, handedness=observation.handedness)

    def transform_landmarks_object(
        self,
        landmarks: Any,
        handedness: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Chuẩn hóa từ đối tượng MediaPipe landmarks hoặc mock landmarks object."""
        if not landmarks or not hasattr(landmarks, "landmark"):
            return None
        points = landmarks.landmark
        if len(points) < 21:
            return None

        coords = np.array([[lm.x, lm.y, lm.z] for lm in points[:21]], dtype=np.float32)
        return self.transform(coords, handedness=handedness)

    def transform(
        self,
        coords: np.ndarray,
        handedness: Optional[str] = None,
    ) -> np.ndarray:
        """Thực hiện tiền xử lý chuẩn hóa mảng tọa độ (21, 3) thành vector 63D."""
        if coords.shape != (21, 3):
            raise ValueError(f"Kích thước coords phải là (21, 3), nhận được {coords.shape}")

        coords = coords.copy()

        # Bước 1: Dịch cổ tay về gốc tọa độ
        wrist = coords[0].copy()
        coords = coords - wrist

        # Bước 2: Chuẩn hóa tỉ lệ theo palm_size (khoảng cách wrist 0 -> middle_mcp 9)
        palm_size = float(np.linalg.norm(coords[9]))
        if palm_size > 1e-6:
            coords = coords / palm_size

        # Bước 3: Lật trục X nếu là bàn tay trái
        if self.mirror_left_hand and handedness == "Left":
            coords[:, 0] = -coords[:, 0]

        # Bước 4: Chuẩn hóa góc xoay trong mặt phẳng nếu bật
        if self.normalize_rotation:
            v_x = coords[9, 0]
            v_y = coords[9, 1]
            norm_v = np.hypot(v_x, v_y)
            if norm_v > 1e-6:
                current_angle = np.arctan2(v_y, v_x)
                rot_angle = (-np.pi / 2.0) - current_angle
                cos_a = np.cos(rot_angle)
                sin_a = np.sin(rot_angle)
                x_rot = cos_a * coords[:, 0] - sin_a * coords[:, 1]
                y_rot = sin_a * coords[:, 0] + cos_a * coords[:, 1]
                coords[:, 0] = x_rot
                coords[:, 1] = y_rot

        # Bước 5: Trải phẳng thành vector 63D
        return coords.reshape(-1).astype(np.float32)
