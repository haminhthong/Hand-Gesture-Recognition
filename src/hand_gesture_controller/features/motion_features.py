"""Mô-đun trích xuất đặc trưng chuyển động (Motion Features) độc lập tốc độ khung hình (FPS-independent)."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..schemas import HandObservation


@dataclass(frozen=True)
class MotionObservation:
    """Chứa các đặc trưng chuyển động tính toán từ hai khung hình liên tiếp."""

    dt: float  # Khoảng thời gian giữa hai khung hình (giây)
    velocity_normalized: float  # Vận tốc chuẩn hóa theo palm_size trên giây (1/s)
    dx_normalized: float  # Độ dịch chuyển theo trục X trên giây (1/s)
    dy_normalized: float  # Độ dịch chuyển theo trục Y trên giây (1/s)
    direction: str  # Hướng di chuyển chính: "Move Left", "Move Right", "Move Up", "Move Down", "Still"
    current_center: Tuple[float, float]  # Tọa độ trọng tâm hiện tại (x, y)


class MotionFeatures:
    """Bộ trích xuất đặc trưng chuyển động theo thời gian thực dựa trên tâm bàn tay và palm size."""

    def __init__(self, still_velocity_threshold: float = 0.40) -> None:
        """Khởi tạo MotionFeatures.

        Args:
            still_velocity_threshold: Ngưỡng vận tốc chuẩn hóa / giây tối thiểu để xác định di chuyển.
        """
        self.still_velocity_threshold = still_velocity_threshold
        self.prev_center: Optional[Tuple[float, float]] = None
        self.prev_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Đặt lại lịch sử theo dõi khi mất bàn tay."""
        self.prev_center = None
        self.prev_timestamp = None

    def update(self, observation: HandObservation) -> MotionObservation:
        """Tính toán các đặc trưng chuyển động từ quan sát bàn tay hiện tại.

        Args:
            observation: Đối tượng HandObservation của khung hình hiện tại.

        Returns:
            MotionObservation: Bộ đặc trưng chuyển động kèm vận tốc chuẩn hóa theo thời gian.
        """
        curr_center = observation.hand_center
        curr_time = observation.timestamp
        palm_size = max(observation.palm_size, 1e-6)

        if self.prev_center is None or self.prev_timestamp is None:
            self.prev_center = curr_center
            self.prev_timestamp = curr_time
            return MotionObservation(
                dt=0.0,
                velocity_normalized=0.0,
                dx_normalized=0.0,
                dy_normalized=0.0,
                direction="Still",
                current_center=curr_center,
            )

        dt = max(curr_time - self.prev_timestamp, 1e-5)
        raw_dx = curr_center[0] - self.prev_center[0]
        raw_dy = curr_center[1] - self.prev_center[1]
        distance_norm = math.hypot(raw_dx, raw_dy) / palm_size

        velocity_norm = distance_norm / dt
        dx_norm = (raw_dx / palm_size) / dt
        dy_norm = (raw_dy / palm_size) / dt

        direction = "Still"
        if velocity_norm >= self.still_velocity_threshold:
            if abs(dx_norm) > abs(dy_norm):
                direction = "Move Right" if dx_norm > 0 else "Move Left"
            else:
                direction = "Move Down" if dy_norm > 0 else "Move Up"

        self.prev_center = curr_center
        self.prev_timestamp = curr_time

        return MotionObservation(
            dt=dt,
            velocity_normalized=velocity_norm,
            dx_normalized=dx_norm,
            dy_normalized=dy_norm,
            direction=direction,
            current_center=curr_center,
        )
