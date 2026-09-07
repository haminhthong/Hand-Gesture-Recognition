"""Mô-đun lọc mượt tọa độ con trỏ (Cursor Filter) sử dụng bộ lọc Exponential Moving Average phụ thuộc thời gian (Time-Aware EMA)."""

import math
from typing import Optional, Tuple


class CursorFilter:
    r"""Bộ lọc mượt tọa độ con trỏ 2D thích ứng theo thời gian (Time-Aware EMA Filter).

    Sử dụng công thức tính hệ số làm mượt \(\alpha\) thích ứng theo khoảng thời gian thực tế giữa hai khung hình \(\Delta t\):
    \[
        \alpha = 1 - e^{-\frac{\Delta t}{\tau}}
    \]
    Trong đó \(\tau\) (tau) là hằng số thời gian đáp ứng (time constant).
    Cách tiếp cận này loại bỏ hoàn toàn sự phụ thuộc vào FPS biến thiên,
    giữ cho độ mượt của con trỏ chuột không bị giật lag khi FPS sụt giảm hoặc dao động.
    """

    def __init__(self, tau: float = 0.08) -> None:
        """Khởi tạo CursorFilter.

        Args:
            tau: Hằng số thời gian phản hồi (giây). Giá trị nhỏ hơn giúp phản hồi nhanh hơn, lớn hơn giúp mượt hơn.
        """
        if tau <= 0:
            raise ValueError("Hằng số thời gian tau phải lớn hơn 0.")
        self.tau = tau
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Đặt lại trạng thái lọc khi mất dấu con trỏ hoặc bàn tay."""
        self.smooth_x = None
        self.smooth_y = None
        self.last_timestamp = None

    def filter(
        self, raw_x: float, raw_y: float, timestamp: float
    ) -> Tuple[int, int]:
        """Cập nhật tọa độ thô và trả về tọa độ nguyên đã được lọc mượt.

        Args:
            raw_x: Tọa độ X thô (pixel).
            raw_y: Tọa độ Y thô (pixel).
            timestamp: Thời điểm đo (giây monotonic).

        Returns:
            Tuple[int, int]: Tọa độ (x, y) đã qua lọc làm mượt.
        """
        if self.smooth_x is None or self.smooth_y is None or self.last_timestamp is None:
            self.smooth_x = raw_x
            self.smooth_y = raw_y
            self.last_timestamp = timestamp
            return int(round(raw_x)), int(round(raw_y))

        dt = max(0.0, timestamp - self.last_timestamp)
        self.last_timestamp = timestamp

        # Alpha thích ứng thời gian: alpha = 1 - exp(-dt / tau)
        alpha = 1.0 - math.exp(-dt / self.tau)
        alpha = max(0.01, min(1.0, alpha))

        self.smooth_x = alpha * raw_x + (1.0 - alpha) * self.smooth_x
        self.smooth_y = alpha * raw_y + (1.0 - alpha) * self.smooth_y

        return int(round(self.smooth_x)), int(round(self.smooth_y))
