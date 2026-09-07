"""Mô-đun ổn định cử chỉ tĩnh theo thời gian (Temporal Stabilizer) sử dụng Hysteresis thuần thời gian."""

from collections import deque
from typing import Deque, Optional, Tuple
import numpy as np

from ..schemas import StableGesture, StaticPrediction


class GestureStabilizer:
    """Bộ ổn định nhãn cử chỉ tĩnh dựa trên Hysteresis thời gian thực (Time-Based Hysteresis).

    Thay thế hoàn toàn cơ chế đếm số khung hình (frame-count window) bằng các khoảng thời gian vật lý:
    - Activation Dwell (120ms): Nhãn thô phải duy trì ổn định liên tục ít nhất 120ms để kích hoạt.
    - Release Dwell (100ms): Khi nhãn ổn định vắng mặt trong ít nhất 100ms, trạng thái được giải phóng.
    - History Horizon (250ms): Giới hạn cửa sổ lưu trữ các dự đoán trong quá khứ.

    Thuật toán hoạt động hoàn toàn độc lập với tốc độ khung hình (15 FPS, 30 FPS, 60 FPS).
    """

    def __init__(
        self,
        activation_dwell_ms: float = 120.0,
        release_dwell_ms: float = 100.0,
        history_horizon_ms: float = 250.0,
    ) -> None:
        """Khởi tạo GestureStabilizer.

        Args:
            activation_dwell_ms: Thời gian duy trì tối thiểu (ms) để kích hoạt cử chỉ mới.
            release_dwell_ms: Thời gian vắng mặt tối thiểu (ms) để giải phóng cử chỉ hiện tại.
            history_horizon_ms: Độ dài cửa sổ lịch sử lưu giữ (ms).
        """
        if activation_dwell_ms <= 0 or release_dwell_ms <= 0 or history_horizon_ms <= 0:
            raise ValueError("Các tham số thời gian dwell phải lớn hơn 0.")
        if activation_dwell_ms > history_horizon_ms:
            raise ValueError("Activation dwell không được vượt quá history horizon.")

        self.activation_dwell_sec = activation_dwell_ms / 1000.0
        self.release_dwell_sec = release_dwell_ms / 1000.0
        self.history_horizon_sec = history_horizon_ms / 1000.0

        # Lịch sử dạng deque các tuple: (timestamp_sec, StaticPrediction)
        self._history: Deque[Tuple[float, StaticPrediction]] = deque()

        # Trạng thái cử chỉ ổn định hiện tại
        self._stable_label: Optional[str] = None
        self._stable_activated_at: Optional[float] = None
        self._candidate_label: Optional[str] = None
        self._candidate_since: Optional[float] = None

    def reset(self) -> None:
        """Xóa toàn bộ bộ đệm và trạng thái khi mất dấu bàn tay."""
        self._history.clear()
        self._stable_label = None
        self._stable_activated_at = None
        self._candidate_label = None
        self._candidate_since = None

    def update(self, prediction: StaticPrediction, timestamp: float) -> StableGesture:
        """Thêm một dự đoán thô mới tại mốc thời gian timestamp và trả về StableGesture.

        Args:
            prediction: Dự đoán StaticPrediction từ static classifier.
            timestamp: Thời điểm ghi nhận (giây monotonic).

        Returns:
            StableGesture: Trạng thái cử chỉ đã qua lọc mượt ổn định thời gian.
        """
        self._history.append((timestamp, prediction))

        # 1. Cắt tỉa các mẫu nằm ngoài history horizon
        cutoff = timestamp - self.history_horizon_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        curr_raw = prediction.label

        # 2. Kiểm tra Release Dwell nếu đang có stable_label
        if self._stable_label is not None:
            # Tìm mốc thời gian gần nhất mà stable_label xuất hiện trong history
            last_seen_stable = -1.0
            for t, pred in reversed(self._history):
                if pred.label == self._stable_label and not pred.rejected:
                    last_seen_stable = t
                    break

            # Nếu không thấy stable_label trong >= release_dwell_sec -> Release
            if last_seen_stable < 0 or (timestamp - last_seen_stable >= self.release_dwell_sec):
                self._stable_label = None
                self._stable_activated_at = None

        # 3. Kiểm tra Activation Dwell
        if curr_raw != "NoAction" and not prediction.rejected:
            if self._candidate_label == curr_raw:
                # Đang tiếp tục duy trì candidate
                candidate_duration = timestamp - (self._candidate_since or timestamp)
                if candidate_duration >= self.activation_dwell_sec:
                    if self._stable_label != curr_raw:
                        self._stable_label = curr_raw
                        self._stable_activated_at = timestamp
            else:
                # Bắt đầu theo dõi candidate mới
                self._candidate_label = curr_raw
                self._candidate_since = timestamp
        else:
            self._candidate_label = None
            self._candidate_since = None

        # 4. Tạo kết quả StableGesture
        effective_label = self._stable_label or "NoAction"
        effective_confidence = prediction.confidence if effective_label == curr_raw else 0.85
        dwell_ms = 0.0
        if self._stable_activated_at is not None and self._stable_label is not None:
            dwell_ms = max(0.0, (timestamp - self._stable_activated_at) * 1000.0)

        return StableGesture(
            label=effective_label,
            confidence=effective_confidence,
            dwell_time_ms=dwell_ms,
            timestamp=timestamp,
        )
