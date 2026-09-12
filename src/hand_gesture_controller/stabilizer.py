"""Mô-đun ổn định cử chỉ tĩnh theo thời gian (Temporal Stabilizer) bằng Time-Based Hysteresis."""

from collections import deque
from typing import Deque, Optional, Tuple

from .schemas import StableGesture, StaticPrediction


class GestureStabilizer:
    """Bộ ổn định nhãn cử chỉ tĩnh dựa trên Hysteresis thời gian thực.

    Giải quyết hiện tượng rung giật (jitter) giữa các frame:
    - Activation Dwell (120ms): Nhãn thô phải duy trì ổn định liên tục ít nhất 120ms để kích hoạt.
    - Release Dwell (100ms): Khi nhãn ổn định vắng mặt trong ít nhất 100ms, trạng thái được giải phóng.
    - History Horizon (250ms): Giới hạn cửa sổ thời gian lưu trữ lịch sử dự đoán.
    """

    def __init__(
        self,
        activation_dwell_ms: float = 120.0,
        release_dwell_ms: float = 100.0,
        history_horizon_ms: float = 250.0,
    ) -> None:
        if activation_dwell_ms <= 0 or release_dwell_ms <= 0 or history_horizon_ms <= 0:
            raise ValueError("Các tham số thời gian dwell phải lớn hơn 0.")
        if activation_dwell_ms > history_horizon_ms:
            raise ValueError("Activation dwell không được vượt quá history horizon.")

        self.activation_dwell_sec = activation_dwell_ms / 1000.0
        self.release_dwell_sec = release_dwell_ms / 1000.0
        self.history_horizon_sec = history_horizon_ms / 1000.0

        self._history: Deque[Tuple[float, StaticPrediction]] = deque()
        self._stable_label: Optional[str] = None
        self._stable_activated_at: Optional[float] = None
        self._candidate_label: Optional[str] = None
        self._candidate_since: Optional[float] = None

    def reset(self) -> None:
        """Xóa bộ đệm và trạng thái khi mất dấu bàn tay."""
        self._history.clear()
        self._stable_label = None
        self._stable_activated_at = None
        self._candidate_label = None
        self._candidate_since = None

    def update(self, prediction: StaticPrediction, timestamp: float) -> StableGesture:
        """Thêm một dự đoán mới tại timestamp và trả về StableGesture."""
        self._history.append((timestamp, prediction))

        # 1. Cắt tỉa các mẫu cũ hơn history horizon
        cutoff = timestamp - self.history_horizon_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        curr_raw = prediction.label

        # 2. Kiểm tra Release Dwell nếu đang có nhãn ổn định
        if self._stable_label is not None:
            last_seen_stable = -1.0
            for t, pred in reversed(self._history):
                if pred.label == self._stable_label and not pred.rejected:
                    last_seen_stable = t
                    break

            if last_seen_stable < 0 or (timestamp - last_seen_stable >= self.release_dwell_sec):
                self._stable_label = None
                self._stable_activated_at = None

        # 3. Kiểm tra Activation Dwell cho candidate mới
        if curr_raw != "NoAction" and not prediction.rejected:
            if self._candidate_label == curr_raw:
                if self._candidate_since is not None:
                    candidate_duration = timestamp - self._candidate_since
                else:
                    candidate_duration = 0.0
                if candidate_duration >= self.activation_dwell_sec:
                    if self._stable_label != curr_raw:
                        self._stable_label = curr_raw
                        self._stable_activated_at = timestamp
            else:
                self._candidate_label = curr_raw
                self._candidate_since = timestamp
        else:
            self._candidate_label = None
            self._candidate_since = None

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
