"""Mô-đun giám sát hiệu năng thực thi (FPS và độ trễ khung hình)."""

import statistics
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


class PerformanceMonitor:
    """Bộ theo dõi tốc độ khung hình (FPS) và độ trễ xử lý (Latency ms)."""

    def __init__(self, window_size: int = 120) -> None:
        if window_size < 1:
            raise ValueError("Kích thước cửa sổ phải lớn hơn 0.")
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.latencies_ms: Deque[float] = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.previous_frame_at: Optional[float] = None

    def record_frame(self, processing_started_at: float) -> None:
        """Ghi nhận mốc hoàn tất khung hình và đo độ trễ xử lý."""
        now = time.perf_counter()
        if self.previous_frame_at is not None:
            self.frame_times.append(now - self.previous_frame_at)
        self.previous_frame_at = now
        self.latencies_ms.append((now - processing_started_at) * 1000.0)
        self.total_frames += 1

    @property
    def average_fps(self) -> float:
        """Tính số khung hình trên giây (FPS) trung bình."""
        if not self.frame_times:
            return 0.0
        avg_dt = statistics.fmean(self.frame_times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Tính độ trễ xử lý trung bình (ms)."""
        if not self.latencies_ms:
            return 0.0
        return statistics.fmean(self.latencies_ms)

    def summary(self) -> Dict[str, Any]:
        """Tóm tắt các chỉ số hiệu năng chính."""
        return {
            "total_frames": self.total_frames,
            "average_fps": round(self.average_fps, 1),
            "average_latency_ms": round(self.average_latency_ms, 1),
        }
