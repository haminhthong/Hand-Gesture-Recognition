"""Mô-đun PerformanceMonitor theo dõi FPS, Latency (ms) toàn cục và Breakdown chi tiết từng chặng."""

from collections import deque
import json
import logging
from pathlib import Path
import statistics
import time
from typing import Any, Deque, Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Bộ giám sát hiệu năng theo dõi Tốc độ khung hình (FPS) và Độ trễ xử lý (Latency ms).

    Hỗ trợ chi tiết Breakdown độ trễ theo từng chặng pipeline:
    - frame_capture
    - mediapipe
    - preprocess
    - static_classifier
    - dynamic_fsm
    - temporal_filter
    - event_mapper
    - render
    """

    STAGES = (
        "frame_capture",
        "mediapipe",
        "preprocess",
        "static_classifier",
        "dynamic_fsm",
        "temporal_filter",
        "event_mapper",
        "render",
    )

    def __init__(self, window_size: int = 120) -> None:
        """Khởi tạo PerformanceMonitor."""
        if window_size < 1:
            raise ValueError("Kích thước cửa sổ phải lớn hơn 0.")
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.latencies_ms: Deque[float] = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.started_at: float = time.perf_counter()
        self.previous_frame_at: Optional[float] = None
        self.stage_latencies: Dict[str, Deque[float]] = {}

    def record_stage(self, stage_name: str, latency_ms: float) -> None:
        """Ghi nhận độ trễ xử lý của một giai đoạn cụ thể (ms)."""
        if stage_name not in self.stage_latencies:
            self.stage_latencies[stage_name] = deque(maxlen=self.window_size)
        self.stage_latencies[stage_name].append(max(0.0, latency_ms))

    def record_frame(self, processing_started_at: float) -> None:
        """Ghi nhận mốc thời gian hoàn tất khung hình và đo độ trễ xử lý."""
        now = time.perf_counter()
        if self.previous_frame_at is not None:
            self.frame_times.append(now - self.previous_frame_at)
        self.previous_frame_at = now
        self.latencies_ms.append((now - processing_started_at) * 1000.0)
        self.total_frames += 1

    @property
    def average_fps(self) -> float:
        """Tính số khung hình trung bình trên giây (FPS) theo cửa sổ trượt."""
        if not self.frame_times:
            return 0.0
        average_frame_time = statistics.fmean(self.frame_times)
        return 1.0 / average_frame_time if average_frame_time > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Tính độ trễ xử lý trung bình (Latency ms) theo cửa sổ trượt."""
        if not self.latencies_ms:
            return 0.0
        return statistics.fmean(self.latencies_ms)

    def get_stage_stats(self, stage_name: str) -> Dict[str, float]:
        """Lấy số liệu thống kê (mean, p50, p95) cho một chặng cụ thể."""
        data = self.stage_latencies.get(stage_name)
        if not data:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
        arr = np.array(data, dtype=np.float32)
        return {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
        }

    def summary(self) -> Dict[str, Any]:
        """Xuất từ điển tóm tắt hiệu năng tổng thể và chi tiết từng chặng."""
        stage_summary = {}
        for stage in self.stage_latencies:
            stage_summary[stage] = self.get_stage_stats(stage)

        p50 = float(np.percentile(self.latencies_ms, 50)) if self.latencies_ms else 0.0
        p95 = float(np.percentile(self.latencies_ms, 95)) if self.latencies_ms else 0.0

        return {
            "total_frames": self.total_frames,
            "average_fps": round(self.average_fps, 2),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "stage_breakdown": stage_summary,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Lưu báo cáo tóm tắt hiệu năng ra tệp JSON."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=4, ensure_ascii=False)
        logger.info("Đã xuất báo cáo hiệu năng ra: %s", path)
