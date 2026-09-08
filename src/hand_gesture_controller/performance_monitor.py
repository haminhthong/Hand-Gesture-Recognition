"""Mô-đun tương thích ngược cho PerformanceMonitor (chuyển tiếp tới package telemetry)."""

from .telemetry import performance as _performance

time = _performance.time  # giữ đường dẫn patch tương thích với API cũ
PerformanceMonitor = _performance.PerformanceMonitor

__all__ = ["PerformanceMonitor"]
