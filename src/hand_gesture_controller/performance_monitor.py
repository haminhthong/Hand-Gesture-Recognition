"""Mô-đun tương thích ngược cho PerformanceMonitor (chuyển tiếp tới package telemetry)."""

from .telemetry.performance import PerformanceMonitor

__all__ = ["PerformanceMonitor"]
