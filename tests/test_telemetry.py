"""Tests cho PerformanceMonitor."""

from unittest.mock import patch

from hand_gesture_controller.telemetry import PerformanceMonitor


def test_performance_monitor_fps_and_latency():
    monitor = PerformanceMonitor(window_size=10)

    # Các mốc thời gian hoàn tất từng frame (tương ứng với now trong record_frame)
    with patch("time.perf_counter", side_effect=[10.000, 10.033, 10.066]):
        # Frame 1: xử lý từ 9.985, hoàn tất tại 10.000 (độ trễ 15ms)
        monitor.record_frame(9.985)

        # Frame 2: xử lý từ 10.015, hoàn tất tại 10.033 (độ trễ 18ms, frame_time = 33ms)
        monitor.record_frame(10.015)

        # Frame 3: xử lý từ 10.050, hoàn tất tại 10.066 (độ trễ 16ms, frame_time = 33ms)
        monitor.record_frame(10.050)

    assert monitor.total_frames == 3
    assert monitor.average_latency_ms > 0
    assert monitor.average_fps > 0

    summary = monitor.summary()
    assert "average_fps" in summary
    assert "average_latency_ms" in summary
    assert summary["total_frames"] == 3
