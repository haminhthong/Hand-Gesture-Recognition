"""Mô-đun đánh giá chuẩn End-to-End HCI Benchmark (không chỉ dừng lại ở độ chính xác classifier)."""

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.events.event_mapper import GestureEventMapper
from hand_gesture_controller.recognition.dynamic_fsm import DynamicGestureFSM
from hand_gesture_controller.recognition.rule_baseline import RuleStaticBaseline
from hand_gesture_controller.recognition.static_predictor import StaticGesturePredictor
from hand_gesture_controller.schemas import GestureEvent, HandObservation
from hand_gesture_controller.temporal.gesture_stabilizer import GestureStabilizer

logger = logging.getLogger("benchmark_hci")


class HCISystemBenchmark:
    """Đánh giá toàn chuỗi tương tác người-máy: Observation -> Recognition -> Stabilizer -> Event."""

    def __init__(
        self,
        model_bundle_path: Optional[str] = None,
        activation_dwell_ms: float = 120.0,
        release_dwell_ms: float = 100.0,
    ) -> None:
        self.predictor = StaticGesturePredictor(model_bundle_path=model_bundle_path)
        self.rule_fallback = RuleStaticBaseline()
        self.stabilizer = GestureStabilizer(
            activation_dwell_ms=activation_dwell_ms,
            release_dwell_ms=release_dwell_ms,
        )
        self.dynamic_fsm = DynamicGestureFSM()
        self.event_mapper = GestureEventMapper()

    def run_trial(
        self,
        observations: List[HandObservation],
        expected_target_event: GestureEvent,
    ) -> Dict[str, Any]:
        """Chạy một thử nghiệm tương tác (Trial) trên một chuỗi HandObservation theo thời gian."""
        self.stabilizer.reset()
        self.dynamic_fsm.reset()
        self.event_mapper.reset()

        events_generated: List[Tuple[float, GestureEvent]] = []
        latencies_ms: List[float] = []
        trial_start_time = observations[0].timestamp if observations else 0.0

        for obs in observations:
            t0 = time.perf_counter()

            # Static recognition
            if self.predictor.is_ready:
                static_pred = self.predictor.predict_observation(obs)
            else:
                static_pred = self.rule_fallback.predict_observation(obs)

            # Dynamic FSM
            dynamic_gesture = self.dynamic_fsm.update(obs)

            # Temporal Stabilizer
            stable_gesture = self.stabilizer.update(static_pred, timestamp=obs.timestamp)

            # Event Mapper
            ev = self.event_mapper.map_gesture_to_event(
                gesture_label=stable_gesture.label,
                motion_label=dynamic_gesture,
                timestamp=obs.timestamp,
            )

            latency = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(latency)

            if ev != GestureEvent.NONE:
                events_generated.append((obs.timestamp, ev))

        # Phân tích kết quả trial
        matched_events = [ev for _, ev in events_generated if ev == expected_target_event]
        success = len(matched_events) >= 1
        first_event_latency_ms = None
        if success:
            first_event_time = next(t for t, ev in events_generated if ev == expected_target_event)
            first_event_latency_ms = max(0.0, (first_event_time - trial_start_time) * 1000.0)

        unwanted_events = [ev for _, ev in events_generated if ev != expected_target_event and ev != GestureEvent.DRAG]

        return {
            "success": success,
            "target_event": expected_target_event.value,
            "total_events_emitted": len(events_generated),
            "unwanted_events_count": len(unwanted_events),
            "gesture_to_event_latency_ms": first_event_latency_ms,
            "mean_pipeline_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        }


def generate_synthetic_observation_sequence(
    gesture_name: str,
    fps: float = 30.0,
    duration_sec: float = 1.0,
) -> List[HandObservation]:
    """Tạo chuỗi synthetic HandObservation mô phỏng người dùng thực hiện cử chỉ."""
    dt = 1.0 / fps
    total_frames = int(round(fps * duration_sec))
    sequence = []

    # Định nghĩa hình học landmark mô phỏng cho từng cử chỉ
    base_coords = np.zeros((21, 3), dtype=np.float32)
    base_coords[0] = [0.5, 0.8, 0.0]  # Cổ tay
    base_coords[9] = [0.5, 0.5, 0.0]  # Middle MCP

    if gesture_name == "Select":
        # Ngón cái và trỏ chạm nhau
        base_coords[4] = [0.48, 0.45, 0.0]
        base_coords[8] = [0.49, 0.45, 0.0]
        # Các ngón khác gập
        for i in [12, 16, 20]:
            base_coords[i] = [0.5, 0.65, 0.0]
    elif gesture_name == "Stop":
        # Cả 5 ngón xòe thẳng
        for i, tip in enumerate([4, 8, 12, 16, 20]):
            base_coords[tip] = [0.3 + i * 0.1, 0.3, 0.0]
    elif gesture_name == "Peace":
        # Trỏ và giữa duỗi
        base_coords[8] = [0.45, 0.3, 0.0]
        base_coords[12] = [0.55, 0.3, 0.0]
        for i in [4, 16, 20]:
            base_coords[i] = [0.5, 0.65, 0.0]
    elif gesture_name == "Fist":
        for tip in [4, 8, 12, 16, 20]:
            base_coords[tip] = [0.5, 0.65, 0.0]
    else:  # NoAction
        for tip in [4, 8, 12, 16, 20]:
            base_coords[tip] = [0.4 + np.random.uniform(0, 0.2), 0.55, 0.0]

    for frame_idx in range(total_frames):
        noise = np.random.normal(0, 0.002, (21, 3)).astype(np.float32)
        lm = base_coords + noise
        t = frame_idx * dt
        obs = HandObservation(
            landmarks=lm,
            handedness="Right",
            handedness_score=0.99,
            timestamp=t,
            frame_width=640,
            frame_height=480,
            palm_size=0.30,
            hand_center=(0.5, 0.6),
        )
        sequence.append(obs)

    return sequence


def run_hci_benchmark(
    model_bundle_path: Optional[str] = None,
    trials_per_gesture: int = 10,
) -> Dict[str, Any]:
    """Chạy bài test benchmark HCI toàn diện."""
    benchmark = HCISystemBenchmark(model_bundle_path=model_bundle_path)
    gesture_targets = [
        ("Select", GestureEvent.START_DRAG),
        ("Stop", GestureEvent.DELETE_OBJECT),
        ("Peace", GestureEvent.OPEN_MENU),
    ]

    results = {}
    total_trials = 0
    successful_trials = 0
    latencies = []

    logger.info("=== BẮT ĐẦU CHẠY END-TO-END HCI BENCHMARK ===")
    for g_name, target_event in gesture_targets:
        g_success = 0
        g_latencies = []

        for _ in range(trials_per_gesture):
            seq = generate_synthetic_observation_sequence(g_name, fps=30.0, duration_sec=0.8)
            trial_res = benchmark.run_trial(seq, target_event)
            total_trials += 1
            if trial_res["success"]:
                successful_trials += 1
                g_success += 1
                if trial_res["gesture_to_event_latency_ms"] is not None:
                    g_latencies.append(trial_res["gesture_to_event_latency_ms"])
                    latencies.append(trial_res["gesture_to_event_latency_ms"])

        results[g_name] = {
            "success_rate": g_success / trials_per_gesture,
            "mean_gesture_to_event_latency_ms": float(np.mean(g_latencies)) if g_latencies else 0.0,
            "p95_latency_ms": float(np.percentile(g_latencies, 95)) if g_latencies else 0.0,
        }
        logger.info("Gesture %s -> Success Rate: %.1f%% | Mean Latency: %.1f ms",
                    g_name, results[g_name]["success_rate"] * 100, results[g_name]["mean_gesture_to_event_latency_ms"])

    overall_success_rate = successful_trials / total_trials if total_trials > 0 else 0.0
    overall_summary = {
        "overall_action_success_rate": overall_success_rate,
        "overall_mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "overall_p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "per_gesture": results,
    }
    logger.info("Overall HCI Action Success Rate: %.2f%%", overall_success_rate * 100)
    return overall_summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Chạy bài benchmark End-to-End HCI System.")
    parser.add_argument("--model", type=str, default="models/static_gesture_svm_v1.joblib", help="Đường dẫn model bundle")
    parser.add_argument("--trials", type=int, default=15, help="Số lần thử nghiệm mỗi gesture")
    args = parser.parse_args()

    model_path = args.model if os.path.exists(args.model) else None
    run_hci_benchmark(model_bundle_path=model_path, trials_per_gesture=args.trials)


if __name__ == "__main__":
    main()
