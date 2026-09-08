"""Ứng dụng chính điều phối Canonical HCI Pipeline: MediaPipe -> Calibrated SVM -> Stabilizer -> Event Mapper -> Canvas."""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .application.object_manager import DraggableObjectManager
from .application.shape_menu import ShapeMenu
from .config import RuntimeConfig
from .events.event_mapper import GestureEventMapper
from .perception.hand_detector import HandDetector
from .recognition.dynamic_fsm import DynamicGestureFSM
from .recognition.rule_baseline import RuleStaticBaseline
from .recognition.static_predictor import StaticGesturePredictor
from .schemas import GestureEvent, HandObservation, StableGesture, StaticPrediction
from .telemetry.performance import PerformanceMonitor
from .temporal.gesture_stabilizer import GestureStabilizer

logger = logging.getLogger("hand_gesture_controller")

def setup_logging(log_level: str = "INFO") -> None:
    """Cấu hình định dạng và mức ghi log hệ thống."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class HandGestureApp:
    """Ứng dụng chính điều phối camera, nhận diện cử chỉ tĩnh/động và điều khiển GUI tương tác."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        config: Optional[RuntimeConfig] = None,
        smoothing_window: int = 5,
        smoothing_votes: int = 3,
        benchmark_output: Optional[str] = None,
    ) -> None:
        """Khởi tạo toàn bộ mô-đun ứng dụng và camera."""
        if width <= 0:
            raise ValueError("Chiều rộng khung hình (--width) phải lớn hơn 0.")
        if height <= 0:
            raise ValueError("Chiều cao khung hình (--height) phải lớn hơn 0.")
        if camera_index < 0:
            raise ValueError("Chỉ số camera (--camera) phải lớn hơn hoặc bằng 0.")

        self.width = width
        self.height = height
        self.config = config or RuntimeConfig(
            camera_index=camera_index,
            target_width=width,
            target_height=height,
            benchmark_output=benchmark_output,
        )

        self.show_debug: bool = self.config.show_debug_hud
        self.benchmark_output = benchmark_output or self.config.benchmark_output

        # 1. Perception
        self.detector = HandDetector(
            detectionCon=self.config.min_detection_confidence,
            trackCon=self.config.min_tracking_confidence,
            maxHands=self.config.max_num_hands,
        )

        # 2. Recognition (Primary: SVM, Fallback: Rules, Dynamic: FSM)
        self.static_predictor = StaticGesturePredictor(
            model_bundle_path=self.config.model_path,
            default_accept_threshold=self.config.default_accept_threshold,
            accept_thresholds=self.config.accept_thresholds,
        )
        self.rule_baseline = RuleStaticBaseline()
        self.dynamic_fsm = DynamicGestureFSM(
            on_off_timeout_seconds=self.config.on_off_timeout_seconds,
            sos_timeout_seconds=self.config.sos_timeout_seconds,
            enable_experimental_gestures=self.config.enable_experimental_gestures,
        )

        # 3. Temporal Stabilization
        self.gesture_stabilizer = GestureStabilizer(
            activation_dwell_ms=self.config.activation_dwell_ms,
            release_dwell_ms=self.config.release_dwell_ms,
            history_horizon_ms=self.config.history_horizon_ms,
        )

        # 4. Events & Application
        self.event_mapper = GestureEventMapper(cooldowns=self.config.cooldowns)
        self.object_manager = DraggableObjectManager(
            cursor_tau=self.config.cursor_tau,
        )
        self.shape_menu = ShapeMenu()

        # 5. Telemetry
        self.performance = PerformanceMonitor()

        # Trạng thái theo dõi
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = camera_index
        self.last_hand_seen_time: float = 0.0

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        """Mở camera với thử lại và fallback."""
        for index in dict.fromkeys([camera_index, 0, 1]):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info("Đã kết nối thành công với camera index %d", index)
                return cap
            cap.release()
        raise RuntimeError("Không tìm thấy camera khả dụng trên hệ thống.")

    def _draw_hud(
        self,
        frame: np.ndarray,
        stable_label: str,
        dynamic_label: str,
        source: str,
        confidence: float,
    ) -> None:
        """Vẽ card HUD hiển thị thông số FPS, Latency Breakdown và trạng thái cử chỉ."""
        if not self.show_debug:
            return

        summary = self.performance.summary()
        fps_text = f"FPS: {summary['average_fps']:.1f}"
        lat_text = f"Latency: {summary['average_latency_ms']:.1f}ms (P95: {summary['p95_latency_ms']:.1f}ms)"
        mode_text = f"Source: {source.upper()} | Conf: {confidence:.2f}"
        state_text = f"Static: {stable_label} | Motion: {dynamic_label}"

        # Card nền
        cv2.rectangle(frame, (10, 10), (360, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (360, 110), (0, 255, 0), 1)

        cv2.putText(frame, fps_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, lat_text, (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, mode_text, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 255), 1)
        cv2.putText(frame, state_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)

    def run(self) -> None:
        """Vòng lặp sự kiện chính của ứng dụng."""
        self.cap = self._open_camera(self.camera_index)
        logger.info("=== BẮT ĐẦU RUNTIME HAND GESTURE CONTROLLER ===")
        logger.info("Mô hình chính: %s | Fallback Rules: %s",
                    "SVM Calibrated" if self.static_predictor.is_ready else "Rule Baseline (Fallback)",
                    self.config.fallback_to_rules)
        logger.info("Phím tắt: 'Q'=Thoát, 'D'=Bật/Tắt HUD, 'C'=Xóa canvas")

        try:
            while True:
                t_frame_start = time.perf_counter()

                # --- STAGE 1: FRAME CAPTURE ---
                t0 = time.perf_counter()
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    logger.warning("Không thể đọc khung hình từ camera. Dừng ứng dụng...")
                    break
                if self.config.mirror_camera:
                    frame = cv2.flip(frame, 1)

                actual_h, actual_w = frame.shape[:2]
                self.performance.record_stage("frame_capture", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 2: PERCEPTION (MediaPipe Hands) ---
                t0 = time.perf_counter()
                observation: Optional[HandObservation] = self.detector.process(frame, timestamp=t_frame_start)
                self.performance.record_stage("mediapipe", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 3: PREPROCESS & MOTION FEATURES ---
                t0 = time.perf_counter()
                dynamic_gesture = "Still"
                if observation is not None:
                    self.last_hand_seen_time = t_frame_start
                self.performance.record_stage("preprocess", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 4: STATIC CLASSIFICATION ---
                t0 = time.perf_counter()
                if observation is not None:
                    if self.static_predictor.is_ready:
                        static_pred = self.static_predictor.predict_observation(observation)
                    elif self.config.fallback_to_rules:
                        static_pred = self.rule_baseline.predict_observation(observation)
                    else:
                        static_pred = StaticPrediction(
                            label="NoAction", confidence=0.0, rejected=True, source="no_model"
                        )
                else:
                    static_pred = StaticPrediction(
                        label="NoAction", confidence=0.0, rejected=True, source="no_hand"
                    )
                self.performance.record_stage("static_classifier", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 5: DYNAMIC FSM ---
                t0 = time.perf_counter()
                dynamic_gesture = self.dynamic_fsm.update(observation)
                self.performance.record_stage("dynamic_fsm", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 6: TEMPORAL STABILIZER & FAILSAFE HAND-LOSS ---
                t0 = time.perf_counter()
                cursor_pos: Optional[Tuple[int, int]] = None
                if observation is not None:
                    stable_gesture = self.gesture_stabilizer.update(static_pred, timestamp=t_frame_start)
                    cursor_pos = self.object_manager.get_cursor_position(observation, actual_w, actual_h)
                else:
                    # Kiểm tra mất dấu tay vượt quá hand_loss_timeout_ms (150ms)
                    hand_lost_duration = t_frame_start - self.last_hand_seen_time
                    if hand_lost_duration >= (self.config.hand_loss_timeout_ms / 1000.0):
                        hand_loss_event = self.event_mapper.handle_hand_loss(timestamp=t_frame_start)
                        self.gesture_stabilizer.reset()
                        self.dynamic_fsm.reset()
                        if hand_loss_event == GestureEvent.STOP_DRAG:
                            self.object_manager.update_event(
                                None, GestureEvent.STOP_DRAG, actual_w, actual_h
                            )
                    stable_gesture = StableGesture(
                        label="NoAction", confidence=0.0, dwell_time_ms=0.0, timestamp=t_frame_start
                    )
                self.performance.record_stage("temporal_filter", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 7: EVENT MAPPER & APPLICATION ACTION ---
                t0 = time.perf_counter()
                event = self.event_mapper.map_gesture_to_event(
                    gesture_label=stable_gesture.label,
                    motion_label=dynamic_gesture,
                    timestamp=t_frame_start,
                )

                if observation is not None:
                    self.object_manager.update_event(
                        observation.landmarks,
                        event,
                        actual_w,
                        actual_h,
                        cursor_pos=cursor_pos,
                    )
                    self.shape_menu.update(
                        observation,
                        stable_gesture.label,
                        actual_w,
                        actual_h,
                        self.object_manager,
                        event=event,
                        cursor_pos=cursor_pos,
                    )
                self.performance.record_stage("event_mapper", (time.perf_counter() - t0) * 1000.0)

                # --- STAGE 8: RENDER & DISPLAY ---
                t0 = time.perf_counter()
                # Vẽ khung xương bàn tay nếu bật debug
                if self.show_debug:
                    frame = self.detector.draw_landmarks(frame, observation)

                # Vẽ canvas vật thể và menu
                self.object_manager.draw_all(frame)
                if cursor_pos is not None:
                    self.object_manager.draw_cursor(
                        frame, None, actual_w, actual_h, cursor_pos=cursor_pos
                    )
                self.shape_menu.draw(frame, actual_h)

                # Vẽ HUD Telemetry
                self._draw_hud(
                    frame=frame,
                    stable_label=stable_gesture.label,
                    dynamic_label=dynamic_gesture,
                    source=static_pred.source,
                    confidence=static_pred.confidence,
                )

                cv2.imshow("Hand Gesture HCI Controller", frame)
                self.performance.record_stage("render", (time.perf_counter() - t0) * 1000.0)

                # Ghi nhận thời gian tổng cộng của frame
                self.performance.record_frame(t_frame_start)

                # Bắt phím điều khiển
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                elif key in (ord("d"), ord("D")):
                    self.show_debug = not self.show_debug
                elif key in (ord("c"), ord("C")):
                    self.object_manager.objects.clear()

        finally:
            self.close()

    def close(self) -> None:
        """Giải phóng camera, MediaPipe và lưu telemetry benchmark nếu có."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()

        if self.benchmark_output:
            self.performance.save(self.benchmark_output)
            logger.info("Đã lưu kết quả đo hiệu năng vào: %s", self.benchmark_output)


def parse_args() -> argparse.Namespace:
    """Xử lý đối số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Real-Time Landmark-Based Hand Gesture HCI Controller"
    )
    parser.add_argument("--camera", type=int, default=None, help="Chỉ số camera")
    parser.add_argument("--width", type=int, default=None, help="Chiều rộng khung hình")
    parser.add_argument("--height", type=int, default=None, help="Chiều cao khung hình")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn model bundle")
    parser.add_argument("--config", type=str, default=None, help="Đường dẫn file config runtime.yaml")
    parser.add_argument("--benchmark-output", type=str, default=None, help="Đường dẫn file json lưu hiệu năng")
    return parser.parse_args()


def main() -> None:
    """Điểm nhập CLI của ứng dụng."""
    setup_logging("INFO")
    args = parse_args()

    cfg = RuntimeConfig.from_yaml(args.config) if args.config else RuntimeConfig()
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.width is not None:
        cfg.target_width = args.width
    if args.height is not None:
        cfg.target_height = args.height
    if args.model is not None:
        cfg.model_path = args.model
    if args.benchmark_output:
        cfg.benchmark_output = args.benchmark_output

    # Cho phép chạy console script từ thư mục khác mà vẫn tìm thấy model mặc định
    # trong root của source tree.
    if not Path(cfg.model_path).is_absolute() and not Path(cfg.model_path).exists():
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / cfg.model_path
        if candidate.exists():
            cfg.model_path = str(candidate)

    app = HandGestureApp(
        camera_index=cfg.camera_index,
        width=cfg.target_width,
        height=cfg.target_height,
        config=cfg,
        benchmark_output=cfg.benchmark_output,
    )
    app.run()


if __name__ == "__main__":
    main()
