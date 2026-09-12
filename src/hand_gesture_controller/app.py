"""Ứng dụng chính điều phối Webcam -> MediaPipe -> Classifier/Rules -> Stabilizer -> Event Mapper -> Canvas."""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .canvas import DraggableObjectManager, ShapeMenu
from .classifier import StaticGestureClassifier
from .config import RuntimeConfig
from .dynamic_gesture import DynamicGestureFSM
from .event_mapper import GestureEventMapper
from .hand_detector import HandDetector
from .rule_baseline import RuleStaticBaseline
from .schemas import GestureEvent, HandObservation, StableGesture, StaticPrediction
from .stabilizer import GestureStabilizer
from .telemetry import PerformanceMonitor

logger = logging.getLogger("hand_gesture_controller")


def setup_logging(log_level: str = "INFO") -> None:
    """Cấu hình định dạng log hệ thống."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class HandGestureApp:
    """Ứng dụng điều khiển tương tác người–máy bằng cử chỉ bàn tay."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        config: Optional[RuntimeConfig] = None,
        mode: Optional[str] = None,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Kích thước khung hình phải lớn hơn 0.")
        if camera_index < 0:
            raise ValueError("Chỉ số camera phải lớn hơn hoặc bằng 0.")

        self.width = width
        self.height = height
        self.config = config or RuntimeConfig(
            camera_index=camera_index,
            target_width=width,
            target_height=height,
        )
        if mode is not None:
            self.config.mode = mode

        self.show_debug: bool = self.config.show_debug_hud

        # 1. Perception
        self.detector = HandDetector(
            detectionCon=self.config.min_detection_confidence,
            trackCon=self.config.min_tracking_confidence,
            maxHands=self.config.max_num_hands,
        )

        # 2. Recognition Mode (SVM hoặc Rule Baseline)
        self.mode = self.config.mode
        self.classifier: Optional[StaticGestureClassifier] = None
        self.rule_baseline = RuleStaticBaseline()

        if self.mode == "svm":
            # Kiểm tra đường dẫn model
            model_file = Path(self.config.model_path)
            if not model_file.is_absolute() and not model_file.exists():
                # Thử tìm từ thư mục gốc dự án
                project_root = Path(__file__).resolve().parents[2]
                candidate = project_root / self.config.model_path
                if candidate.exists():
                    self.config.model_path = str(candidate)
                    model_file = candidate

            if not model_file.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy mô hình SVM tại '{self.config.model_path}'.\n"
                    f"Vui lòng huấn luyện mô hình bằng 'python scripts/train.py' "
                    f"hoặc chạy chế độ rule baseline với '--mode rules'."
                )

            self.classifier = StaticGestureClassifier(
                model_bundle_path=self.config.model_path,
                confidence_threshold=self.config.confidence_threshold,
            )

        # 3. Dynamic Gesture FSM & Temporal Stabilizer
        self.dynamic_fsm = DynamicGestureFSM(
            timeout_seconds=self.config.on_off_timeout_seconds,
        )
        self.gesture_stabilizer = GestureStabilizer(
            activation_dwell_ms=self.config.activation_dwell_ms,
            release_dwell_ms=self.config.release_dwell_ms,
            history_horizon_ms=self.config.history_horizon_ms,
        )

        # 4. Event Mapper & Canvas
        self.event_mapper = GestureEventMapper(cooldowns=self.config.cooldowns)
        self.object_manager = DraggableObjectManager(cursor_tau=self.config.cursor_tau)
        self.shape_menu = ShapeMenu()

        # 5. Performance Monitor
        self.performance = PerformanceMonitor()

        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = camera_index
        self.last_hand_seen_time: float = 0.0

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        """Mở camera khả dụng."""
        for idx in dict.fromkeys([camera_index, 0, 1]):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info("Kết nối thành công camera index %d", idx)
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
        """Vẽ bảng thông số HUD tinh gọn."""
        if not self.show_debug:
            return

        summary = self.performance.summary()
        fps_text = f"FPS: {summary['average_fps']:.1f}"
        lat_text = f"Latency: {summary['average_latency_ms']:.1f}ms"
        mode_text = f"Mode: {self.mode.upper()} | Conf: {confidence:.2f}"
        state_text = f"Gesture: {stable_label}" + (f" | Motion: {dynamic_label}" if dynamic_label != "Still" else "")

        # Card nền trong suốt nhẹ
        cv2.rectangle(frame, (10, 10), (320, 110), (25, 25, 25), -1)
        cv2.rectangle(frame, (10, 10), (320, 110), (0, 200, 100), 1)

        cv2.putText(frame, fps_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
        cv2.putText(frame, lat_text, (20, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(frame, mode_text, (20, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(frame, state_text, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 80), 1)

    def run(self) -> None:
        """Vòng lặp tương tác chính."""
        self.cap = self._open_camera(self.camera_index)
        logger.info("=== HAND GESTURE CONTROLLER ĐANG CHẠY ===")
        logger.info("Chế độ: %s", self.mode.upper())
        logger.info("Phím tắt: 'Q'=Thoát, 'D'=Bật/Tắt HUD, 'C'=Xóa canvas")

        try:
            while True:
                t_frame_start = time.perf_counter()

                ok, frame = self.cap.read()
                if not ok or frame is None:
                    logger.warning("Không thể đọc khung hình từ camera. Dừng lại.")
                    break

                if self.config.mirror_camera:
                    frame = cv2.flip(frame, 1)

                actual_h, actual_w = frame.shape[:2]

                # 1. Perception
                observation: Optional[HandObservation] = self.detector.process(
                    frame, timestamp=t_frame_start
                )

                # 2. Recognition
                cursor_pos: Optional[Tuple[int, int]] = None
                if observation is not None:
                    self.last_hand_seen_time = t_frame_start
                    if self.mode == "svm" and self.classifier is not None:
                        static_pred = self.classifier.predict_observation(observation)
                    else:
                        static_pred = self.rule_baseline.predict_observation(observation)

                    dynamic_gesture = self.dynamic_fsm.update(observation)
                    stable_gesture = self.gesture_stabilizer.update(static_pred, timestamp=t_frame_start)
                    cursor_pos = self.object_manager.get_cursor_position(observation, actual_w, actual_h)
                else:
                    dynamic_gesture = self.dynamic_fsm.update(None)
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
                    static_pred = StaticPrediction(
                        label="NoAction", confidence=0.0, rejected=True, source="no_hand"
                    )

                # 3. Event Mapping
                event = self.event_mapper.map_gesture_to_event(
                    gesture_label=stable_gesture.label,
                    motion_label=dynamic_gesture,
                    timestamp=t_frame_start,
                )

                # 4. Canvas Update
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

                # 5. Render
                if self.show_debug and observation is not None:
                    frame = self.detector.draw_landmarks(frame, observation)

                self.object_manager.draw_all(frame)
                if cursor_pos is not None:
                    self.object_manager.draw_cursor(frame, cursor_pos=cursor_pos)
                self.shape_menu.draw(frame, actual_h)

                self._draw_hud(
                    frame=frame,
                    stable_label=stable_gesture.label,
                    dynamic_label=dynamic_gesture,
                    source=static_pred.source,
                    confidence=static_pred.confidence,
                )

                cv2.imshow("Hand Gesture Controller", frame)
                self.performance.record_frame(t_frame_start)

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
        """Giải phóng tài nguyên camera và cửa sổ."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Hand Gesture HCI Controller"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["svm", "rules"],
        default="svm",
        help="Recognition mode: 'svm' (Machine Learning) or 'rules' (Geometric)",
    )
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=None, help="Target frame width")
    parser.add_argument("--height", type=int, default=None, help="Target frame height")
    parser.add_argument("--model", type=str, default=None, help="Path to SVM model .joblib file")
    parser.add_argument("--config", type=str, default=None, help="Path to runtime YAML config file")
    return parser.parse_args()


def main() -> None:
    """Điểm khởi chạy ứng dụng CLI."""
    setup_logging("INFO")
    args = parse_args()

    cfg = RuntimeConfig.from_yaml(args.config) if args.config else RuntimeConfig()
    if args.mode is not None:
        cfg.mode = args.mode
    if args.camera is not None:
        cfg.camera_index = args.camera
    if args.width is not None:
        cfg.target_width = args.width
    if args.height is not None:
        cfg.target_height = args.height
    if args.model is not None:
        cfg.model_path = args.model

    app = HandGestureApp(
        camera_index=cfg.camera_index,
        width=cfg.target_width,
        height=cfg.target_height,
        config=cfg,
        mode=cfg.mode,
    )
    app.run()


if __name__ == "__main__":
    main()
