"""Mô-đun HandDetector đóng gói MediaPipe Hands và trích xuất HandObservation."""

import logging
import math
import time
from typing import Any, Optional

import cv2
import numpy as np

from .schemas import HandObservation

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class HandDetector:
    """Bộ cảm nhận thị giác đóng gói MediaPipe Hands trả về schema HandObservation chuẩn."""

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 1,
        detectionCon: float = 0.7,
        trackCon: float = 0.6,
    ) -> None:
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.results: Optional[Any] = None

        if mp is None:
            logger.warning("MediaPipe chưa được cài đặt hoặc không khả dụng trong môi trường này.")
            self.mp_hands = None
            self.hands = None
            self.mp_draw = None
            return

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def close(self) -> None:
        """Giải phóng tài nguyên MediaPipe Hands."""
        if hasattr(self, "hands") and self.hands:
            try:
                self.hands.close()
            except Exception as e:
                logger.warning("Lỗi giải phóng MediaPipe Hands: %s", e)

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def process(
        self, frame: Optional[np.ndarray], timestamp: Optional[float] = None
    ) -> Optional[HandObservation]:
        """Xử lý khung hình OpenCV BGR và trích xuất HandObservation."""
        if frame is None or self.hands is None:
            return None

        h, w = frame.shape[:2]
        now = timestamp if timestamp is not None else time.perf_counter()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if not self.results or not self.results.multi_hand_landmarks:
            return None

        primary_landmarks = self.results.multi_hand_landmarks[0]
        pts = primary_landmarks.landmark
        if len(pts) < 21:
            return None

        coords = np.array([[lm.x, lm.y, lm.z] for lm in pts[:21]], dtype=np.float32)

        handedness = "Right"
        handedness_score = 1.0
        if self.results.multi_handedness:
            cls_info = self.results.multi_handedness[0].classification[0]
            handedness = cls_info.label
            handedness_score = float(cls_info.score)

        palm_size = float(math.hypot(coords[9, 0] - coords[0, 0], coords[9, 1] - coords[0, 1]))
        center_x = float(np.mean(coords[:, 0]))
        center_y = float(np.mean(coords[:, 1]))

        return HandObservation(
            landmarks=coords,
            handedness=handedness,
            handedness_score=handedness_score,
            timestamp=now,
            frame_width=w,
            frame_height=h,
            palm_size=palm_size,
            hand_center=(center_x, center_y),
        )

    def draw_landmarks(
        self,
        img: np.ndarray,
        observation: Optional[HandObservation] = None,
    ) -> np.ndarray:
        """Vẽ khung xương bàn tay lên ảnh."""
        if img is None:
            return img

        if self.results and self.results.multi_hand_landmarks and self.mp_draw and self.mp_hands:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return img
