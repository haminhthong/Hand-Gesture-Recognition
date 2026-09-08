"""Mô-đun nhận diện cử chỉ tĩnh dựa trên luật hình học (Rule Engine) đóng vai trò Baseline & Fallback."""

import math
from typing import List, Optional

import numpy as np

from ..config import DEFAULT_THRESHOLDS, GestureThresholds
from ..schemas import HandObservation, StaticPrediction


class RuleStaticBaseline:
    """Bộ nhận diện cử chỉ tĩnh dựa trên các luật hình học góc và khoảng cách (Rule Baseline).

    Được giữ lại làm baseline chuẩn để so sánh thực nghiệm (experiment benchmark)
    và làm cơ chế fallback an toàn trong trường hợp model artifact chưa được nạp.
    """

    def __init__(self, thresholds: Optional[GestureThresholds] = None) -> None:
        """Khởi tạo RuleStaticBaseline.

        Args:
            thresholds: Ngưỡng hình học cấu hình. Nếu None dùng DEFAULT_THRESHOLDS.
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    @staticmethod
    def _calculate_distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _calculate_angle(p_tip: np.ndarray, p_joint: np.ndarray, p_base: np.ndarray) -> float:
        v1_x = p_tip[0] - p_joint[0]
        v1_y = p_tip[1] - p_joint[1]
        v2_x = p_base[0] - p_joint[0]
        v2_y = p_base[1] - p_joint[1]

        norm1 = math.hypot(v1_x, v1_y)
        norm2 = math.hypot(v2_x, v2_y)
        if norm1 <= 1e-6 or norm2 <= 1e-6:
            return 0.0

        dot = v1_x * v2_x + v1_y * v2_y
        cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
        return math.degrees(math.acos(cos_angle))

    def _is_finger_up(self, coords: np.ndarray, tip: int, pip: int, mcp: int) -> bool:
        wrist = coords[0]
        angle = self._calculate_angle(coords[tip], coords[pip], coords[mcp])
        d_tip = self._calculate_distance_2d(coords[tip], wrist)
        d_pip = self._calculate_distance_2d(coords[pip], wrist)
        return angle >= self.thresholds.min_finger_extension_angle_deg and d_tip > d_pip

    def _is_thumb_up(self, coords: np.ndarray, tip: int, ip: int) -> bool:
        wrist = coords[0]
        return self._calculate_distance_2d(coords[tip], wrist) > self._calculate_distance_2d(coords[ip], wrist)

    def _finger_states(self, coords: np.ndarray) -> List[bool]:
        return [
            self._is_thumb_up(coords, 4, 3),
            self._is_finger_up(coords, 8, 6, 5),
            self._is_finger_up(coords, 12, 10, 9),
            self._is_finger_up(coords, 16, 14, 13),
            self._is_finger_up(coords, 20, 18, 17),
        ]

    def predict_observation(self, observation: HandObservation) -> StaticPrediction:
        """Dự đoán cử chỉ tĩnh từ HandObservation bằng các luật hình học.

        Args:
            observation: HandObservation từ perception layer.

        Returns:
            StaticPrediction: Dự đoán chứa nhãn cử chỉ và điểm khớp heuristic rule_score.
        """
        coords = observation.landmarks
        palm_sz = max(observation.palm_size, 1e-6)

        thumb_tip = coords[4]
        index_tip = coords[8]
        middle_tip = coords[12]

        finger_states = self._finger_states(coords)
        fingers_up = sum(finger_states)
        thumb_up, index_up, middle_up, _, _ = finger_states

        label = "NoAction"
        score = 0.50

        norm_d_thumb_index = self._calculate_distance_2d(thumb_tip, index_tip) / palm_sz
        norm_d_middle_index = self._calculate_distance_2d(middle_tip, index_tip) / palm_sz

        # Bảng ưu tiên (Precedence Order):
        # 1. Fist: Không có ngón nào duỗi
        if fingers_up == 0:
            label = "Fist"
            score = 0.95
        # 2. Stop: Cả 5 ngón đều duỗi
        elif fingers_up == 5:
            label = "Stop"
            score = 0.95
        # 3. Peace: Ngón trỏ & giữa duỗi, ngón cái gập
        elif fingers_up == 2 and index_up and middle_up and not thumb_up and norm_d_thumb_index >= self.thresholds.select_distance:
            label = "Peace"
            score = 0.90
        # 4. Cụm Pinch cử chỉ chụm ngón
        elif norm_d_thumb_index < self.thresholds.select_distance:
            pinch_margin = max(0.0, min(1.0, 1.0 - norm_d_thumb_index / max(self.thresholds.select_distance, 1e-6)))
            if norm_d_middle_index < self.thresholds.options_distance:
                label = "Options"
                opt_margin = max(0.0, min(1.0, 1.0 - norm_d_middle_index / max(self.thresholds.options_distance, 1e-6)))
                score = round(0.70 + 0.25 * ((pinch_margin + opt_margin) / 2.0), 2)
            elif norm_d_middle_index > 0.4 and fingers_up == 2:
                label = "Select"
                score = round(0.70 + 0.25 * pinch_margin, 2)
            else:
                label = "NoAction"
                score = 0.55
        else:
            label = "NoAction"
            score = 0.50

        return StaticPrediction(
            label=label,
            confidence=score,
            probabilities={label: score},
            rejected=False,
            source="rule_baseline",
        )
