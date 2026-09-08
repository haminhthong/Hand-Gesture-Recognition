"""Mô-đun máy trạng thái hữu hạn (Dynamic FSM) nhận diện cử chỉ chuỗi động On/Off độc lập FPS."""

import math
from typing import Optional, Tuple

import numpy as np

from ..schemas import HandObservation


class DynamicGestureFSM:
    """Máy trạng thái hữu hạn (Finite State Machine) nhận diện chuỗi cử chỉ động On/Off.

    Quy trình FSM của On/Off:
    - Trạng thái bắt đầu (START): Ngón trỏ và ngón giữa duỗi, ngón giữa chụm vào ngón cái.
    - Chuyển tiếp (TRANSITION): Ngón giữa gập lại trong khoảng thời gian quy định.
    - Trạng thái hoàn tất (COMPLETED): Chỉ còn ngón trỏ duỗi thẳng. Phát sinh sự kiện On/Off một lần duy nhất.
    """

    def __init__(
        self,
        on_off_timeout_seconds: float = 1.5,
        sos_timeout_seconds: float = 1.5,
        enable_experimental_gestures: bool = False,
    ) -> None:
        """Khởi tạo DynamicGestureFSM.

        Args:
            on_off_timeout_seconds: Thời gian tối đa (giây) giữa tư thế Start và End của On/Off.
            sos_timeout_seconds: Thời gian tối đa cho chuỗi SOS (thử nghiệm).
            enable_experimental_gestures: Bật cử chỉ thử nghiệm (SOS, Wave).
        """
        self.on_off_timeout_seconds = on_off_timeout_seconds
        self.sos_timeout_seconds = sos_timeout_seconds
        self.enable_experimental_gestures = enable_experimental_gestures

        self.current_fsm_state: str = "IDLE"
        self.state_start_time: float = 0.0
        self.last_observation_time: float = 0.0

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái FSM về IDLE khi mất dấu bàn tay."""
        self.current_fsm_state = "IDLE"
        self.state_start_time = 0.0
        self.last_observation_time = 0.0

    @staticmethod
    def _is_finger_extended(
        landmarks: np.ndarray, tip_idx: int, pip_idx: int, mcp_idx: int
    ) -> bool:
        """Kiểm tra ngón tay duỗi dựa trên khoảng cách tới cổ tay và khớp MCP."""
        wrist = landmarks[0]
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]

        d_tip = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
        d_pip = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
        return d_tip > d_pip

    def _get_finger_states(self, landmarks: np.ndarray) -> Tuple[bool, bool, bool, bool, bool]:
        """Trả về trạng thái duỗi/gập của [Cái, Trỏ, Giữa, Áp Út, Út]."""
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        d_thumb_tip = math.hypot(thumb_tip[0] - wrist[0], thumb_tip[1] - wrist[1])
        d_thumb_ip = math.hypot(thumb_ip[0] - wrist[0], thumb_ip[1] - wrist[1])
        thumb_up = d_thumb_tip > d_thumb_ip

        index_up = self._is_finger_extended(landmarks, 8, 6, 5)
        middle_up = self._is_finger_extended(landmarks, 12, 10, 9)
        ring_up = self._is_finger_extended(landmarks, 16, 14, 13)
        pinky_up = self._is_finger_extended(landmarks, 20, 18, 17)

        return (thumb_up, index_up, middle_up, ring_up, pinky_up)

    def _is_on_off_start(
        self, landmarks: np.ndarray, palm_size: float, finger_states: Tuple[bool, bool, bool, bool, bool]
    ) -> bool:
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        d8_4 = math.hypot(landmarks[8, 0] - landmarks[4, 0], landmarks[8, 1] - landmarks[4, 1]) / palm_size
        d12_4 = math.hypot(landmarks[12, 0] - landmarks[4, 0], landmarks[12, 1] - landmarks[4, 1]) / palm_size
        return (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
            and d8_4 >= 0.50
            and d12_4 < 0.35
        )

    def _is_on_off_end(
        self, landmarks: np.ndarray, palm_size: float, finger_states: Tuple[bool, bool, bool, bool, bool]
    ) -> bool:
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        d8_4 = math.hypot(landmarks[8, 0] - landmarks[4, 0], landmarks[8, 1] - landmarks[4, 1]) / palm_size
        d12_4 = math.hypot(landmarks[12, 0] - landmarks[4, 0], landmarks[12, 1] - landmarks[4, 1]) / palm_size
        return (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
            and d8_4 >= 0.50
            and d12_4 >= 0.35
        )

    def update(self, observation: Optional[HandObservation]) -> str:
        """Cập nhật máy trạng thái FSM với quan sát bàn tay mới nhất.

        Args:
            observation: HandObservation từ perception layer, hoặc None nếu mất dấu bàn tay.

        Returns:
            str: Tên cử chỉ động được hoàn tất ("On/Off", "SOS", "Still").
        """
        if observation is None:
            self.reset()
            return "Still"

        now = observation.timestamp
        self.last_observation_time = now
        landmarks = observation.landmarks
        palm_size = max(observation.palm_size, 1e-6)
        finger_states = self._get_finger_states(landmarks)
        fingers_up = sum(finger_states)
        thumb_up = finger_states[0]

        # Kiểm tra timeout cho trạng thái hiện tại
        if self.current_fsm_state != "IDLE":
            elapsed = now - self.state_start_time
            timeout = (
                self.sos_timeout_seconds
                if self.current_fsm_state == "START_SOS"
                else self.on_off_timeout_seconds
            )
            if elapsed > timeout:
                self.reset()

        # FSM Transition Logic
        # 1. Nhận diện chuỗi On/Off
        if self._is_on_off_start(landmarks, palm_size, finger_states):
            self.current_fsm_state = "START_ONOFF"
            self.state_start_time = now
            return "Still"

        if self.current_fsm_state == "START_ONOFF":
            if self._is_on_off_end(landmarks, palm_size, finger_states):
                self.reset()
                return "On/Off"

        # 2. Nhận diện SOS (chỉ kích hoạt nếu bật experimental)
        if self.enable_experimental_gestures:
            if not thumb_up and fingers_up == 4:
                if self.current_fsm_state != "START_SOS":
                    self.current_fsm_state = "START_SOS"
                    self.state_start_time = now
            elif fingers_up == 0 and self.current_fsm_state == "START_SOS":
                self.reset()
                return "SOS"

        return "Still"
