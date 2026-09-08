"""Mô-đun chuyển đổi trạng thái cử chỉ ổn định thành sự kiện tương tác HCI (Event Mapper)."""

import time
from typing import Dict, Mapping, Optional, Set, Tuple, Union

from ..schemas import GestureEvent, HCIEvent, StableGesture

CONTINUOUS_EVENTS: Set[GestureEvent] = {
    GestureEvent.START_DRAG,
    GestureEvent.DRAG,
    GestureEvent.STOP_DRAG,
}

EDGE_TRIGGERED_EVENTS: Set[GestureEvent] = {
    GestureEvent.CHANGE_COLOR,
    GestureEvent.DELETE_OBJECT,
    GestureEvent.TOGGLE_CANVAS,
    GestureEvent.OPEN_MENU,
    GestureEvent.EMERGENCY_SOS,
}

DEFAULT_COOLDOWNS: Dict[GestureEvent, float] = {
    GestureEvent.CHANGE_COLOR: 0.5,
    GestureEvent.DELETE_OBJECT: 0.5,
    GestureEvent.TOGGLE_CANVAS: 0.8,
    GestureEvent.OPEN_MENU: 0.5,
    GestureEvent.EMERGENCY_SOS: 1.0,
}


class GestureEventMapper:
    """Bộ chuyển đổi nhãn cử chỉ tĩnh và chuyển động thành GestureEvent cho Canvas Manager.

    Hỗ trợ:
    - Continuous Events (START_DRAG / DRAG / STOP_DRAG).
    - Edge-Triggered Events (CHANGE_COLOR, DELETE_OBJECT, TOGGLE_CANVAS, OPEN_MENU).
    - Cơ chế Cooldown theo sự kiện.
    - Fail-Safe Hand-Loss: Khi mất dấu tay trong lúc đang kéo, bắt buộc phát sinh STOP_DRAG.
    """

    def __init__(
        self,
        cooldowns: Optional[Mapping[Union[GestureEvent, str], float]] = None,
    ) -> None:
        """Khởi tạo GestureEventMapper.

        Args:
            cooldowns: Từ điển thời gian chờ (giây) cho từng loại sự kiện edge-triggered.
        """
        self.is_dragging: bool = False
        self.prev_gesture: Optional[str] = None
        self.prev_motion: Optional[str] = None
        self.cooldowns = self._normalize_cooldowns(cooldowns)
        self.last_event_timestamps: Dict[GestureEvent, float] = {}

    @staticmethod
    def _normalize_cooldowns(
        cooldowns: Optional[Mapping[Union[GestureEvent, str], float]],
    ) -> Dict[GestureEvent, float]:
        """Chuẩn hóa cấu hình cooldown từ YAML hoặc enum về cùng một kiểu khóa."""
        if cooldowns is None:
            return DEFAULT_COOLDOWNS.copy()

        normalized: Dict[GestureEvent, float] = {}
        for raw_event, raw_seconds in cooldowns.items():
            if isinstance(raw_event, GestureEvent):
                event = raw_event
            else:
                try:
                    event = GestureEvent(raw_event)
                except ValueError:
                    try:
                        event = GestureEvent[raw_event.upper()]
                    except KeyError as exc:
                        raise ValueError(f"Sự kiện cooldown không hợp lệ: {raw_event!r}") from exc

            seconds = float(raw_seconds)
            if seconds < 0:
                raise ValueError(f"Cooldown của {event.value} không được âm.")
            normalized[event] = seconds

        return normalized

    def _can_trigger_edge_event(self, event: GestureEvent, now: float) -> bool:
        """Kiểm tra sự kiện edge-triggered có thỏa mãn thời gian cooldown hay không."""
        last_time = self.last_event_timestamps.get(event, -1e9)
        cooldown = self.cooldowns.get(event, 0.5)
        if now - last_time >= cooldown:
            self.last_event_timestamps[event] = now
            return True
        return False

    def handle_hand_loss(self, timestamp: Optional[float] = None) -> GestureEvent:
        """Xử lý trường hợp mất dấu bàn tay an toàn (Fail-Safe).

        Nếu đang trong quá trình kéo vật thể (is_dragging=True), bắt buộc trả về STOP_DRAG
        để vật thể không bị kẹt hay trôi dạt trên màn hình.
        """
        self.prev_gesture = None
        self.prev_motion = None
        if self.is_dragging:
            self.is_dragging = False
            return GestureEvent.STOP_DRAG
        return GestureEvent.NONE

    def map_gesture_to_event(
        self,
        gesture_label: str,
        motion_label: str = "Still",
        timestamp: Optional[float] = None,
    ) -> GestureEvent:
        """Ánh xạ nhãn cử chỉ tĩnh và chuyển động thành GestureEvent tương ứng.

        Args:
            gesture_label: Nhãn cử chỉ tĩnh đã làm mượt (Select, Options, Stop, Fist, Peace, NoAction, ...).
            motion_label: Nhãn cử chỉ động từ Dynamic FSM (On/Off, SOS, Still).
            timestamp: Thời điểm khung hình (giây monotonic).

        Returns:
            GestureEvent: Sự kiện điều khiển hệ thống.
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        # 1. Ưu tiên cử chỉ động (Motion FSM) - Edge-Triggered
        if motion_label == "On/Off":
            if self.is_dragging:
                self.is_dragging = False
            if self.prev_motion != "On/Off":
                self.prev_motion = motion_label
                self.prev_gesture = gesture_label
                if self._can_trigger_edge_event(GestureEvent.TOGGLE_CANVAS, now):
                    return GestureEvent.TOGGLE_CANVAS
            return GestureEvent.NONE

        if motion_label == "SOS":
            if self.is_dragging:
                self.is_dragging = False
            if self.prev_motion != "SOS":
                self.prev_motion = motion_label
                self.prev_gesture = gesture_label
                if self._can_trigger_edge_event(GestureEvent.EMERGENCY_SOS, now):
                    return GestureEvent.EMERGENCY_SOS
            return GestureEvent.NONE

        self.prev_motion = motion_label

        # 2. Cử chỉ kéo thả (Continuous Event)
        if gesture_label == "Select":
            self.prev_gesture = gesture_label
            if not self.is_dragging:
                self.is_dragging = True
                return GestureEvent.START_DRAG
            return GestureEvent.DRAG

        # Thoát trạng thái kéo thả khi không còn duy trì Select
        if self.is_dragging:
            self.is_dragging = False
            self.prev_gesture = gesture_label
            return GestureEvent.STOP_DRAG

        # 3. Ánh xạ các cử chỉ tĩnh tác vụ (Edge Triggered: Rising Edge)
        is_rising_edge = (gesture_label != self.prev_gesture)
        self.prev_gesture = gesture_label

        if is_rising_edge and gesture_label != "NoAction":
            if gesture_label == "Options":
                if self._can_trigger_edge_event(GestureEvent.CHANGE_COLOR, now):
                    return GestureEvent.CHANGE_COLOR

            elif gesture_label == "Stop":
                if self._can_trigger_edge_event(GestureEvent.DELETE_OBJECT, now):
                    return GestureEvent.DELETE_OBJECT

            elif gesture_label == "Peace":
                if self._can_trigger_edge_event(GestureEvent.OPEN_MENU, now):
                    return GestureEvent.OPEN_MENU

        return GestureEvent.NONE

    def process(
        self,
        stable_gesture: StableGesture,
        dynamic_gesture: str = "Still",
        cursor_pos: Optional[Tuple[int, int]] = None,
    ) -> HCIEvent:
        """Hàm chuẩn nhận vào StableGesture và trả về HCIEvent đầy đủ."""
        ev = self.map_gesture_to_event(
            gesture_label=stable_gesture.label,
            motion_label=dynamic_gesture,
            timestamp=stable_gesture.timestamp,
        )
        return HCIEvent(
            event_type=ev,
            timestamp=stable_gesture.timestamp,
            gesture=stable_gesture.label,
            cursor_pos=cursor_pos,
        )

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái event mapper khi khởi động lại."""
        self.is_dragging = False
        self.prev_gesture = None
        self.prev_motion = None
        self.last_event_timestamps.clear()
