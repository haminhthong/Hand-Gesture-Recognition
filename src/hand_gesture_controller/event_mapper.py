"""Mô-đun ánh xạ trạng thái cử chỉ ổn định thành sự kiện tương tác HCI (Event Mapper)."""

import time
from typing import Dict, Mapping, Optional, Set, Union

from .schemas import GestureEvent

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
}

DEFAULT_COOLDOWNS: Dict[GestureEvent, float] = {
    GestureEvent.CHANGE_COLOR: 0.5,
    GestureEvent.DELETE_OBJECT: 0.5,
    GestureEvent.TOGGLE_CANVAS: 0.8,
    GestureEvent.OPEN_MENU: 0.5,
}


class GestureEventMapper:
    """Bộ chuyển đổi nhãn cử chỉ thành GestureEvent tương tác với Canvas.

    Thiết kế giải quyết bài toán HCI thực tế:
    - Continuous Tracking: Kéo thả liên tục (START_DRAG -> DRAG -> STOP_DRAG).
    - Rising Edge: Chỉ kích hoạt hành động một lần khi cử chỉ mới xuất hiện (Options, Stop, Peace).
    - Cooldown: Ngăn chặn kích hoạt lặp ngoài ý muốn.
    - Hand-loss Failsafe: Khi mất dấu bàn tay trong lúc đang kéo, lập tức phát sinh STOP_DRAG.
    """

    def __init__(
        self,
        cooldowns: Optional[Mapping[Union[GestureEvent, str], float]] = None,
    ) -> None:
        self.is_dragging: bool = False
        self.prev_gesture: Optional[str] = None
        self.prev_motion: Optional[str] = None
        self.cooldowns = self._normalize_cooldowns(cooldowns)
        self.last_event_timestamps: Dict[GestureEvent, float] = {}

    @staticmethod
    def _normalize_cooldowns(
        cooldowns: Optional[Mapping[Union[GestureEvent, str], float]],
    ) -> Dict[GestureEvent, float]:
        """Chuẩn hóa cấu hình cooldown từ chuỗi/YAML về GestureEvent."""
        if cooldowns is None:
            return DEFAULT_COOLDOWNS.copy()

        normalized: Dict[GestureEvent, float] = DEFAULT_COOLDOWNS.copy()
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

        Nếu đang kéo vật thể, bắt buộc phát sinh STOP_DRAG để vật thể không bị kẹt hay trôi dạt.
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
        """Ánh xạ nhãn cử chỉ tĩnh và chuyển động thành GestureEvent tương ứng."""
        now = timestamp if timestamp is not None else time.perf_counter()

        # 1. Cử chỉ động On/Off (Edge-Triggered)
        if motion_label == "On/Off":
            if self.is_dragging:
                self.is_dragging = False
            if self.prev_motion != "On/Off":
                self.prev_motion = motion_label
                self.prev_gesture = gesture_label
                if self._can_trigger_edge_event(GestureEvent.TOGGLE_CANVAS, now):
                    return GestureEvent.TOGGLE_CANVAS
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

        # 3. Ánh xạ các cử chỉ tĩnh tác vụ (Rising Edge)
        is_rising_edge = gesture_label != self.prev_gesture
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

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái bộ ánh xạ sự kiện."""
        self.is_dragging = False
        self.prev_gesture = None
        self.prev_motion = None
        self.last_event_timestamps.clear()
