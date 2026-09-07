"""Mô-đun tương thích ngược cho GestureEventMapper (chuyển tiếp tới package events)."""

from .events.event_mapper import (
    CONTINUOUS_EVENTS,
    DEFAULT_COOLDOWNS,
    EDGE_TRIGGERED_EVENTS,
    GestureEvent,
    GestureEventMapper,
)

__all__ = [
    "CONTINUOUS_EVENTS",
    "DEFAULT_COOLDOWNS",
    "EDGE_TRIGGERED_EVENTS",
    "GestureEvent",
    "GestureEventMapper",
]
