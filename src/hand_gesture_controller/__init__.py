"""Hand Gesture Controller package.

Cung cấp các thành phần cho luồng nhận diện cử chỉ bàn tay HCI:
Perception -> Preprocessing -> Classification/Rules -> Stabilization -> Event Mapping -> Canvas.
"""

from .canvas import (
    CircleObject,
    CursorFilter,
    DraggableObject,
    DraggableObjectManager,
    RectangleObject,
    ShapeMenu,
    StarObject,
    TriangleObject,
)
from .classifier import StaticGestureClassifier, StaticGesturePredictor
from .config import DEFAULT_THRESHOLDS, GestureThresholds, RuntimeConfig, TrainingConfig
from .dynamic_gesture import DynamicGestureFSM
from .event_mapper import GestureEventMapper
from .hand_detector import HandDetector
from .preprocessing import LandmarkPreprocessor
from .rule_baseline import RuleStaticBaseline
from .schemas import (
    GestureEvent,
    GestureModelBundle,
    HandObservation,
    StableGesture,
    StaticPrediction,
)
from .stabilizer import GestureStabilizer
from .telemetry import PerformanceMonitor

__version__ = "1.0.0"

__all__ = [
    "CircleObject",
    "CursorFilter",
    "DEFAULT_THRESHOLDS",
    "DraggableObject",
    "DraggableObjectManager",
    "DynamicGestureFSM",
    "GestureEvent",
    "GestureEventMapper",
    "GestureModelBundle",
    "GestureStabilizer",
    "GestureThresholds",
    "HandDetector",
    "HandObservation",
    "LandmarkPreprocessor",
    "PerformanceMonitor",
    "RectangleObject",
    "RuleStaticBaseline",
    "RuntimeConfig",
    "ShapeMenu",
    "StableGesture",
    "StarObject",
    "StaticGestureClassifier",
    "StaticGesturePredictor",
    "StaticPrediction",
    "TrainingConfig",
    "TriangleObject",
    "__version__",
]
