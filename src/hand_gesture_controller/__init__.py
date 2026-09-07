"""Hand Gesture Controller Package - Real-Time Landmark-Based Hand Gesture HCI System."""

from .application.object_manager import DraggableObject, DraggableObjectManager
from .application.shape_menu import ShapeMenu
from .config import DEFAULT_THRESHOLDS, GestureThresholds, RuntimeConfig, TrainingConfig
from .events.event_mapper import GestureEventMapper
from .features.landmark_preprocessor import LandmarkPreprocessor
from .features.motion_features import MotionFeatures
from .perception.hand_detector import HandDetector
from .recognition.dynamic_fsm import DynamicGestureFSM
from .recognition.rule_baseline import RuleStaticBaseline
from .recognition.static_predictor import StaticGesturePredictor
from .schemas import (
    GestureEvent,
    GestureModelBundle,
    HandObservation,
    HCIEvent,
    StableGesture,
    StaticPrediction,
)
from .telemetry.performance import PerformanceMonitor
from .temporal.cursor_filter import CursorFilter
from .temporal.gesture_stabilizer import GestureStabilizer

__version__ = "1.0.0"

__all__ = [
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
    "HCIEvent",
    "HandDetector",
    "HandObservation",
    "LandmarkPreprocessor",
    "MotionFeatures",
    "PerformanceMonitor",
    "RuleStaticBaseline",
    "RuntimeConfig",
    "ShapeMenu",
    "StableGesture",
    "StaticGesturePredictor",
    "StaticPrediction",
    "TrainingConfig",
]
