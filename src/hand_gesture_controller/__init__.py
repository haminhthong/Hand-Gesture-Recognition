"""Hand Gesture Controller package.

Public classes are loaded lazily so lightweight logic modules can be imported
without requiring the OpenCV/MediaPipe GUI stack at import time.
"""

from importlib import import_module

__version__ = "1.0.0"

_EXPORTS = {
    "CursorFilter": (".temporal.cursor_filter", "CursorFilter"),
    "DEFAULT_THRESHOLDS": (".config", "DEFAULT_THRESHOLDS"),
    "DraggableObject": (".application.object_manager", "DraggableObject"),
    "DraggableObjectManager": (".application.object_manager", "DraggableObjectManager"),
    "DynamicGestureFSM": (".recognition.dynamic_fsm", "DynamicGestureFSM"),
    "GestureEvent": (".schemas", "GestureEvent"),
    "GestureEventMapper": (".events.event_mapper", "GestureEventMapper"),
    "GestureModelBundle": (".schemas", "GestureModelBundle"),
    "GestureStabilizer": (".temporal.gesture_stabilizer", "GestureStabilizer"),
    "GestureThresholds": (".config", "GestureThresholds"),
    "HCIEvent": (".schemas", "HCIEvent"),
    "HandDetector": (".perception.hand_detector", "HandDetector"),
    "HandObservation": (".schemas", "HandObservation"),
    "LandmarkPreprocessor": (".features.landmark_preprocessor", "LandmarkPreprocessor"),
    "MotionFeatures": (".features.motion_features", "MotionFeatures"),
    "PerformanceMonitor": (".telemetry.performance", "PerformanceMonitor"),
    "RuleStaticBaseline": (".recognition.rule_baseline", "RuleStaticBaseline"),
    "RuntimeConfig": (".config", "RuntimeConfig"),
    "ShapeMenu": (".application.shape_menu", "ShapeMenu"),
    "StableGesture": (".schemas", "StableGesture"),
    "StaticGesturePredictor": (".recognition.static_predictor", "StaticGesturePredictor"),
    "StaticPrediction": (".schemas", "StaticPrediction"),
    "TrainingConfig": (".config", "TrainingConfig"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    """Resolve public symbols on first access."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _EXPORTS[name]
    symbol = getattr(import_module(module_name, __name__), symbol_name)
    globals()[name] = symbol
    return symbol
