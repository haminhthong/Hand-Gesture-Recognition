"""Tests kiểm tra validation của CLI và RuntimeConfig."""

import pytest

from hand_gesture_controller.app import HandGestureApp
from hand_gesture_controller.config import RuntimeConfig
from hand_gesture_controller.dynamic_gesture import DynamicGestureFSM


def test_app_invalid_parameters():
    with pytest.raises(ValueError, match="Kích thước khung hình phải lớn hơn 0"):
        HandGestureApp(width=-100)

    with pytest.raises(ValueError, match="Chỉ số camera phải lớn hơn hoặc bằng 0"):
        HandGestureApp(camera_index=-1)


def test_runtime_config_validation():
    with pytest.raises(ValueError, match="Chế độ .* phải là 'svm' hoặc 'rules'"):
        RuntimeConfig(mode="invalid_mode")

    with pytest.raises(ValueError, match="confidence_threshold phải nằm trong"):
        RuntimeConfig(confidence_threshold=1.5)

    with pytest.raises(ValueError, match="on_off_timeout_seconds phải lớn hơn 0"):
        RuntimeConfig(on_off_timeout_seconds=-1.0)


def test_dynamic_fsm_validation():
    with pytest.raises(ValueError, match="Timeout của dynamic FSM phải lớn hơn 0"):
        DynamicGestureFSM(timeout_seconds=-0.5)


def test_svm_mode_missing_model_raises_error():
    cfg = RuntimeConfig(mode="svm", model_path="non_existent_model_123.joblib")
    with pytest.raises(FileNotFoundError, match="Không tìm thấy mô hình SVM"):
        HandGestureApp(config=cfg)


def test_rules_mode_works_without_model():
    cfg = RuntimeConfig(mode="rules")
    app = HandGestureApp(config=cfg)
    assert app.mode == "rules"
    assert app.classifier is None
