"""Tests cho StaticGestureClassifier và cơ chế Confidence Threshold."""

import numpy as np
import pytest

from hand_gesture_controller.classifier import StaticGestureClassifier
from hand_gesture_controller.schemas import GestureModelBundle


class MockModel:
    def __init__(self, classes, prob_return):
        self.classes_ = np.array(classes)
        self.prob_return = np.array(prob_return)

    def predict_proba(self, X):
        return np.tile(self.prob_return, (len(X), 1))


def test_classifier_shape_and_nan_validation():
    classifier = StaticGestureClassifier()
    with pytest.raises(ValueError, match="Vector đặc trưng phải có kích thước"):
        classifier.predict_features(np.zeros((62,), dtype=np.float32))

    invalid_feat = np.full((63,), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="Vector đặc trưng chứa giá trị không hợp lệ"):
        classifier.predict_features(invalid_feat)


def test_classifier_uninitialized():
    classifier = StaticGestureClassifier()
    pred = classifier.predict_features(np.zeros((63,), dtype=np.float32))
    assert pred.label == "NoAction"
    assert pred.rejected is True


def test_classifier_confidence_threshold_acceptance():
    classes = ["Fist", "Select", "Options", "Stop", "Peace", "NoAction"]
    # Xác suất Peace cao (0.85) > ngưỡng 0.65
    probs = [0.02, 0.03, 0.05, 0.03, 0.85, 0.02]
    mock = MockModel(classes, probs)
    bundle = GestureModelBundle(
        model=mock,
        scaler=None,
        labels=classes,
        threshold=0.65,
    )

    classifier = StaticGestureClassifier(bundle=bundle, confidence_threshold=0.65)
    pred = classifier.predict_features(np.zeros((63,), dtype=np.float32))

    assert pred.label == "Peace"
    assert pred.confidence == pytest.approx(0.85)
    assert pred.rejected is False


def test_classifier_confidence_threshold_rejection():
    classes = ["Fist", "Select", "Options", "Stop", "Peace", "NoAction"]
    # Xác suất cao nhất là 0.55 < ngưỡng 0.65 -> bị reject về NoAction
    probs = [0.10, 0.10, 0.10, 0.55, 0.10, 0.05]
    mock = MockModel(classes, probs)
    bundle = GestureModelBundle(
        model=mock,
        scaler=None,
        labels=classes,
        threshold=0.65,
    )

    classifier = StaticGestureClassifier(bundle=bundle, confidence_threshold=0.65)
    pred = classifier.predict_features(np.zeros((63,), dtype=np.float32))

    assert pred.label == "NoAction"
    assert pred.confidence == pytest.approx(0.55)
    assert pred.rejected is True
