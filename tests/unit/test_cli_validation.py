"""Unit tests cho tham số dòng lệnh CLI và kiểm tra hợp lệ ứng dụng."""

import pytest

from hand_gesture_controller.app import HandGestureApp
from tools.collect_landmarks import collect_landmarks


def test_app_invalid_parameters():
    with pytest.raises(ValueError, match="Chiều rộng khung hình"):
        HandGestureApp(width=-100)

    with pytest.raises(ValueError, match="Chiều cao khung hình"):
        HandGestureApp(height=0)

    with pytest.raises(ValueError, match="Chỉ số camera"):
        HandGestureApp(camera_index=-1)


def test_collect_landmarks_validation():
    with pytest.raises(ValueError, match="Mã người tham gia"):
        collect_landmarks(subject_id="", session_id="ses_01", label="Fist")

    with pytest.raises(ValueError, match="Mã phiên"):
        collect_landmarks(subject_id="sub_01", session_id=" ", label="Fist")

    with pytest.raises(ValueError, match="Số lượng mẫu"):
        collect_landmarks(subject_id="sub_01", session_id="ses_01", label="Fist", max_samples=0)


def test_train_baseline_validation():
    from tools.train_baseline import evaluate_baselines

    with pytest.raises(FileNotFoundError, match="Không tìm thấy file dataset"):
        evaluate_baselines("data/raw/non_existent_dataset.csv")

