"""Mô-đun cấu hình chứa các tham số hình học, runtime và huấn luyện của hệ thống HCI."""

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import yaml


@dataclass(frozen=True)
class GestureThresholds:
    """Các ngưỡng khoảng cách hình học chuẩn hóa theo kích thước lòng bàn tay."""

    pinch_distance: float = 0.35
    select_distance: float = 0.60
    options_distance: float = 0.35
    min_finger_extension_angle_deg: float = 140.0


DEFAULT_THRESHOLDS = GestureThresholds()


@dataclass
class RuntimeConfig:
    """Cấu hình thực thi cho ứng dụng Hand Gesture Controller."""

    # Chế độ nhận diện ("svm" hoặc "rules")
    mode: str = "svm"

    # Camera & Capture
    camera_index: int = 0
    target_width: int = 640
    target_height: int = 480
    mirror_camera: bool = True

    # Perception
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    max_num_hands: int = 1

    # Static Classifier
    model_path: str = "models/static_gesture_svm.joblib"
    confidence_threshold: float = 0.65

    # Temporal Stabilization
    activation_dwell_ms: float = 120.0
    release_dwell_ms: float = 100.0
    history_horizon_ms: float = 250.0
    hand_loss_timeout_ms: float = 150.0

    # Dynamic gestures
    on_off_timeout_seconds: float = 1.5

    # Cursor Filter
    cursor_tau: float = 0.08

    # Event Cooldowns (seconds)
    cooldowns: Dict[str, float] = field(
        default_factory=lambda: {
            "change_color": 0.5,
            "delete_object": 0.5,
            "toggle_canvas": 0.8,
            "open_menu": 0.5,
        }
    )

    # UI HUD
    show_debug_hud: bool = True

    def __post_init__(self) -> None:
        """Kiểm tra cấu hình hợp lệ."""
        if self.mode not in ("svm", "rules"):
            raise ValueError(f"Chế độ (--mode) phải là 'svm' hoặc 'rules', nhận được {self.mode!r}")
        if self.camera_index < 0:
            raise ValueError("camera_index phải lớn hơn hoặc bằng 0.")
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError("target_width và target_height phải lớn hơn 0.")
        if not 0.0 < self.min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence phải nằm trong (0, 1].")
        if not 0.0 < self.min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence phải nằm trong (0, 1].")
        if self.max_num_hands < 1:
            raise ValueError("max_num_hands phải lớn hơn 0.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold phải nằm trong [0, 1].")
        if self.cursor_tau <= 0:
            raise ValueError("cursor_tau phải lớn hơn 0.")
        if self.on_off_timeout_seconds <= 0:
            raise ValueError("on_off_timeout_seconds phải lớn hơn 0.")
        if any(seconds < 0 for seconds in self.cooldowns.values()):
            raise ValueError("Các giá trị cooldown không được âm.")

    @property
    def default_accept_threshold(self) -> float:
        """Thuộc tính tương thích ngược cho confidence_threshold."""
        return self.confidence_threshold

    @property
    def fallback_to_rules(self) -> bool:
        """Thuộc tính tương thích ngược."""
        return self.mode == "rules"

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
        """Nạp cấu hình từ tệp YAML."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Hỗ trợ migrate các trường cũ
        if "default_accept_threshold" in data and "confidence_threshold" not in data:
            data["confidence_threshold"] = data.pop("default_accept_threshold")
        if "accept_thresholds" in data:
            data.pop("accept_thresholds")
        if "cooldowns" in data and "emergency_sos" in data["cooldowns"]:
            data["cooldowns"].pop("emergency_sos")
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Lưu cấu hình ra tệp YAML."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, allow_unicode=True)


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện mô hình SVM cử chỉ tĩnh."""

    dataset_path: str = "data/raw/landmarks_dataset.csv"
    splits_path: str = "data/processed/splits.json"
    model_output_path: str = "models/static_gesture_svm.joblib"

    # Preprocessing
    mirror_left_hand: bool = True
    normalize_rotation: bool = True

    # Vocabulary
    labels: List[str] = field(
        default_factory=lambda: [
            "Fist",
            "Select",
            "Options",
            "Stop",
            "Peace",
            "NoAction",
        ]
    )

    # Grid Search Params
    cv_splits: int = 5
    c_candidates: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 50.0])
    gamma_candidates: List[str] = field(default_factory=lambda: ["scale", "auto", "0.01", "0.1"])

    confidence_threshold: float = 0.65

    def __post_init__(self) -> None:
        """Kiểm tra các tham số huấn luyện."""
        if self.cv_splits < 2:
            raise ValueError("cv_splits phải lớn hơn hoặc bằng 2.")
        if not self.labels:
            raise ValueError("labels không được rỗng.")

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Nạp cấu hình từ tệp YAML."""
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Lưu cấu hình ra tệp YAML."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, allow_unicode=True)
