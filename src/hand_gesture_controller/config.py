"""Mô-đun cấu hình chứa các tham số hình học, runtime và training của hệ thống HCI."""

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass(frozen=True)
class GestureThresholds:
    """Dataclass tập trung các ngưỡng khoảng cách đã chuẩn hóa và thời gian chờ FSM."""

    pinch_distance: float = 0.35
    select_distance: float = 0.60
    options_distance: float = 0.35
    movement_distance: float = 0.12
    sos_timeout_seconds: float = 1.5
    wave_timeout_seconds: float = 2.0
    wave_direction_changes: int = 3
    min_finger_extension_angle_deg: float = 140.0


DEFAULT_THRESHOLDS = GestureThresholds()


@dataclass
class RuntimeConfig:
    """Cấu hình thực thi cho toàn bộ ứng dụng Hand Gesture HCI Controller."""

    # Camera & Capture
    camera_index: int = 0
    target_width: int = 640
    target_height: int = 480
    mirror_camera: bool = True

    # Perception
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    max_num_hands: int = 1

    # Recognition (ML & Fallback)
    model_path: str = "models/static_gesture_svm_v1.joblib"
    fallback_to_rules: bool = True
    default_accept_threshold: float = 0.60
    accept_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "Select": 0.65,
            "Options": 0.70,
            "Stop": 0.75,
            "Peace": 0.65,
            "Fist": 0.60,
            "NoAction": 0.50,
        }
    )

    # Temporal Stabilization
    activation_dwell_ms: float = 120.0
    release_dwell_ms: float = 100.0
    history_horizon_ms: float = 250.0
    hand_loss_timeout_ms: float = 150.0

    # Dynamic gestures
    on_off_timeout_seconds: float = 1.5
    sos_timeout_seconds: float = 1.5
    enable_experimental_gestures: bool = False

    # Cursor EMA Filter
    cursor_tau: float = 0.08  # Thời hằng tau cho time-aware EMA

    # Event Cooldowns (seconds)
    cooldowns: Dict[str, float] = field(
        default_factory=lambda: {
            "change_color": 0.5,
            "delete_object": 0.5,
            "toggle_canvas": 0.8,
            "open_menu": 0.5,
            "emergency_sos": 1.0,
        }
    )

    # UI & Rendering
    show_debug_hud: bool = True
    benchmark_output: Optional[str] = None

    def __post_init__(self) -> None:
        """Kiểm tra cấu hình sớm để lỗi không trôi tới vòng lặp camera."""
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
        if not 0.0 <= self.default_accept_threshold <= 1.0:
            raise ValueError("default_accept_threshold phải nằm trong [0, 1].")
        if self.cursor_tau <= 0:
            raise ValueError("cursor_tau phải lớn hơn 0.")
        if self.on_off_timeout_seconds <= 0 or self.sos_timeout_seconds <= 0:
            raise ValueError("Các timeout của dynamic FSM phải lớn hơn 0.")
        if any(seconds < 0 for seconds in self.cooldowns.values()):
            raise ValueError("Các giá trị cooldown không được âm.")

    @classmethod
    def from_yaml(cls, path: str) -> "RuntimeConfig":
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


@dataclass
class TrainingConfig:
    """Cấu hình huấn luyện mô hình Calibrated RBF-SVM."""

    dataset_path: str = "data/raw/landmarks_dataset.csv"
    manifest_path: str = "data/processed/manifest.json"
    splits_path: str = "data/processed/splits.json"
    model_output_path: str = "models/static_gesture_svm_v1.joblib"

    # Preprocessing
    mirror_left_hand: bool = True
    normalize_rotation: bool = True

    # Target Vocabulary
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

    # Threshold Tuning Targets
    target_precision_actionable: float = 0.90
    max_false_action_rate: float = 0.05

    def __post_init__(self) -> None:
        """Kiểm tra các tham số huấn luyện trước khi đọc dataset."""
        if self.cv_splits < 2:
            raise ValueError("cv_splits phải lớn hơn hoặc bằng 2.")
        if not self.labels:
            raise ValueError("labels không được rỗng.")
        if not 0.0 <= self.target_precision_actionable <= 1.0:
            raise ValueError("target_precision_actionable phải nằm trong [0, 1].")
        if not 0.0 <= self.max_false_action_rate <= 1.0:
            raise ValueError("max_false_action_rate phải nằm trong [0, 1].")

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
