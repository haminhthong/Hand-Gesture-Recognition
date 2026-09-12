"""Định nghĩa các cấu trúc dữ liệu cốt lõi cho hệ thống nhận diện cử chỉ HCI."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


class GestureEvent(str, Enum):
    """Các sự kiện tương tác phát sinh tới tầng ứng dụng."""

    NONE = "none"
    START_DRAG = "start_drag"
    DRAG = "drag"
    STOP_DRAG = "stop_drag"
    CHANGE_COLOR = "change_color"
    DELETE_OBJECT = "delete_object"
    TOGGLE_CANVAS = "toggle_canvas"
    OPEN_MENU = "open_menu"


@dataclass(frozen=True)
class HandObservation:
    """Dữ liệu cảm nhận từ khung hình camera sau khi xử lý qua MediaPipe Hands."""

    landmarks: np.ndarray  # Mảng float32 hình dạng (21, 3) đại diện (x, y, z)
    handedness: str  # "Left" hoặc "Right"
    handedness_score: float  # Độ tin cậy nhận diện tay [0.0, 1.0]
    timestamp: float  # Thời điểm monotonic (time.perf_counter)
    frame_width: int  # Chiều rộng khung hình
    frame_height: int  # Chiều cao khung hình
    palm_size: float  # Khoảng cách 2D tham chiếu (cổ tay -> middle_mcp)
    hand_center: Tuple[float, float]  # Tọa độ tâm bàn tay chuẩn hóa [0.0, 1.0]

    def __post_init__(self) -> None:
        if self.landmarks.shape != (21, 3):
            raise ValueError(f"Landmarks phải có kích thước (21, 3), nhận được {self.landmarks.shape}")


@dataclass(frozen=True)
class StaticPrediction:
    """Kết quả phân loại cử chỉ tĩnh từ mô hình ML hoặc Rule Baseline."""

    label: str  # Nhãn cử chỉ ("Fist", "Select", "Options", "Stop", "Peace", "NoAction")
    confidence: float  # Độ tin cậy hoặc xác suất [0.0, 1.0]
    probabilities: Dict[str, float] = field(default_factory=dict)
    rejected: bool = False  # True nếu confidence < confidence_threshold
    source: str = "svm"


@dataclass(frozen=True)
class StableGesture:
    """Cử chỉ sau khi đã qua bộ lọc ổn định thời gian (Temporal Stabilizer)."""

    label: str
    confidence: float
    dwell_time_ms: float
    timestamp: float


@dataclass
class GestureModelBundle:
    """Gói lưu trữ mô hình cử chỉ tĩnh gồm model, scaler, danh sách nhãn và cấu hình."""

    model: Any
    scaler: Any
    labels: List[str]
    threshold: float = 0.65
    preprocessing: Dict[str, Any] = field(
        default_factory=lambda: {
            "mirror_left_hand": True,
            "normalize_rotation": True,
        }
    )

    # Thuộc tính tương thích ngược nếu cần
    @property
    def label_names(self) -> List[str]:
        return self.labels

    @property
    def preprocessor_config(self) -> Dict[str, Any]:
        return self.preprocessing

    @property
    def accept_thresholds(self) -> Dict[str, float]:
        return {label: self.threshold for label in self.labels}
