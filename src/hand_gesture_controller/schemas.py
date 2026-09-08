"""Mô-đun định nghĩa các hợp đồng dữ liệu (Data Schemas / Contracts) cho toàn bộ hệ thống HCI."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GestureEvent(str, Enum):
    """Tập các sự kiện hành động không phụ thuộc trực tiếp vào tên cử chỉ cụ thể."""

    NONE = "none"
    START_DRAG = "start_drag"
    DRAG = "drag"
    STOP_DRAG = "stop_drag"
    CHANGE_COLOR = "change_color"
    DELETE_OBJECT = "delete_object"
    TOGGLE_CANVAS = "toggle_canvas"
    OPEN_MENU = "open_menu"
    EMERGENCY_SOS = "emergency_sos"


@dataclass(frozen=True)
class HandObservation:
    """Biểu diễn dữ liệu cảm nhận (Perception) từ khung hình camera sau khi xử lý MediaPipe.

    Module Perception chỉ trả ra đối tượng này, tách biệt hoàn toàn MediaPipe khỏi downstream.
    """

    landmarks: np.ndarray  # Mảng float32 hình dạng (21, 3) đại diện (x, y, z)
    handedness: str  # "Left" hoặc "Right"
    handedness_score: float  # Độ tin cậy nhận diện tay trái/phải [0.0, 1.0]
    timestamp: float  # Thời điểm monotonic (time.perf_counter)
    frame_width: int  # Chiều rộng thực tế của khung hình camera
    frame_height: int  # Chiều cao thực tế của khung hình camera
    palm_size: float  # Khoảng cách 2D tham chiếu (cổ tay -> middle_mcp)
    hand_center: Tuple[float, float]  # Tọa độ trọng tâm bàn tay 2D (x, y) chuẩn hóa [0.0, 1.0]

    def __post_init__(self) -> None:
        if self.landmarks.shape != (21, 3):
            raise ValueError(f"Landmarks phải có kích thước (21, 3), nhận được {self.landmarks.shape}")


@dataclass(frozen=True)
class StaticPrediction:
    """Kết quả dự đoán cử chỉ tĩnh từ mô hình Machine Learning hoặc Rule Baseline."""

    label: str  # Nhãn cử chỉ ("Fist", "Select", "Options", "Stop", "Peace", "NoAction")
    confidence: float  # Xác suất cao nhất hoặc độ tin cậy [0.0, 1.0]
    probabilities: Dict[str, float] = field(default_factory=dict)  # Xác suất từng lớp
    rejected: bool = False  # True nếu xác suất < accept_threshold và bị chuyển về NoAction
    source: str = "svm_v1"  # Nguồn nhận diện ("svm_v1", "rule_baseline", ...)


@dataclass(frozen=True)
class StableGesture:
    """Cử chỉ sau khi đã đi qua bộ lọc ổn định thời gian (Temporal Stabilizer)."""

    label: str  # Nhãn cử chỉ ổn định
    confidence: float  # Độ tin cậy đại diện
    dwell_time_ms: float  # Thời gian duy trì cử chỉ liên tục (mili-giây)
    timestamp: float  # Thời điểm ghi nhận (giây)


@dataclass(frozen=True)
class HCIEvent:
    """Sự kiện tương tác người-máy hoàn chỉnh phát sinh cho Application Layer."""

    event_type: GestureEvent  # Loại sự kiện hành động
    timestamp: float  # Mốc thời gian phát sinh sự kiện
    gesture: str  # Cử chỉ kích hoạt sự kiện
    cursor_pos: Optional[Tuple[int, int]] = None  # Tọa độ con trỏ màn hình (px, py)


@dataclass
class GestureModelBundle:
    """Gói artifact mô hình static gesture hoàn chỉnh lưu trữ trên đĩa (.joblib)."""

    model: Any  # Mô hình phân loại (CalibratedClassifierCV hoặc SVC)
    scaler: Any  # StandardScaler tương thích 63D
    label_names: List[str]  # Danh sách tên các nhãn cử chỉ
    accept_thresholds: Dict[str, float]  # Ngưỡng chấp nhận xác suất theo từng nhãn
    preprocessor_config: Dict[str, Any]  # Cấu hình tiền xử lý (mirror, rotation, ...)
    feature_schema_version: str = "1.0.0"
    dataset_manifest_hash: str = ""
    training_config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    model_version: str = "1.0.0"
