"""Mô-đun nhận diện cử chỉ tĩnh production sử dụng Calibrated RBF-SVM và Reject Policy."""

import logging
import os
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

from ..features.landmark_preprocessor import LandmarkPreprocessor
from ..schemas import GestureModelBundle, HandObservation, StaticPrediction

logger = logging.getLogger(__name__)


class StaticGesturePredictor:
    """Bộ dự đoán cử chỉ tĩnh bằng mô hình Calibrated RBF-SVM kết hợp chính sách loại bỏ (Rejection Policy).

    Attributes:
        bundle (GestureModelBundle): Gói chứa mô hình, scaler, ngưỡng chấp nhận và siêu dữ liệu.
        preprocessor (LandmarkPreprocessor): Bộ chuẩn hóa Landmark 63D đồng bộ với pha huấn luyện.
    """

    DEFAULT_LABELS = ["Fist", "Select", "Options", "Stop", "Peace", "NoAction"]

    def __init__(
        self,
        model_bundle_path: Optional[str] = None,
        bundle: Optional[GestureModelBundle] = None,
        default_accept_threshold: float = 0.60,
    ) -> None:
        """Khởi tạo StaticGesturePredictor từ đường dẫn file .joblib hoặc bundle trực tiếp.

        Args:
            model_bundle_path: Đường dẫn tệp artifact mô hình (.joblib).
            bundle: Đối tượng GestureModelBundle đã nạp sẵn.
            default_accept_threshold: Ngưỡng xác suất tối thiểu mặc định để chấp nhận một cử chỉ.
        """
        self.default_accept_threshold = default_accept_threshold
        self.bundle: Optional[GestureModelBundle] = None

        if bundle is not None:
            self.bundle = bundle
        elif model_bundle_path and os.path.exists(model_bundle_path):
            self.bundle = self._load_bundle(model_bundle_path)

        if self.bundle is not None:
            pre_cfg = self.bundle.preprocessor_config or {}
            self.preprocessor = LandmarkPreprocessor(
                mirror_left_hand=pre_cfg.get("mirror_left_hand", True),
                normalize_rotation=pre_cfg.get("normalize_rotation", True),
            )
            self.label_names = self.bundle.label_names
            self.accept_thresholds = self.bundle.accept_thresholds or {}
            logger.info("Đã nạp thành công StaticGesturePredictor artifact v%s", self.bundle.model_version)
        else:
            self.preprocessor = LandmarkPreprocessor(mirror_left_hand=True, normalize_rotation=True)
            self.label_names = self.DEFAULT_LABELS
            self.accept_thresholds = {}
            logger.warning("Không có model bundle được nạp. Predictor đang ở trạng thái chưa khởi tạo mô hình.")

    @staticmethod
    def _load_bundle(path: str) -> GestureModelBundle:
        """Nạp và kiểm tra tính hợp lệ của tệp bundle .joblib."""
        loaded = joblib.load(path)
        if isinstance(loaded, GestureModelBundle):
            return loaded
        if isinstance(loaded, dict):
            return GestureModelBundle(
                model=loaded.get("model"),
                scaler=loaded.get("scaler"),
                label_names=loaded.get("label_names", []),
                accept_thresholds=loaded.get("accept_thresholds", {}),
                preprocessor_config=loaded.get("preprocessor_config", {}),
                feature_schema_version=loaded.get("feature_schema_version", "1.0.0"),
                dataset_manifest_hash=loaded.get("dataset_manifest_hash", ""),
                training_config=loaded.get("training_config", {}),
                metrics=loaded.get("metrics", {}),
                git_commit=loaded.get("git_commit", ""),
                model_version=loaded.get("model_version", "1.0.0"),
            )
        raise ValueError(f"Định dạng model artifact không hợp lệ: {type(loaded)}")

    @property
    def is_ready(self) -> bool:
        """Kiểm tra mô hình đã sẵn sàng thực hiện suy luận hay chưa."""
        return self.bundle is not None and self.bundle.model is not None

    def predict_observation(self, observation: HandObservation) -> StaticPrediction:
        """Dự đoán cử chỉ từ quan sát bàn tay HandObservation.

        Args:
            observation: Quan sát bàn tay từ module perception.

        Returns:
            StaticPrediction: Kết quả phân loại kèm xác suất và cờ rejected.
        """
        features_63d = self.preprocessor.transform_observation(observation)
        return self.predict_features(features_63d)

    def predict_features(self, features_63d: np.ndarray) -> StaticPrediction:
        """Dự đoán cử chỉ từ vector đặc trưng 63D đã tiền xử lý.

        Quy trình:
        1. Chuẩn hóa vector bằng StandardScaler trong bundle.
        2. Tính xác suất các lớp P(c) bằng mô hình SVM đã hiệu chuẩn (Calibrated SVM).
        3. Chọn lớp argmax c* có xác suất p_max.
        4. Áp dụng Rejection Policy: Nếu p_max < T_accept(c*) -> trả về NoAction kèm rejected=True.

        Args:
            features_63d: Mảng 1D 63 chiều (float32).

        Returns:
            StaticPrediction: Dự đoán đã áp dụng chính sách an toàn.
        """
        if not self.is_ready or self.bundle is None:
            return StaticPrediction(
                label="NoAction",
                confidence=0.0,
                probabilities={},
                rejected=True,
                source="uninitialized",
            )

        feat_2d = features_63d.reshape(1, -1)
        if self.bundle.scaler is not None:
            feat_2d = self.bundle.scaler.transform(feat_2d)

        # Tính xác suất các lớp
        if hasattr(self.bundle.model, "predict_proba"):
            probs = self.bundle.model.predict_proba(feat_2d)[0]
            classes = getattr(self.bundle.model, "classes_", self.label_names)
            prob_dict: Dict[str, float] = {
                str(cls): float(p) for cls, p in zip(classes, probs)
            }
            # Lấy nhãn có xác suất cao nhất
            best_label = max(prob_dict, key=prob_dict.get)  # type: ignore
            max_prob = prob_dict[best_label]
        else:
            best_label = str(self.bundle.model.predict(feat_2d)[0])
            max_prob = 1.0
            prob_dict = {best_label: 1.0}

        # Áp dụng Rejection Policy theo ngưỡng chấp nhận
        accept_threshold = self.accept_thresholds.get(best_label, self.default_accept_threshold)
        if max_prob < accept_threshold:
            return StaticPrediction(
                label="NoAction",
                confidence=max_prob,
                probabilities=prob_dict,
                rejected=True,
                source=f"svm_{self.bundle.model_version}_rejected",
            )

        return StaticPrediction(
            label=best_label,
            confidence=max_prob,
            probabilities=prob_dict,
            rejected=False,
            source=f"svm_{self.bundle.model_version}",
        )
