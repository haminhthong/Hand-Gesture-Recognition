"""Bộ phân loại cử chỉ tĩnh (Static Gesture Classifier) sử dụng RBF-SVM và ngưỡng tin cậy."""

import logging
import os
from typing import Dict, List, Optional

import joblib
import numpy as np

from .preprocessing import LandmarkPreprocessor
from .schemas import GestureModelBundle, HandObservation, StaticPrediction

logger = logging.getLogger(__name__)


class StaticGestureClassifier:
    """Mô hình phân loại cử chỉ tĩnh 63D sử dụng RBF-SVM kết hợp Confidence Threshold."""

    DEFAULT_LABELS = ["Fist", "Select", "Options", "Stop", "Peace", "NoAction"]

    def __init__(
        self,
        model_bundle_path: Optional[str] = None,
        bundle: Optional[GestureModelBundle] = None,
        confidence_threshold: float = 0.65,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.bundle: Optional[GestureModelBundle] = None

        if bundle is not None:
            self.bundle = bundle
        elif model_bundle_path and os.path.exists(model_bundle_path):
            self.bundle = self._load_bundle(model_bundle_path)

        if self.bundle is not None:
            pre_cfg = self.bundle.preprocessing or {}
            self.preprocessor = LandmarkPreprocessor(
                mirror_left_hand=pre_cfg.get("mirror_left_hand", True),
                normalize_rotation=pre_cfg.get("normalize_rotation", True),
            )
            self.label_names: List[str] = self.bundle.labels
            # Nếu bundle có ngưỡng riêng và người dùng không override khác mặc định
            if hasattr(self.bundle, "threshold") and self.bundle.threshold is not None:
                self.confidence_threshold = self.bundle.threshold
            logger.info("Đã nạp thành công mô hình StaticGestureClassifier.")
        else:
            self.preprocessor = LandmarkPreprocessor(mirror_left_hand=True, normalize_rotation=True)
            self.label_names = self.DEFAULT_LABELS

    @staticmethod
    def _load_bundle(path: str) -> GestureModelBundle:
        """Nạp và chuẩn hóa tệp artifact .joblib thành GestureModelBundle."""
        loaded = joblib.load(path)
        if isinstance(loaded, GestureModelBundle):
            return loaded
        if isinstance(loaded, dict):
            return GestureModelBundle(
                model=loaded.get("model"),
                scaler=loaded.get("scaler"),
                labels=loaded.get("labels") or loaded.get("label_names") or StaticGestureClassifier.DEFAULT_LABELS,
                threshold=loaded.get("threshold", 0.65),
                preprocessing=loaded.get("preprocessing") or loaded.get("preprocessor_config") or {},
            )
        raise ValueError(f"Định dạng model artifact không hợp lệ: {type(loaded)}")

    @property
    def is_ready(self) -> bool:
        """Kiểm tra mô hình đã sẵn sàng thực hiện suy luận hay chưa."""
        return self.bundle is not None and self.bundle.model is not None

    def predict_observation(self, observation: HandObservation) -> StaticPrediction:
        """Dự đoán cử chỉ từ HandObservation."""
        features_63d = self.preprocessor.transform_observation(observation)
        return self.predict_features(features_63d)

    def predict_features(self, features_63d: np.ndarray) -> StaticPrediction:
        """Dự đoán cử chỉ từ vector đặc trưng 63D đã chuẩn hóa."""
        if features_63d.shape != (63,):
            raise ValueError(
                f"Vector đặc trưng phải có kích thước (63,), nhận được {features_63d.shape}"
            )
        if not np.isfinite(features_63d).all():
            raise ValueError("Vector đặc trưng chứa giá trị không hợp lệ (NaN hoặc Inf).")

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
            best_label = max(prob_dict, key=prob_dict.get)  # type: ignore
            max_prob = prob_dict[best_label]
        else:
            best_label = str(self.bundle.model.predict(feat_2d)[0])
            max_prob = 1.0
            prob_dict = {best_label: 1.0}

        # Áp dụng Confidence Threshold
        if max_prob < self.confidence_threshold:
            return StaticPrediction(
                label="NoAction",
                confidence=max_prob,
                probabilities=prob_dict,
                rejected=True,
                source="svm_rejected",
            )

        return StaticPrediction(
            label=best_label,
            confidence=max_prob,
            probabilities=prob_dict,
            rejected=False,
            source="svm",
        )


# Alias tương thích ngược cho StaticGesturePredictor
StaticGesturePredictor = StaticGestureClassifier
