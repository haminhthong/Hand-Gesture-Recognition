"""Quy trình huấn luyện mô hình SVM cử chỉ tĩnh với đánh giá độc lập theo subject."""

import argparse
import json
import logging
import os
import sys
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Đưa src vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.config import TrainingConfig
from hand_gesture_controller.preprocessing import LandmarkPreprocessor
from hand_gesture_controller.schemas import GestureModelBundle

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def load_dataset(
    csv_path: str,
    preprocessor: LandmarkPreprocessor,
    allowed_labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Đọc dataset CSV và chuẩn hóa tọa độ landmark thành vector 63D."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    if allowed_labels:
        df = df[df["gesture"].isin(allowed_labels)].copy()

    features = []
    labels = []
    subjects = []

    for _, row in df.iterrows():
        coords = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            coords[i, 0] = float(row[f"x{i}"])
            coords[i, 1] = float(row[f"y{i}"])
            coords[i, 2] = float(row[f"z{i}"])

        feat = preprocessor.transform(coords, handedness=row.get("handedness", "Right"))
        features.append(feat)
        labels.append(row["gesture"])
        subjects.append(row["subject_id"])

    return np.array(features, dtype=np.float32), np.array(labels), np.array(subjects), df


def train_model(config: TrainingConfig) -> GestureModelBundle:
    """Huấn luyện RBF-SVM, tinh chỉnh siêu tham số và lưu mô hình."""
    logger.info("=== HUẤN LUYỆN STATIC GESTURE SVM ===")
    preprocessor = LandmarkPreprocessor(
        mirror_left_hand=config.mirror_left_hand,
        normalize_rotation=config.normalize_rotation,
    )

    X, y, subjects, df = load_dataset(config.dataset_path, preprocessor, allowed_labels=config.labels)
    if len(X) == 0:
        raise ValueError("Dataset không có mẫu nào thuộc các labels cấu hình.")

    unique_subs = set(subjects.tolist())
    if len(unique_subs) < 3:
        raise ValueError("Cần ít nhất 3 subjects để chia Train/Val/Test độc lập.")

    # Nạp hoặc tự động tạo splits
    if not os.path.exists(config.splits_path):
        from scripts.prepare_data import prepare_dataset
        splits = prepare_dataset(config.dataset_path, output_splits_path=config.splits_path)
    else:
        with open(config.splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

    train_subs = splits["train"]
    val_subs = splits["val"]
    test_subs = splits["test"]

    # Phân chia dữ liệu
    train_mask = np.isin(subjects, train_subs)
    val_mask = np.isin(subjects, val_subs)
    test_mask = np.isin(subjects, test_subs)

    X_train, y_train, sub_train = X[train_mask], y[train_mask], subjects[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    logger.info(
        "Kích thước phân chia: Train=%d mẫu (%d subs), Val=%d mẫu (%d subs), Test=%d mẫu (%d subs)",
        len(X_train), len(train_subs), len(X_val), len(val_subs), len(X_test), len(test_subs)
    )

    # StandardScaler FIT CHỈ TRÊN TRAIN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Tối ưu siêu tham số RBF-SVM bằng GroupKFold trên TRAIN
    logger.info("Bước 1: Tối ưu C và gamma qua GroupKFold...")
    best_c = 1.0
    best_gamma: Any = "scale"
    best_cv_f1 = -1.0

    train_groups = len(np.unique(sub_train))
    n_splits = min(config.cv_splits, train_groups)
    gkf = GroupKFold(n_splits=n_splits)

    for c in config.c_candidates:
        for g_str in config.gamma_candidates:
            gamma_val = float(g_str) if g_str not in ("scale", "auto") else g_str
            fold_f1s = []

            for tr_idx, hld_idx in gkf.split(X_train_scaled, y_train, groups=sub_train):
                clf = SVC(kernel="rbf", C=c, gamma=gamma_val)
                clf.fit(X_train_scaled[tr_idx], y_train[tr_idx])
                preds = clf.predict(X_train_scaled[hld_idx])
                fold_f1s.append(f1_score(y_train[hld_idx], preds, average="macro", zero_division=0))

            mean_f1 = float(np.mean(fold_f1s))
            if mean_f1 > best_cv_f1:
                best_cv_f1 = mean_f1
                best_c = c
                best_gamma = gamma_val

    logger.info("Tham số tốt nhất: C=%.2f, gamma=%s (CV Macro-F1: %.4f)", best_c, str(best_gamma), best_cv_f1)

    # 2. Huấn luyện và hiệu chuẩn xác suất trên toàn bộ TRAIN
    logger.info("Bước 2: Huấn luyện CalibratedClassifierCV trên tập Train...")
    base_svm = SVC(kernel="rbf", C=best_c, gamma=best_gamma, probability=True, random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_model.fit(X_train_scaled, y_train)

    # 3. Kiểm tra ngưỡng tin cậy trên VAL
    val_probs = calibrated_model.predict_proba(X_val_scaled)
    classes = calibrated_model.classes_

    best_threshold = config.confidence_threshold
    best_val_f1 = -1.0
    for t_cand in np.linspace(0.50, 0.80, 7):
        y_val_pred = []
        for p_row in val_probs:
            max_p = float(np.max(p_row))
            if max_p < t_cand:
                y_val_pred.append("NoAction")
            else:
                y_val_pred.append(classes[np.argmax(p_row)])
        val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = round(float(t_cand), 2)

    logger.info("Ngưỡng tin cậy tối ưu trên Val: %.2f (Val Macro-F1: %.4f)", best_threshold, best_val_f1)

    # 4. Đánh giá trên tập TEST độc lập (Unseen Subjects)
    test_probs = calibrated_model.predict_proba(X_test_scaled)
    y_test_pred = []
    rejected_count = 0
    for p_row in test_probs:
        max_p = float(np.max(p_row))
        if max_p < best_threshold:
            y_test_pred.append("NoAction")
            rejected_count += 1
        else:
            y_test_pred.append(classes[np.argmax(p_row)])

    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    reject_rate = rejected_count / len(X_test) if len(X_test) > 0 else 0.0

    logger.info("=== KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (NGƯỜI CHƯA TỪNG THẤY) ===")
    logger.info("Test Macro-F1: %.4f | Macro Precision: %.4f | Reject Rate: %.1f%%",
                test_macro_f1, test_precision, reject_rate * 100)
    logger.info("\nChi tiết phân loại:\n%s", classification_report(y_test, y_test_pred, zero_division=0))

    # 5. Lưu báo cáo đánh giá ra reports/evaluation.json
    metrics = {
        "train_cv_macro_f1": round(best_cv_f1, 4),
        "test_macro_f1": round(test_macro_f1, 4),
        "test_macro_precision": round(test_precision, 4),
        "reject_rate": round(reject_rate, 4),
        "best_c": best_c,
        "best_gamma": str(best_gamma),
        "confidence_threshold": best_threshold,
    }
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # 6. Xuất artifact mô hình tinh gọn
    bundle = GestureModelBundle(
        model=calibrated_model,
        scaler=scaler,
        labels=list(classes),
        threshold=best_threshold,
        preprocessing={
            "mirror_left_hand": config.mirror_left_hand,
            "normalize_rotation": config.normalize_rotation,
        },
    )

    os.makedirs(os.path.dirname(os.path.abspath(config.model_output_path)), exist_ok=True)
    joblib.dump(bundle, config.model_output_path)
    logger.info("Đã lưu model bundle tới: %s", config.model_output_path)

    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train static gesture SVM classifier with GroupKFold CV.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training YAML config file")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config) if os.path.exists(args.config) else TrainingConfig()
    train_model(cfg)


if __name__ == "__main__":
    main()
