"""Quy trình huấn luyện mô hình Calibrated RBF-SVM nhận diện cử chỉ tĩnh cho production."""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, precision_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Thêm src vào sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.config import TrainingConfig
from hand_gesture_controller.features.landmark_preprocessor import LandmarkPreprocessor
from hand_gesture_controller.schemas import GestureModelBundle

logger = logging.getLogger("train_static_model")


def load_dataset_features(
    csv_path: str,
    preprocessor: LandmarkPreprocessor,
    allowed_labels: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Đọc CSV và trích xuất vector 63D qua LandmarkPreprocessor."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy dataset: {csv_path}")

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


def train_static_gesture_model(
    config: TrainingConfig,
) -> GestureModelBundle:
    """Toàn bộ quy trình huấn luyện, tối ưu siêu tham số, hiệu chuẩn và xuất artifact."""
    logger.info("=== KHỞI CHẠY HUẤN LUYỆN STATIC GESTURE SVM MODEL ===")
    preprocessor = LandmarkPreprocessor(
        mirror_left_hand=config.mirror_left_hand,
        normalize_rotation=config.normalize_rotation,
    )

    X, y, subjects, df = load_dataset_features(
        config.dataset_path, preprocessor, allowed_labels=config.labels
    )
    if len(X) == 0:
        raise ValueError("Dataset không có mẫu nào thuộc các labels đã cấu hình.")
    unique_subjects = set(subjects.tolist())
    if len(unique_subjects) < 3:
        raise ValueError("Cần ít nhất 3 subjects để có Train/Val/Test độc lập.")
    logger.info("Tổng số mẫu nạp được: %d với %d đặc trưng 63D", len(X), X.shape[1])

    # Nạp splits nếu có, nếu chưa thì tạo tự động
    splits_path = config.splits_path
    if not os.path.exists(splits_path):
        from scripts.create_splits import create_subject_splits
        splits = create_subject_splits(config.dataset_path, output_splits_path=splits_path)
    else:
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

    train_subs = splits["train"]
    val_subs = splits["val"]
    test_subs = splits["test"]

    train_subject_set = set(train_subs)
    val_subject_set = set(val_subs)
    test_subject_set = set(test_subs)
    if (
        train_subject_set & val_subject_set
        or train_subject_set & test_subject_set
        or val_subject_set & test_subject_set
    ):
        raise ValueError("Splits bị rò rỉ subject giữa train, val và test.")
    if train_subject_set | val_subject_set | test_subject_set != unique_subjects:
        raise ValueError("Splits không bao phủ đúng toàn bộ subjects trong dataset.")

    train_mask = np.isin(subjects, train_subs)
    val_mask = np.isin(subjects, val_subs)
    test_mask = np.isin(subjects, test_subs)

    X_train, y_train, sub_train = X[train_mask], y[train_mask], subjects[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Train, Val và Test đều phải có dữ liệu.")
    train_class_counts = pd.Series(y_train).value_counts()
    if train_class_counts.min() < 3:
        raise ValueError("Mỗi lớp trong Train cần ít nhất 3 mẫu để hiệu chuẩn SVM.")

    logger.info("Kích thước phân chia: Train=%d mẫu (%d subs), Val=%d mẫu (%d subs), Test=%d mẫu (%d subs)",
                len(X_train), len(train_subs), len(X_val), len(val_subs), len(X_test), len(test_subs))

    # Chuẩn hóa StandardScaler fit trên TRAIN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 1. Tối ưu siêu tham số RBF-SVM (C, gamma) bằng GroupKFold trên TRAIN
    logger.info("--- BƯỚC 1: TỐI ƯU C & GAMMA TRÊN TRAIN VỚI GROUPKFOLD ---")
    best_c = 1.0
    best_gamma: Any = "scale"
    best_cv_f1 = -1.0

    train_group_count = len(np.unique(sub_train))
    if train_group_count < 2:
        raise ValueError("Train cần ít nhất 2 subjects để chạy GroupKFold.")
    gkf = GroupKFold(n_splits=min(config.cv_splits, train_group_count))

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

    logger.info("Cặp tham số tốt nhất: C=%.2f, gamma=%s (Train CV Macro-F1: %.4f)",
                best_c, str(best_gamma), best_cv_f1)

    # 2. Huấn luyện mô hình cơ sở trên toàn bộ TRAIN và hiệu chuẩn xác suất
    logger.info("--- BƯỚC 2: FIT VÀ CALIBRATION TRÊN TOÀN BỘ TRAIN ---")
    base_svm = SVC(kernel="rbf", C=best_c, gamma=best_gamma, probability=True, random_state=42)
    calibrated_model = CalibratedClassifierCV(estimator=base_svm, cv=3)
    calibrated_model.fit(X_train_scaled, y_train)

    # 3. Tinh chỉnh ngưỡng chấp nhận (Reject Thresholds) trên VAL
    logger.info("--- BƯỚC 3: DÒ TÌM NGƯỠNG CHẤP NHẬN TRÊN TẬP VAL ---")
    val_probs = calibrated_model.predict_proba(X_val_scaled)
    classes = calibrated_model.classes_

    accept_thresholds: Dict[str, float] = {}
    actionable_gestures = [g for g in config.labels if g not in ("NoAction", "Fist")]

    for cls in config.labels:
        accept_thresholds[cls] = 0.60  # Ngưỡng ban đầu

    # Sweep ngưỡng toàn cục, ưu tiên đạt precision và false-action rate mục tiêu.
    best_t = 0.60
    best_val_score = -1.0
    best_feasible = False

    for t_cand in np.linspace(0.50, 0.85, 8):
        y_val_pred_filtered = []
        for p_row in val_probs:
            max_p = float(np.max(p_row))
            argmax_cls = classes[np.argmax(p_row)]
            if max_p < t_cand:
                y_val_pred_filtered.append("NoAction")
            else:
                y_val_pred_filtered.append(argmax_cls)

        macro_f1 = f1_score(y_val, y_val_pred_filtered, average="macro", zero_division=0)
        # Tính False Action Rate: Mẫu NoAction nhưng bị đoán thành actionable
        no_action_mask = (y_val == "NoAction")
        if np.sum(no_action_mask) > 0:
            false_actions = sum(1 for yt, yp in zip(y_val, y_val_pred_filtered) if yt == "NoAction" and yp in actionable_gestures)
            false_action_rate = false_actions / np.sum(no_action_mask)
        else:
            false_action_rate = 0.0

        predicted_actionable = np.isin(y_val_pred_filtered, actionable_gestures)
        true_actionable = np.isin(y_val, actionable_gestures)
        predicted_count = int(predicted_actionable.sum())
        actionable_precision = (
            float((predicted_actionable & true_actionable).sum() / predicted_count)
            if predicted_count
            else 0.0
        )
        feasible = (
            actionable_precision >= config.target_precision_actionable
            and false_action_rate <= config.max_false_action_rate
        )
        # Khi có nhiều ngưỡng đạt mục tiêu, chọn Macro-F1 cao nhất; nếu chưa
        # có ngưỡng khả thi, dùng điểm phạt để tìm ứng viên gần mục tiêu nhất.
        score = macro_f1 if feasible else macro_f1 - 2.0 * false_action_rate
        if (feasible and not best_feasible) or (feasible == best_feasible and score > best_val_score):
            best_val_score = score
            best_t = float(t_cand)
            best_feasible = feasible

    logger.info("Ngưỡng tối ưu dò được trên Val: T_accept=%.2f", best_t)
    for g in config.labels:
        accept_thresholds[g] = best_t
    # Đặt ngưỡng cao hơn cho các hành động nhạy cảm
    accept_thresholds["Stop"] = max(best_t, 0.70)
    accept_thresholds["Options"] = max(best_t, 0.65)

    # 4. Đánh giá cuối cùng trên LOCKED TEST (Unseen Subjects)
    logger.info("--- BƯỚC 4: ĐÁNH GIÁ TRÊN LOCKED UNSEEN-SUBJECT TEST ---")
    test_probs = calibrated_model.predict_proba(X_test_scaled)
    y_test_pred = []
    rejected_count = 0

    for p_row in test_probs:
        max_p = float(np.max(p_row))
        pred_cls = classes[np.argmax(p_row)]
        if max_p < accept_thresholds.get(pred_cls, best_t):
            y_test_pred.append("NoAction")
            rejected_count += 1
        else:
            y_test_pred.append(pred_cls)

    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
    reject_rate = rejected_count / len(X_test) if len(X_test) > 0 else 0.0

    logger.info("=== KẾT QUẢ TRÊN TẬP TEST KHÓA ===")
    logger.info("Test Macro-F1: %.4f | Macro Precision: %.4f | Reject Rate: %.2f%%",
                test_macro_f1, test_precision, reject_rate * 100)
    logger.info("\nBáo cáo phân loại:\n%s", classification_report(y_test, y_test_pred, zero_division=0))

    # 5. Đóng gói Artifact GestureModelBundle
    manifest_hash = ""
    if os.path.exists(config.manifest_path):
        with open(config.manifest_path, "r", encoding="utf-8") as f:
            manifest_hash = json.load(f).get("csv_sha256", "")

    metrics = {
        "train_cv_macro_f1": best_cv_f1,
        "test_macro_f1": test_macro_f1,
        "test_macro_precision": test_precision,
        "reject_rate": reject_rate,
        "best_c": best_c,
        "best_gamma": str(best_gamma),
        "best_threshold": best_t,
    }

    bundle = GestureModelBundle(
        model=calibrated_model,
        scaler=scaler,
        label_names=list(classes),
        accept_thresholds=accept_thresholds,
        preprocessor_config={
            "mirror_left_hand": config.mirror_left_hand,
            "normalize_rotation": config.normalize_rotation,
        },
        feature_schema_version="1.0.0",
        dataset_manifest_hash=manifest_hash,
        training_config={
            "c": best_c,
            "gamma": str(best_gamma),
            "cv_splits": config.cv_splits,
        },
        metrics=metrics,
        model_version="1.0.0",
    )

    os.makedirs(os.path.dirname(os.path.abspath(config.model_output_path)), exist_ok=True)
    joblib.dump(bundle, config.model_output_path)
    logger.info("Đã xuất artifact hoàn chỉnh tới: %s", config.model_output_path)

    return bundle


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Calibrated RBF-SVM.")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="File config training")
    args = parser.parse_args()

    cfg = TrainingConfig.from_yaml(args.config) if os.path.exists(args.config) else TrainingConfig()
    train_static_gesture_model(cfg)


if __name__ == "__main__":
    main()
