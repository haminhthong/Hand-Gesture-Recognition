"""Kịch bản huấn luyện và đánh giá mô hình Machine Learning Baseline (KNN, SVM, Random Forest) & Rule Engine.

Thực hiện phân chia dataset theo subject_id (GroupKFold hoặc Leave-One-Subject-Out / LeaveOneGroupOut)
nhằm triệt tiêu data leakage và đảm bảo tính khái quát hóa cho người dùng chưa từng xuất hiện (unseen subjects).
"""

import argparse
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.gesture_detector import GestureDetector
from hand_gesture_controller.preprocessing import LandmarkPreprocessor

logger = logging.getLogger("train_baseline")


def load_and_preprocess_dataset(
    csv_path: str,
    normalize_rotation: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Đọc tệp CSV và chuyển đổi 21 landmarks 3D qua LandmarkPreprocessor.

    Args:
        csv_path: Đường dẫn tệp CSV dataset.
        normalize_rotation: Có chuẩn hóa xoay mặt phẳng hay không.

    Returns:
        Tuple[X, y, groups, raw_landmarks]:
            - X: Vector đặc trưng 63 chiều đã chuẩn hóa.
            - y: Mảng nhãn cử chỉ.
            - groups: Mảng subject_id tương ứng từng mẫu.
            - raw_landmarks: Danh sách đối tượng landmarks giả lập phục vụ Rule Engine.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["subject_id", "gesture", "handedness"] + [
        f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu các cột quy định: {missing}")

    preprocessor = LandmarkPreprocessor(mirror_left_hand=True, normalize_rotation=normalize_rotation)
    features_list = []
    labels_list = []
    groups_list = []
    raw_landmarks_list = []

    for _, row in df.iterrows():
        coords = np.zeros((21, 3), dtype=np.float32)
        mock_pts = []
        for i in range(21):
            x = float(row[f"x{i}"])
            y = float(row[f"y{i}"])
            z = float(row[f"z{i}"])
            coords[i, 0] = x
            coords[i, 1] = y
            coords[i, 2] = z
            mock_pts.append(SimpleNamespace(x=x, y=y, z=z))

        feat = preprocessor.transform(coords, handedness=row["handedness"])
        features_list.append(feat)
        labels_list.append(row["gesture"])
        groups_list.append(row["subject_id"])
        raw_landmarks_list.append(SimpleNamespace(landmark=mock_pts))

    return (
        np.array(features_list),
        np.array(labels_list),
        np.array(groups_list),
        raw_landmarks_list,
    )


def evaluate_baselines(
    csv_path: str,
    cv_strategy: str = "groupkfold",
    n_splits: int = 5,
    compare_rules: bool = True,
    normalize_rotation: bool = False,
) -> None:
    """Chạy đánh giá Subject-Independent Cross-Validation cho ML Baselines và Rule Engine.

    Args:
        csv_path: Đường dẫn dataset CSV.
        cv_strategy: Phương pháp chia nhóm ('groupkfold' hoặc 'loso').
        n_splits: Số fold nếu dùng groupkfold.
        compare_rules: Có đánh giá Rule Engine trên cùng các fold hay không.
        normalize_rotation: Có bật tính năng chuẩn hóa góc xoay mặt phẳng hay không.
    """
    logger.info("Đang nạp và tiền xử lý dataset từ %s...", csv_path)
    X, y, groups, raw_landmarks = load_and_preprocess_dataset(
        csv_path, normalize_rotation=normalize_rotation
    )

    unique_subjects = np.unique(groups)
    logger.info(
        "Dataset có tổng cộng %d mẫu từ %d subjects: %s",
        len(X),
        len(unique_subjects),
        list(unique_subjects),
    )

    if len(unique_subjects) < 2:
        logger.error("Cần ít nhất 2 subjects khác nhau trong dataset để kiểm thử subject-independent!")
        return

    # Lựa chọn Cross-Validation Splitter
    if cv_strategy.lower() == "loso":
        cv = LeaveOneGroupOut()
        total_folds = len(unique_subjects)
        logger.info("Sử dụng Leave-One-Subject-Out (LOSO) Cross-Validation với %d folds.", total_folds)
    else:
        splits = min(n_splits, len(unique_subjects))
        cv = GroupKFold(n_splits=splits)
        total_folds = splits
        logger.info("Sử dụng GroupKFold Cross-Validation với %d splits.", total_folds)

    classifiers: Dict[str, Any] = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0),
        "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    if compare_rules:
        classifiers["Rule Engine (Heuristic)"] = "rule_engine"

    results = []

    for name, clf in classifiers.items():
        macro_f1_scores = []
        acc_scores = []
        bal_acc_scores = []
        all_y_val = []
        all_y_pred = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Kiểm soát triệt để Data Leakage: tập train và val không được giao nhau về subject
            train_subs = set(groups[train_idx])
            val_subs = set(groups[val_idx])
            assert train_subs.isdisjoint(val_subs), "Cảnh báo Data Leakage: Subject xuất hiện ở cả Train và Val!"

            if clf == "rule_engine":
                # Rule Engine đánh giá trực tiếp trên các mẫu raw landmarks của fold val
                rule_detector = GestureDetector()
                y_pred = [
                    rule_detector.detect_static_gesture_result(raw_landmarks[idx]).label
                    for idx in val_idx
                ]
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

            acc_scores.append(acc)
            bal_acc_scores.append(bal_acc)
            macro_f1_scores.append(f1)

            all_y_val.extend(y_val)
            all_y_pred.extend(y_pred)

        avg_acc = np.mean(acc_scores)
        avg_bal_acc = np.mean(bal_acc_scores)
        avg_f1 = np.mean(macro_f1_scores)
        std_f1 = np.std(macro_f1_scores)

        results.append({
            "Phương pháp": name,
            "Macro-F1 (Mean ± Std)": f"{avg_f1:.4f} ± {std_f1:.4f}",
            "Balanced Acc": f"{avg_bal_acc:.4f}",
            "Accuracy": f"{avg_acc:.4f}",
        })

    logger.info("=== BẢNG KẾT QUẢ ĐÁNH GIÁ SUBJECT-INDEPENDENT BENCHMARK ===")
    res_df = pd.DataFrame(results)
    print("\n", res_df.to_string(index=False), "\n")


def main() -> None:
    """Khởi chạy CLI cho train_baseline."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    parser = argparse.ArgumentParser(description="Kịch bản huấn luyện & đánh giá Baseline ML & Rule Engine")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/landmarks_dataset.csv",
        help="Đường dẫn tệp CSV dataset",
    )
    parser.add_argument(
        "--cv",
        type=str,
        choices=["groupkfold", "loso"],
        default="groupkfold",
        help="Chiến lược Cross-Validation: groupkfold hoặc loso (Leave-One-Subject-Out)",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Số folds cho GroupKFold (mặc định: 5)",
    )
    parser.add_argument(
        "--compare-rules",
        action="store_true",
        default=True,
        help="So sánh đối đầu trực tiếp Rule Engine trên cùng tập test",
    )
    parser.add_argument(
        "--normalize-rotation",
        action="store_true",
        default=False,
        help="Thử nghiệm chuẩn hóa xoay mặt phẳng bàn tay (Rotation Invariance)",
    )

    args = parser.parse_args()
    evaluate_baselines(
        args.dataset,
        cv_strategy=args.cv,
        n_splits=args.splits,
        compare_rules=args.compare_rules,
        normalize_rotation=args.normalize_rotation,
    )


if __name__ == "__main__":
    main()
