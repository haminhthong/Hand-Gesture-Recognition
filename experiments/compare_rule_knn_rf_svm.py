"""Thực nghiệm so sánh đa mô hình Baseline (KNN, SVM, Random Forest) và Rule Engine trên Subject-Level CV."""

import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.features.landmark_preprocessor import LandmarkPreprocessor
from hand_gesture_controller.recognition.rule_baseline import RuleStaticBaseline
from hand_gesture_controller.schemas import HandObservation

logger = logging.getLogger("compare_baselines")


def load_dataset_features(
    csv_path: str,
    normalize_rotation: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[HandObservation]]:
    """Đọc dataset CSV và trích xuất vector đặc trưng 63D cùng HandObservation cho Rule Engine."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    preprocessor = LandmarkPreprocessor(mirror_left_hand=True, normalize_rotation=normalize_rotation)

    features = []
    labels = []
    groups = []
    observations = []

    for _, row in df.iterrows():
        coords = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            coords[i, 0] = float(row[f"x{i}"])
            coords[i, 1] = float(row[f"y{i}"])
            coords[i, 2] = float(row[f"z{i}"])

        feat = preprocessor.transform(coords, handedness=row.get("handedness", "Right"))
        features.append(feat)
        labels.append(row["gesture"])
        groups.append(row["subject_id"])

        obs = HandObservation(
            landmarks=coords,
            handedness=row.get("handedness", "Right"),
            handedness_score=1.0,
            timestamp=0.0,
            frame_width=640,
            frame_height=480,
            palm_size=float(np.linalg.norm(coords[9] - coords[0])),
            hand_center=(0.5, 0.5),
        )
        observations.append(obs)

    return np.array(features, dtype=np.float32), np.array(labels), np.array(groups), observations


def compare_baselines(
    csv_path: str,
    cv_strategy: str = "groupkfold",
    n_splits: int = 5,
    normalize_rotation: bool = True,
) -> Dict[str, Any]:
    """Chạy thực nghiệm so sánh độc lập giữa Rule Engine, KNN, Random Forest và RBF-SVM."""
    X, y, groups, observations = load_dataset_features(csv_path, normalize_rotation=normalize_rotation)
    unique_groups = np.unique(groups)

    if cv_strategy.lower() == "loso" or len(unique_groups) < n_splits:
        cv = LeaveOneGroupOut()
        splits_count = len(unique_groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        splits_count = n_splits

    models: Dict[str, Any] = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Calibrated RBF-SVM (C=10)": SVC(kernel="rbf", C=10.0, probability=True, random_state=42),
    }

    results: Dict[str, Dict[str, List[float]]] = {
        m_name: {"macro_f1": [], "balanced_acc": [], "acc": []} for m_name in models
    }
    results["Rule Baseline"] = {"macro_f1": [], "balanced_acc": [], "acc": []}

    rule_engine = RuleStaticBaseline()

    logger.info("=== BẮT ĐẦU EXPERIMENT SO SÁNH BASELINES (%s - %d Folds) ===", cv_strategy.upper(), splits_count)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # 1. Đánh giá ML models
        for m_name, clf in models.items():
            clf.fit(X_tr_sc, y_tr)
            preds = clf.predict(X_te_sc)
            results[m_name]["macro_f1"].append(f1_score(y_te, preds, average="macro", zero_division=0))
            results[m_name]["balanced_acc"].append(balanced_accuracy_score(y_te, preds))
            results[m_name]["acc"].append(accuracy_score(y_te, preds))

        # 2. Đánh giá Rule Engine trên tập test
        rule_preds = [rule_engine.predict_observation(observations[i]).label for i in test_idx]
        results["Rule Baseline"]["macro_f1"].append(f1_score(y_te, rule_preds, average="macro", zero_division=0))
        results["Rule Baseline"]["balanced_acc"].append(balanced_accuracy_score(y_te, rule_preds))
        results["Rule Baseline"]["acc"].append(accuracy_score(y_te, rule_preds))

    # Tóm tắt
    summary = {}
    print("\n" + "=" * 70)
    print(f"{'Mô hình':<30} | {'Macro-F1':<12} | {'Balanced Acc':<12} | {'Accuracy':<10}")
    print("-" * 70)
    for name, metrics in results.items():
        mf1 = float(np.mean(metrics["macro_f1"]))
        bacc = float(np.mean(metrics["balanced_acc"]))
        acc = float(np.mean(metrics["acc"]))
        summary[name] = {"macro_f1": mf1, "balanced_acc": bacc, "acc": acc}
        print(f"{name:<30} | {mf1:<12.4f} | {bacc:<12.4f} | {acc:<10.4f}")
    print("=" * 70 + "\n")

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="So sánh Baselines và Rule Engine.")
    parser.add_argument("--csv", type=str, default="data/raw/landmarks_dataset.csv", help="Dataset CSV")
    parser.add_argument("--cv", type=str, default="groupkfold", choices=["groupkfold", "loso"], help="Chiến lược CV")
    parser.add_argument("--splits", type=int, default=5, help="Số folds")
    args = parser.parse_args()

    compare_baselines(csv_path=args.csv, cv_strategy=args.cv, n_splits=args.splits)


if __name__ == "__main__":
    main()
