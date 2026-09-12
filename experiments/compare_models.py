"""Thực nghiệm so sánh các phương pháp: Rule-based Baseline vs KNN vs Random Forest vs SVM."""

import argparse
import logging
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.preprocessing import LandmarkPreprocessor
from hand_gesture_controller.rule_baseline import RuleStaticBaseline
from hand_gesture_controller.schemas import HandObservation

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("compare_models")


def load_dataset_features(
    csv_path: str,
    normalize_rotation: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[HandObservation]]:
    """Đọc dataset CSV và trích xuất vector đặc trưng 63D cùng HandObservation cho Rule Engine."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    preprocessor = LandmarkPreprocessor(mirror_left_hand=True, normalize_rotation=normalize_rotation)

    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    coords_flat = df[coord_cols].to_numpy(dtype=np.float32)
    coords_all = coords_flat.reshape(-1, 21, 3)
    handedness_list = df["handedness"].tolist() if "handedness" in df.columns else ["Right"] * len(df)
    labels = df["gesture"].to_numpy()
    groups = df["subject_id"].to_numpy()

    features = []
    observations = []

    for coords, handedness in zip(coords_all, handedness_list):
        feat = preprocessor.transform(coords, handedness=handedness)
        features.append(feat)

        obs = HandObservation(
            landmarks=coords,
            handedness=handedness,
            handedness_score=1.0,
            timestamp=0.0,
            frame_width=640,
            frame_height=480,
            palm_size=float(np.linalg.norm(coords[9] - coords[0])),
            hand_center=(0.5, 0.5),
        )
        observations.append(obs)

    return np.array(features, dtype=np.float32), labels, groups, observations


def compare_models(csv_path: str, n_splits: int = 5) -> None:
    """So sánh hiệu năng giữa Rule Baseline, KNN, Random Forest và SVM qua Subject-Level CV."""
    X, y, groups, observations = load_dataset_features(csv_path)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=splits)

    rule_engine = RuleStaticBaseline()
    models = {
        "Rule Baseline": None,
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Random Forest (n=100)": RandomForestClassifier(n_estimators=100, random_state=42),
        "RBF-SVM (C=10, gamma='scale')": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
    }

    results = {name: {"acc": [], "f1": []} for name in models}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X[train_idx])
        X_val_scaled = scaler.transform(X[val_idx])
        y_train, y_val = y[train_idx], y[val_idx]

        for name, clf in models.items():
            if clf is None:
                # Rule Baseline
                preds = [rule_engine.predict_observation(observations[i]).label for i in val_idx]
            else:
                clf.fit(X_train_scaled, y_train)
                preds = clf.predict(X_val_scaled)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)
            results[name]["acc"].append(acc)
            results[name]["f1"].append(f1)

    print("\n" + "=" * 65)
    print(f"{'MÔ HÌNH':<30} | {'ACCURACY (MEAN ± STD)':<18} | {'MACRO-F1':<12}")
    print("=" * 65)
    for name, metrics in results.items():
        mean_acc = np.mean(metrics["acc"])
        std_acc = np.std(metrics["acc"])
        mean_f1 = np.mean(metrics["f1"])
        print(f"{name:<30} | {mean_acc:.4f} ± {std_acc:.4f}     | {mean_f1:.4f}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare gesture recognition models across subject-independent folds.")
    parser.add_argument("--csv", type=str, default="data/raw/landmarks_dataset.csv", help="Path to landmark dataset CSV")
    parser.add_argument("--splits", type=int, default=5, help="Number of GroupKFold splits")
    args = parser.parse_args()

    compare_models(csv_path=args.csv, n_splits=args.splits)


if __name__ == "__main__":
    main()
