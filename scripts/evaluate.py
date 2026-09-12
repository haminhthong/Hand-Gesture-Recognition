"""Đánh giá mô hình đã huấn luyện trên tập dữ liệu kiểm thử độc lập."""

import argparse
import logging
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.classifier import StaticGestureClassifier

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


def evaluate_model(
    model_path: str,
    data_csv_path: str,
    splits_json_path: str = "",
) -> Dict[str, Any]:
    """Đánh giá model artifact trên tệp CSV dữ liệu (có thể lọc theo split test)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model artifact: {model_path}")
    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f"Không tìm thấy file csv: {data_csv_path}")

    classifier = StaticGestureClassifier(model_bundle_path=model_path)
    df = pd.read_csv(data_csv_path)

    if splits_json_path and os.path.exists(splits_json_path):
        import json
        with open(splits_json_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        test_subs = splits.get("test", [])
        if test_subs:
            df = df[df["subject_id"].isin(test_subs)].copy()
            logger.info("Đã lọc %d mẫu thuộc tập Test subjects: %s", len(df), test_subs)

    y_true = df["gesture"].tolist()
    y_pred = []
    rejected_count = 0

    # Vectorized landmark coordinate extraction
    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    coords_flat = df[coord_cols].to_numpy(dtype=np.float32)
    coords_all = coords_flat.reshape(-1, 21, 3)
    handedness_list = df["handedness"].tolist() if "handedness" in df.columns else ["Right"] * len(df)

    for coords, handedness in zip(coords_all, handedness_list):
        feat_63d = classifier.preprocessor.transform(coords, handedness=handedness)
        res = classifier.predict_features(feat_63d)
        y_pred.append(res.label)
        if res.rejected:
            rejected_count += 1

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    reject_rate = rejected_count / len(y_true) if len(y_true) > 0 else 0.0

    logger.info("=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ===")
    logger.info("Model: %s | Số mẫu kiểm thử: %d", model_path, len(y_true))
    logger.info("Accuracy: %.4f | Macro-F1: %.4f | Reject Rate: %.1f%%", acc, macro_f1, reject_rate * 100)
    logger.info("\nChi tiết phân loại:\n%s", classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "reject_rate": round(reject_rate, 4),
        "total_samples": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate static gesture classifier on independent test set.")
    parser.add_argument("--model", type=str, default="models/static_gesture_svm.joblib", help="Path to SVM model .joblib file")
    parser.add_argument("--csv", type=str, default="data/raw/landmarks_dataset.csv", help="Path to landmark dataset CSV")
    parser.add_argument("--splits-json", type=str, default="data/processed/splits.json", help="Path to splits.json file to filter test set")
    args = parser.parse_args()

    evaluate_model(args.model, args.csv, args.splits_json)


if __name__ == "__main__":
    main()
