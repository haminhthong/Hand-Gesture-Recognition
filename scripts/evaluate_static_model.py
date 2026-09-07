"""Đánh giá độc lập mô hình GestureModelBundle trên tập dữ liệu thử nghiệm."""

import argparse
import logging
import os
import sys
from typing import Any, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.recognition.static_predictor import StaticGesturePredictor

logger = logging.getLogger("evaluate_static_model")


def evaluate_model(
    model_bundle_path: str,
    test_csv_path: str,
) -> Dict[str, Any]:
    """Đánh giá model artifact trên tập CSV dữ liệu."""
    if not os.path.exists(model_bundle_path):
        raise FileNotFoundError(f"Không tìm thấy model artifact: {model_bundle_path}")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Không tìm thấy test csv: {test_csv_path}")

    predictor = StaticGesturePredictor(model_bundle_path=model_bundle_path)
    df = pd.read_csv(test_csv_path)

    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    y_true = df["gesture"].tolist()
    y_pred = []
    rejected_count = 0

    for _, row in df.iterrows():
        coords = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            coords[i, 0] = float(row[f"x{i}"])
            coords[i, 1] = float(row[f"y{i}"])
            coords[i, 2] = float(row[f"z{i}"])

        feat_63d = predictor.preprocessor.transform(coords, handedness=row.get("handedness", "Right"))
        res = predictor.predict_features(feat_63d)
        y_pred.append(res.label)
        if res.rejected:
            rejected_count += 1

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    reject_rate = rejected_count / len(y_true) if len(y_true) > 0 else 0.0

    actionable = ["Select", "Options", "Stop", "Peace"]
    false_actions = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "NoAction" and yp in actionable)
    no_actions_total = sum(1 for yt in y_true if yt == "NoAction")
    false_action_rate = (false_actions / no_actions_total) if no_actions_total > 0 else 0.0

    logger.info("=== BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ARTIFACT ===")
    logger.info("Model: %s | Test Samples: %d", model_bundle_path, len(y_true))
    logger.info("Macro-F1: %.4f | Balanced Accuracy: %.4f | Reject Rate: %.2f%%", macro_f1, bal_acc, reject_rate * 100)
    logger.info("False Action Rate (NoAction -> Action): %.2f%%", false_action_rate * 100)
    logger.info("\nChi tiết phân loại:\n%s", classification_report(y_true, y_pred, zero_division=0))

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "reject_rate": reject_rate,
        "false_action_rate": false_action_rate,
        "total_samples": len(y_true),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Đánh giá mô hình artifact.")
    parser.add_argument("--model", type=str, default="models/static_gesture_svm_v1.joblib", help="Đường dẫn model .joblib")
    parser.add_argument("--test-csv", type=str, required=True, help="Đường dẫn test dataset CSV")
    args = parser.parse_args()

    evaluate_model(args.model, args.test_csv)


if __name__ == "__main__":
    main()
