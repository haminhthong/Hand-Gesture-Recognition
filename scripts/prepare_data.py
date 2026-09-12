"""Chuẩn bị và kiểm tra dữ liệu: Kiểm tra tính toàn vẹn và chia tập Train/Val/Test theo subject_id."""

import argparse
import json
import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_data")


def validate_dataset(csv_path: str) -> pd.DataFrame:
    """Kiểm tra tính toàn vẹn của tệp dữ liệu landmark."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("Tệp dataset rỗng.")

    required_cols = ["sample_id", "subject_id", "session_id", "gesture", "handedness"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu các cột bắt buộc: {missing_cols}")

    coord_cols = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]
    missing_coords = [c for c in coord_cols if c not in df.columns]
    if missing_coords:
        raise ValueError(f"Thiếu {len(missing_coords)} cột tọa độ landmark.")

    duplicates = int(df["sample_id"].duplicated().sum())
    if duplicates > 0:
        logger.warning("Phát hiện %d sample_id trùng lặp, đang loại bỏ...", duplicates)
        df = df.drop_duplicates(subset=["sample_id"]).copy()

    coords = df[coord_cols].to_numpy(dtype=np.float32)
    nan_count = int(np.isnan(coords).sum())
    inf_count = int(np.isinf(coords).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Dữ liệu tọa độ chứa giá trị không hợp lệ: NaN={nan_count}, Inf={inf_count}")

    logger.info("Dữ liệu hợp lệ: %d mẫu, %d subjects, phân bố cử chỉ: %s",
                len(df), df["subject_id"].nunique(), dict(df["gesture"].value_counts()))
    return df


def split_by_subject(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Phân chia tập train/val/test hoàn toàn rời rạc theo subject_id (chống rò rỉ dữ liệu)."""
    subjects = sorted(df["subject_id"].unique().tolist())
    n_subs = len(subjects)
    if n_subs < 3:
        raise ValueError(f"Cần ít nhất 3 subjects để chia Train/Val/Test độc lập, hiện có: {n_subs}")

    rng = random.Random(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(n_subs * test_ratio)))
    n_val = max(1, int(round(n_subs * val_ratio)))
    n_train = n_subs - n_val - n_test

    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    test_subs = sorted(shuffled[:n_test])
    val_subs = sorted(shuffled[n_test : n_test + n_val])
    train_subs = sorted(shuffled[n_test + n_val :])

    # Đảm bảo tính rời rạc
    assert not (set(train_subs) & set(val_subs)), "Lỗi: Train và Val bị trùng subject!"
    assert not (set(train_subs) & set(test_subs)), "Lỗi: Train và Test bị trùng subject!"
    assert not (set(val_subs) & set(test_subs)), "Lỗi: Val và Test bị trùng subject!"

    return {
        "train": train_subs,
        "val": val_subs,
        "test": test_subs,
    }


def prepare_dataset(
    csv_path: str = "data/raw/landmarks_dataset.csv",
    output_splits_path: str = "data/processed/splits.json",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Kiểm tra và xuất phân chia dữ liệu theo đối tượng."""
    df = validate_dataset(csv_path)
    splits = split_by_subject(df, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    os.makedirs(os.path.dirname(os.path.abspath(output_splits_path)), exist_ok=True)
    with open(output_splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    logger.info("Đã tạo phân chia subject-independent:")
    logger.info("  Train (%d subjects): %s", len(splits["train"]), splits["train"])
    logger.info("  Val   (%d subjects): %s", len(splits["val"]), splits["val"])
    logger.info("  Test  (%d subjects): %s", len(splits["test"]), splits["test"])
    logger.info("Đã lưu splits tới: %s", output_splits_path)

    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate landmark dataset and create subject-independent splits.")
    parser.add_argument("--csv", type=str, default="data/raw/landmarks_dataset.csv", help="Path to landmark dataset CSV")
    parser.add_argument("--output-splits", type=str, default="data/processed/splits.json", help="Path to output splits.json")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation subject ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test subject ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject shuffling")
    args = parser.parse_args()

    prepare_dataset(
        csv_path=args.csv,
        output_splits_path=args.output_splits,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
