"""Công cụ phân chia Dataset theo đối tượng (Subject-Level Split: Train / Val / Locked Test) ngăn ngừa data leakage."""

import argparse
import hashlib
import json
import logging
import os
import random
from typing import Dict, List

import pandas as pd

logger = logging.getLogger("create_splits")


def create_subject_splits(
    csv_path: str,
    output_splits_path: str = "data/processed/splits.json",
    output_manifest_path: str = "data/processed/manifest.json",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Tạo phân chia Train / Val / Locked Test ở cấp độ đối tượng (subject-level)."""
    if not 0.0 < val_ratio < 1.0 or not 0.0 < test_ratio < 1.0:
        raise ValueError("val_ratio và test_ratio phải nằm trong khoảng (0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("Tổng val_ratio và test_ratio phải nhỏ hơn 1.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    subjects = sorted(df["subject_id"].unique().tolist())
    n_subjects = len(subjects)

    if n_subjects < 3:
        raise ValueError(f"Cần ít nhất 3 subjects để chia Train/Val/Test, hiện chỉ có {n_subjects}.")

    rng = random.Random(seed)
    shuffled_subjects = subjects.copy()
    rng.shuffle(shuffled_subjects)

    n_test = max(1, int(round(n_subjects * test_ratio)))
    n_val = max(1, int(round(n_subjects * val_ratio)))
    n_train = n_subjects - n_val - n_test

    if n_train < 1:
        n_train = 1
        if n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1

    test_subjects = sorted(shuffled_subjects[:n_test])
    val_subjects = sorted(shuffled_subjects[n_test : n_test + n_val])
    train_subjects = sorted(shuffled_subjects[n_test + n_val :])

    # Kiểm tra bất biến tính rời rạc
    assert not (set(train_subjects) & set(val_subjects)), "Lỗi: Train và Val giao nhau!"
    assert not (set(train_subjects) & set(test_subjects)), "Lỗi: Train và Test giao nhau!"
    assert not (set(val_subjects) & set(test_subjects)), "Lỗi: Val và Test giao nhau!"

    splits = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
        "seed": seed,
        "ratios": {"val": val_ratio, "test": test_ratio},
    }

    # Tính mã băm SHA256 của dataset CSV để truy vết dữ liệu (Data Lineage)
    hasher = hashlib.sha256()
    with open(csv_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    csv_hash = hasher.hexdigest()

    manifest = {
        "dataset_csv": os.path.abspath(csv_path),
        "csv_sha256": csv_hash,
        "total_samples": len(df),
        "total_subjects": n_subjects,
        "train_samples": int(df["subject_id"].isin(train_subjects).sum()),
        "val_samples": int(df["subject_id"].isin(val_subjects).sum()),
        "test_samples": int(df["subject_id"].isin(test_subjects).sum()),
        "splits": splits,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_splits_path)), exist_ok=True)
    with open(output_splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)

    os.makedirs(os.path.dirname(os.path.abspath(output_manifest_path)), exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    logger.info("=== ĐÃ TẠO SUBJECT-LEVEL SPLITS ===")
    logger.info("Train (%d subjects): %s", len(train_subjects), train_subjects)
    logger.info("Val   (%d subjects): %s", len(val_subjects), val_subjects)
    logger.info("Test  (%d subjects): %s", len(test_subjects), test_subjects)
    logger.info("Đã lưu: %s và %s", output_splits_path, output_manifest_path)

    return splits


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Tạo subject-level splits Train/Val/Test.")
    parser.add_argument("--csv", type=str, default="data/raw/landmarks_dataset.csv", help="Đường dẫn dataset CSV")
    parser.add_argument("--splits-out", type=str, default="data/processed/splits.json", help="File xuất splits.json")
    parser.add_argument("--manifest-out", type=str, default="data/processed/manifest.json", help="File xuất manifest.json")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Tỉ lệ validation")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Tỉ lệ test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    create_subject_splits(
        csv_path=args.csv,
        output_splits_path=args.splits_out,
        output_manifest_path=args.manifest_out,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
