"""Công cụ kiểm toán chất lượng dataset (Data Audit) trước khi huấn luyện mô hình ML."""

import argparse
import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("audit_data")


def audit_dataset(
    csv_path: str,
    splits_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Kiểm tra toàn diện tính toàn vẹn và phân bố của tập dữ liệu landmark."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    if total_rows == 0:
        raise ValueError("Tệp dataset rỗng (0 dòng).")

    # 1. Kiểm tra các cột bắt buộc
    required_meta = ["sample_id", "subject_id", "session_id", "gesture", "handedness"]
    missing_meta = [c for c in required_meta if c not in df.columns]
    if missing_meta:
        raise ValueError(f"Dataset thiếu các cột bắt buộc: {missing_meta}")

    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    missing_coords = [c for c in coord_cols if c not in df.columns]
    if missing_coords:
        raise ValueError(f"Dataset thiếu các cột tọa độ landmark: {len(missing_coords)} cột.")

    # 2. Kiểm tra trùng lặp sample_id
    duplicate_samples = int(df["sample_id"].duplicated().sum())

    # 3. Kiểm tra NaN / Inf trong tọa độ
    coords_matrix = df[coord_cols].to_numpy(dtype=np.float32)
    nan_count = int(np.isnan(coords_matrix).sum())
    inf_count = int(np.isinf(coords_matrix).sum())

    # 4. Thống kê phân bố
    subjects = df["subject_id"].unique().tolist()
    sessions = df["session_id"].unique().tolist()
    class_counts = df["gesture"].value_counts().to_dict()
    subject_counts = df["subject_id"].value_counts().to_dict()

    # 5. Kiểm tra tính rời rạc của các tập Train/Val/Test (Disjointness)
    split_leakage = False
    if splits_path and os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)
        train_subs = set(splits.get("train", []))
        val_subs = set(splits.get("val", []))
        test_subs = set(splits.get("test", []))

        if (train_subs & val_subs) or (train_subs & test_subs) or (val_subs & test_subs):
            split_leakage = True
            logger.error("Phát hiện Data Leakage giữa các tập split: Train/Val/Test không rời rạc!")

    audit_summary = {
        "csv_path": csv_path,
        "total_samples": total_rows,
        "unique_subjects": len(subjects),
        "subjects": subjects,
        "unique_sessions": len(sessions),
        "class_distribution": class_counts,
        "subject_distribution": subject_counts,
        "duplicate_sample_ids": duplicate_samples,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "split_leakage": split_leakage,
        "is_healthy": (nan_count == 0 and inf_count == 0 and duplicate_samples == 0 and not split_leakage),
    }

    logger.info("=== KẾT QUẢ KIỂM TOÁN DỮ LIỆU ===")
    logger.info("Tổng số mẫu: %d | Số đối tượng: %d | Số phiên: %d", total_rows, len(subjects), len(sessions))
    logger.info("Phân bố lớp: %s", class_counts)
    logger.info("NaN/Inf: %d / %d | Duplicate IDs: %d", nan_count, inf_count, duplicate_samples)
    logger.info("Trạng thái toàn vẹn: %s", "ĐẠT (HEALTHY)" if audit_summary["is_healthy"] else "KHÔNG ĐẠT")

    if not audit_summary["is_healthy"]:
        raise ValueError(f"Kiểm toán thất bại: NaN={nan_count}, Inf={inf_count}, Dup={duplicate_samples}, Leakage={split_leakage}")

    return audit_summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Kiểm toán chất lượng dữ liệu dataset landmark.")
    parser.add_argument("--csv", type=str, default="data/raw/landmarks_dataset.csv", help="Đường dẫn dataset CSV")
    parser.add_argument("--splits", type=str, default="data/processed/splits.json", help="Đường dẫn file splits.json (nếu có)")
    args = parser.parse_args()

    audit_dataset(csv_path=args.csv, splits_path=args.splits if os.path.exists(args.splits) else None)


if __name__ == "__main__":
    main()
