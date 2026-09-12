"""Fixture tạo tập dữ liệu Landmark giả lập phục vụ kiểm thử tự động và CI/CD."""

import csv
import os
import random
import uuid
from typing import List, Optional

import numpy as np

GESTURES = ["Fist", "Select", "Options", "Stop", "Peace", "NoAction"]
SUBJECTS = [f"subject_{i:03d}" for i in range(1, 13)]  # 12 subjects
SESSIONS = ["session_01", "session_02"]


def get_base_landmarks(gesture: str) -> np.ndarray:
    """Tạo bộ tọa độ (21, 3) đại diện cho cử chỉ bàn tay."""
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.5, 0.8, 0.0]  # Wrist

    # Khớp MCP
    pts[1] = [0.42, 0.72, 0.0]
    pts[2] = [0.38, 0.65, 0.0]
    pts[3] = [0.36, 0.58, 0.0]

    pts[5] = [0.45, 0.55, 0.0]
    pts[6] = [0.45, 0.45, 0.0]
    pts[7] = [0.45, 0.38, 0.0]

    pts[9] = [0.50, 0.52, 0.0]
    pts[10] = [0.50, 0.42, 0.0]
    pts[11] = [0.50, 0.35, 0.0]

    pts[13] = [0.55, 0.55, 0.0]
    pts[14] = [0.55, 0.45, 0.0]
    pts[15] = [0.55, 0.38, 0.0]

    pts[17] = [0.60, 0.58, 0.0]
    pts[18] = [0.60, 0.50, 0.0]
    pts[19] = [0.60, 0.44, 0.0]

    if gesture == "Stop":
        pts[4] = [0.32, 0.50, 0.0]
        pts[8] = [0.45, 0.28, 0.0]
        pts[12] = [0.50, 0.25, 0.0]
        pts[16] = [0.55, 0.28, 0.0]
        pts[20] = [0.60, 0.35, 0.0]

    elif gesture == "Fist":
        pts[4] = [0.42, 0.60, 0.0]
        pts[8] = [0.46, 0.60, 0.0]
        pts[12] = [0.50, 0.60, 0.0]
        pts[16] = [0.54, 0.60, 0.0]
        pts[20] = [0.58, 0.62, 0.0]

    elif gesture == "Peace":
        pts[4] = [0.42, 0.60, 0.0]
        pts[8] = [0.44, 0.28, 0.0]
        pts[12] = [0.52, 0.28, 0.0]
        pts[16] = [0.54, 0.60, 0.0]
        pts[20] = [0.58, 0.62, 0.0]

    elif gesture == "Select":
        pts[4] = [0.46, 0.38, 0.0]
        pts[8] = [0.47, 0.37, 0.0]
        pts[12] = [0.50, 0.25, 0.0]
        pts[16] = [0.55, 0.28, 0.0]
        pts[20] = [0.60, 0.35, 0.0]

    elif gesture == "Options":
        pts[4] = [0.48, 0.38, 0.0]
        pts[8] = [0.48, 0.37, 0.0]
        pts[12] = [0.49, 0.38, 0.0]
        pts[16] = [0.55, 0.28, 0.0]
        pts[20] = [0.60, 0.35, 0.0]

    else:  # NoAction
        pts[4] = [0.40, 0.55, 0.0]
        pts[8] = [0.46, 0.40, 0.0]
        pts[12] = [0.50, 0.35, 0.0]
        pts[16] = [0.54, 0.50, 0.0]
        pts[20] = [0.58, 0.55, 0.0]

    return pts


def generate_fake_dataset(
    output_csv: str = "data/raw/landmarks_dataset.csv",
    samples_per_subject_gesture: int = 15,
    seed: int = 42,
    subjects: Optional[List[str]] = None,
) -> str:
    """Tạo file CSV chứa dữ liệu landmark giả lập."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    subj_list = subjects or SUBJECTS
    header = [
        "sample_id",
        "subject_id",
        "session_id",
        "sequence_id",
        "gesture",
        "frame_index",
        "timestamp_ms",
        "handedness",
        "camera_width",
        "camera_height",
        "device_id",
        "lighting",
    ] + [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]

    rows = []
    for sub in subj_list:
        for ses in SESSIONS:
            for g in GESTURES:
                base_pts = get_base_landmarks(g)
                for idx in range(samples_per_subject_gesture):
                    noise = np_rng.normal(0.0, 0.012, size=(21, 3)).astype(np.float32)
                    scale = rng.uniform(0.85, 1.15)
                    shift = [rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08), 0.0]

                    pts = (base_pts + noise) * scale + shift
                    sample_id = f"smp_{uuid.uuid4().hex[:12]}"
                    handedness = "Left" if rng.random() > 0.5 else "Right"

                    row = [
                        sample_id,
                        sub,
                        ses,
                        f"seq_{g.lower()}_01",
                        g,
                        idx,
                        idx * 33,
                        handedness,
                        640,
                        480,
                        "mock_cam",
                        "normal",
                    ]
                    for i in range(21):
                        row.extend([float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])])
                    rows.append(row)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    return output_csv


if __name__ == "__main__":
    generate_fake_dataset()
