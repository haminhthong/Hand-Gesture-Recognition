"""Tạo tập dữ liệu Landmark tổng hợp chất lượng cao phục vụ huấn luyện mô hình ban đầu và kiểm thử CI/CD."""

import csv
import math
import os
import random
import uuid
import numpy as np

OUTPUT_CSV = "data/raw/landmarks_dataset.csv"
GESTURES = ["Fist", "Select", "Options", "Stop", "Peace", "NoAction"]
SUBJECTS = [f"subject_{i:03d}" for i in range(1, 13)]  # 12 subjects
SESSIONS = ["session_01", "session_02"]


def get_base_landmarks(gesture: str) -> np.ndarray:
    """Tạo bộ tọa độ (21, 3) đại diện cho cử chỉ bàn tay."""
    pts = np.zeros((21, 3), dtype=np.float32)
    # Cổ tay tại gốc tham chiếu
    pts[0] = [0.5, 0.8, 0.0]

    # Các khớp MCP gốc ngón
    pts[1] = [0.42, 0.72, 0.0]  # thumb cmc
    pts[2] = [0.38, 0.65, 0.0]  # thumb mcp
    pts[3] = [0.36, 0.58, 0.0]  # thumb ip

    pts[5] = [0.45, 0.55, 0.0]  # index mcp
    pts[6] = [0.45, 0.45, 0.0]  # index pip
    pts[7] = [0.45, 0.38, 0.0]  # index dip

    pts[9] = [0.50, 0.52, 0.0]  # middle mcp
    pts[10] = [0.50, 0.42, 0.0] # middle pip
    pts[11] = [0.50, 0.35, 0.0] # middle dip

    pts[13] = [0.55, 0.55, 0.0] # ring mcp
    pts[14] = [0.55, 0.45, 0.0] # ring pip
    pts[15] = [0.55, 0.38, 0.0] # ring dip

    pts[17] = [0.60, 0.58, 0.0] # pinky mcp
    pts[18] = [0.60, 0.50, 0.0] # pinky pip
    pts[19] = [0.60, 0.44, 0.0] # pinky dip

    if gesture == "Stop":
        # 5 ngón xòe thẳng
        pts[4] = [0.32, 0.50, 0.0]  # thumb tip
        pts[8] = [0.45, 0.28, 0.0]  # index tip
        pts[12] = [0.50, 0.25, 0.0] # middle tip
        pts[16] = [0.55, 0.28, 0.0] # ring tip
        pts[20] = [0.60, 0.35, 0.0] # pinky tip

    elif gesture == "Fist":
        # Tất cả ngón gập chặt
        pts[4] = [0.42, 0.60, 0.0]
        pts[8] = [0.46, 0.60, 0.0]
        pts[12] = [0.50, 0.60, 0.0]
        pts[16] = [0.54, 0.60, 0.0]
        pts[20] = [0.58, 0.62, 0.0]

    elif gesture == "Peace":
        # Trỏ và giữa duỗi thẳng, các ngón khác gập
        pts[4] = [0.42, 0.65, 0.0]
        pts[8] = [0.44, 0.28, 0.0]
        pts[12] = [0.52, 0.27, 0.0]
        pts[16] = [0.55, 0.60, 0.0]
        pts[20] = [0.58, 0.62, 0.0]

    elif gesture == "Select":
        # Đầu ngón cái và đầu ngón trỏ chụm sát nhau
        pinch_pt = [0.46, 0.45, 0.0]
        pts[4] = [pinch_pt[0] - 0.01, pinch_pt[1], 0.0]
        pts[8] = [pinch_pt[0] + 0.01, pinch_pt[1], 0.0]
        # Giữa, áp út, út gập
        pts[12] = [0.52, 0.60, 0.0]
        pts[16] = [0.55, 0.60, 0.0]
        pts[20] = [0.58, 0.62, 0.0]

    elif gesture == "Options":
        # Cái, trỏ, giữa cùng chụm nhau
        pinch_pt = [0.48, 0.44, 0.0]
        pts[4] = [pinch_pt[0] - 0.015, pinch_pt[1], 0.0]
        pts[8] = [pinch_pt[0], pinch_pt[1] - 0.015, 0.0]
        pts[12] = [pinch_pt[0] + 0.015, pinch_pt[1], 0.0]
        pts[16] = [0.55, 0.60, 0.0]
        pts[20] = [0.58, 0.62, 0.0]

    else:  # NoAction: Tư thế trung gian, nửa mở, không rõ ràng
        sub_type = random.choice(["half_open", "partial_pinch", "loose_hand"])
        if sub_type == "half_open":
            pts[4] = [0.38, 0.58, 0.0]
            pts[8] = [0.46, 0.42, 0.0]
            pts[12] = [0.50, 0.40, 0.0]
            pts[16] = [0.55, 0.48, 0.0]
            pts[20] = [0.59, 0.52, 0.0]
        elif sub_type == "partial_pinch":
            # Pinch chưa chạm nhau
            pts[4] = [0.41, 0.50, 0.0]
            pts[8] = [0.49, 0.42, 0.0]
            pts[12] = [0.51, 0.50, 0.0]
            pts[16] = [0.55, 0.58, 0.0]
            pts[20] = [0.58, 0.60, 0.0]
        else:
            for tip in [4, 8, 12, 16, 20]:
                pts[tip] = [0.35 + random.uniform(0, 0.3), 0.40 + random.uniform(0, 0.25), random.uniform(-0.05, 0.05)]

    return pts


def generate_dataset(samples_per_sequence: int = 15) -> None:
    """Tạo file CSV dataset chuẩn hóa với đầy đủ 12 subjects, 6 nhãn và sequence."""
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
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
    ] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    total_rows = 0
    with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for subject in SUBJECTS:
            for session in SESSIONS:
                for gesture in GESTURES:
                    # 2 sequences/bursts cho mỗi tổ hợp (subject, session, gesture)
                    for seq_idx in range(1, 3):
                        seq_id = f"seq_{seq_idx:03d}"
                        base_pts = get_base_landmarks(gesture)

                        # Chọn tay trái hoặc phải
                        handedness = "Right" if random.random() > 0.3 else "Left"

                        for frame_idx in range(samples_per_sequence):
                            sample_id = f"{subject}_{session}_{seq_id}_{frame_idx:04d}_{uuid.uuid4().hex[:6]}"
                            t_ms = 100000.0 + total_rows * 150.0

                            # Thêm biến thiên tự nhiên: noise, tỉ lệ, dịch chuyển
                            scale = random.uniform(0.92, 1.08)
                            shift_x = random.uniform(-0.04, 0.04)
                            shift_y = random.uniform(-0.04, 0.04)
                            noise = np.random.normal(0, 0.005, (21, 3)).astype(np.float32)

                            coords = (base_pts * scale) + [shift_x, shift_y, 0.0] + noise

                            # Nếu là tay trái, phản chiếu X
                            if handedness == "Left":
                                coords[:, 0] = 1.0 - coords[:, 0]

                            row = [
                                sample_id,
                                subject,
                                session,
                                seq_id,
                                gesture,
                                frame_idx,
                                round(t_ms, 2),
                                handedness,
                                640,
                                480,
                                "cam_01",
                                "normal",
                            ]
                            for x, y, z in coords:
                                row.extend([round(float(x), 5), round(float(y), 5), round(float(z), 5)])

                            writer.writerow(row)
                            total_rows += 1

    print(f"Generated {total_rows} samples successfully at: {OUTPUT_CSV}")


if __name__ == "__main__":
    generate_dataset()
