"""Công cụ CLI thu thập Landmark MediaPipe theo burst/sequence và lấy mẫu giãn cách thời gian (100-200ms)."""

import argparse
import csv
import logging
import os
import sys
import time
import uuid

import cv2

# Thêm đường dẫn src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.perception.hand_detector import HandDetector

logger = logging.getLogger("collect_data")

# Production Vocabulary v1
ALLOWED_GESTURES = (
    "Fist",
    "Select",
    "Options",
    "Stop",
    "Peace",
    "NoAction",
    "Unknown",
)

CSV_HEADER = [
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


def init_dataset_csv(csv_path: str) -> None:
    """Khởi tạo tệp CSV chứa tiêu đề chuẩn nếu chưa tồn tại."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def collect_landmarks(
    subject_id: str,
    session_id: str,
    label: str,
    output_csv: str = "data/raw/landmarks_dataset.csv",
    camera_index: int = 0,
    max_samples: int = 100,
    sampling_interval_ms: float = 150.0,
    lighting: str = "normal",
    device_id: str = "cam_01",
) -> None:
    """Thu thập mẫu landmark theo chuỗi (sequence/burst) kèm lấy mẫu giãn cách thời gian."""
    if not subject_id or not subject_id.strip():
        raise ValueError("Mã người tham gia (--subject-id) không được để trống.")
    if not session_id or not session_id.strip():
        raise ValueError("Mã phiên (--session-id) không được để trống.")
    if max_samples <= 0:
        raise ValueError("Số lượng mẫu (--samples) phải lớn hơn 0.")

    init_dataset_csv(output_csv)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không thể mở camera index {camera_index}.")

    detector = HandDetector(detectionCon=0.7, maxHands=1)
    logger.info("=== BẮT ĐẦU THU THẬP DỮ LIỆU CỬ CHỈ: '%s' ===", label)
    logger.info("Subject: %s | Session: %s | Mục tiêu: %d mẫu | Giãn cách: %.1f ms",
                subject_id, session_id, max_samples, sampling_interval_ms)
    logger.info("Phím: 'S'=Bắt đầu/Dừng burst hiện tại, 'N'=Bắt đầu sequence mới, 'Q'=Thoát.")

    recording = False
    samples_count = 0
    frame_index = 0
    sequence_count = 1
    current_seq_id = f"seq_{sequence_count:03d}"
    last_sample_time_ms = 0.0

    try:
        output_file = open(output_csv, mode="a", newline="", encoding="utf-8")
        writer = csv.writer(output_file)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.error("Mất tín hiệu camera.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            frame_index += 1
            now_ms = time.time() * 1000.0

            observation = detector.process(frame, timestamp=now_ms / 1000.0)

            if observation is not None:
                detector.draw_landmarks(frame, observation)

                # Kiểm tra điều kiện ghi mẫu: đang bật recording và đã trôi qua sampling_interval_ms
                if recording and (now_ms - last_sample_time_ms >= sampling_interval_ms):
                    sample_id = f"{subject_id}_{session_id}_{current_seq_id}_{samples_count + 1:04d}_{uuid.uuid4().hex[:6]}"
                    last_sample_time_ms = now_ms

                    row = [
                        sample_id,
                        subject_id,
                        session_id,
                        current_seq_id,
                        label,
                        frame_index,
                        round(now_ms, 2),
                        observation.handedness,
                        frame_w,
                        frame_h,
                        device_id,
                        lighting,
                    ]
                    for x, y, z in observation.landmarks:
                        row.extend([x, y, z])

                    writer.writerow(row)
                    output_file.flush()
                    samples_count += 1

                    if samples_count >= max_samples:
                        logger.info("Hoàn tất thu thập %d mẫu cho nhãn '%s'!", max_samples, label)
                        break

            status_text = f"BURST RECORDING ({samples_count}/{max_samples})" if recording else "IDLE (PRESS 'S')"
            status_color = (0, 255, 0) if recording else (0, 255, 255)

            cv2.putText(
                frame,
                f"Label: {label} | Seq: {current_seq_id} | Status: {status_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            cv2.imshow("Landmark Burst Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("s"), ord("S")):
                recording = not recording
                logger.info("Trạng thái ghi mẫu: %s", "BẬT" if recording else "TẮT")
            elif key in (ord("n"), ord("N")):
                sequence_count += 1
                current_seq_id = f"seq_{sequence_count:03d}"
                logger.info("Chuyển sang sequence mới: %s", current_seq_id)

    finally:
        output_file.close()
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Thu thập landmark theo burst/sequence.")
    parser.add_argument("--subject-id", type=str, required=True, help="Mã người tham gia (ví dụ: subject_001)")
    parser.add_argument("--session-id", type=str, required=True, help="Mã phiên (ví dụ: session_01)")
    parser.add_argument("--label", type=str, required=True, choices=ALLOWED_GESTURES, help="Nhãn cử chỉ")
    parser.add_argument("--samples", type=int, default=100, help="Số mẫu mục tiêu")
    parser.add_argument("--interval", type=float, default=150.0, help="Khoảng cách lấy mẫu ms (100-200ms)")
    parser.add_argument("--output", type=str, default="data/raw/landmarks_dataset.csv", help="File output CSV")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera")
    parser.add_argument("--lighting", type=str, default="normal", help="Điều kiện ánh sáng")
    args = parser.parse_args()

    collect_landmarks(
        subject_id=args.subject_id,
        session_id=args.session_id,
        label=args.label,
        output_csv=args.output,
        camera_index=args.camera,
        max_samples=args.samples,
        sampling_interval_ms=args.interval,
        lighting=args.lighting,
    )


if __name__ == "__main__":
    main()
