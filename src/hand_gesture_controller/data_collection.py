"""CLI thu thập landmark MediaPipe theo schema dataset của dự án."""

import argparse
import csv
import logging
import os
import time
import uuid

import cv2

from .perception.hand_detector import HandDetector

logger = logging.getLogger("collect_data")

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
    """Tạo thư mục và header CSV nếu tệp chưa tồn tại."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as output_file:
            csv.writer(output_file).writerow(CSV_HEADER)


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
    """Thu thập mẫu landmark theo sequence và lấy mẫu cách nhau theo thời gian."""
    if not subject_id or not subject_id.strip():
        raise ValueError("Mã người tham gia (--subject-id) không được để trống.")
    if not session_id or not session_id.strip():
        raise ValueError("Mã phiên (--session-id) không được để trống.")
    if label not in ALLOWED_GESTURES:
        raise ValueError(f"Nhãn không hợp lệ: {label!r}. Chọn một trong {ALLOWED_GESTURES}.")
    if camera_index < 0:
        raise ValueError("Chỉ số camera phải lớn hơn hoặc bằng 0.")
    if max_samples <= 0:
        raise ValueError("Số lượng mẫu (--samples) phải lớn hơn 0.")
    if sampling_interval_ms <= 0:
        raise ValueError("Khoảng cách lấy mẫu (--interval) phải lớn hơn 0.")

    init_dataset_csv(output_csv)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không thể mở camera index {camera_index}.")

    detector = HandDetector(detectionCon=0.7, maxHands=1)
    recording = False
    samples_count = 0
    frame_index = 0
    sequence_count = 1
    current_sequence_id = f"seq_{sequence_count:03d}"
    last_sample_time_ms = 0.0

    logger.info("=== BẮT ĐẦU THU THẬP CỬ CHỈ: '%s' ===", label)
    logger.info(
        "Subject: %s | Session: %s | Mục tiêu: %d mẫu | Giãn cách: %.1f ms",
        subject_id,
        session_id,
        max_samples,
        sampling_interval_ms,
    )

    try:
        with open(output_csv, "a", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    logger.error("Không thể đọc khung hình từ camera.")
                    break

                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                frame_index += 1
                now_ms = time.time() * 1000.0
                observation = detector.process(frame, timestamp=now_ms / 1000.0)

                if observation is not None:
                    detector.draw_landmarks(frame, observation)
                    can_sample = recording and (
                        now_ms - last_sample_time_ms >= sampling_interval_ms
                    )
                    if can_sample:
                        sample_id = (
                            f"{subject_id}_{session_id}_{current_sequence_id}_"
                            f"{samples_count + 1:04d}_{uuid.uuid4().hex[:6]}"
                        )
                        row = [
                            sample_id,
                            subject_id,
                            session_id,
                            current_sequence_id,
                            label,
                            frame_index,
                            round(now_ms, 2),
                            observation.handedness,
                            frame_width,
                            frame_height,
                            device_id,
                            lighting,
                        ]
                        for x, y, z in observation.landmarks:
                            row.extend([x, y, z])
                        writer.writerow(row)
                        output_file.flush()
                        samples_count += 1
                        last_sample_time_ms = now_ms

                        if samples_count >= max_samples:
                            logger.info("Hoàn tất thu thập %d mẫu.", max_samples)
                            break

                status = (
                    f"BURST RECORDING ({samples_count}/{max_samples})"
                    if recording
                    else "IDLE (PRESS 'S')"
                )
                cv2.putText(
                    frame,
                    f"Label: {label} | Seq: {current_sequence_id} | {status}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if recording else (0, 255, 255),
                    2,
                )
                cv2.imshow("Landmark Burst Collector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("s"), ord("S")):
                    recording = not recording
                elif key in (ord("n"), ord("N")):
                    sequence_count += 1
                    current_sequence_id = f"seq_{sequence_count:03d}"
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Điểm nhập CLI thu thập landmark."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Thu thập landmark theo burst/sequence.")
    parser.add_argument("--subject-id", required=True, help="Mã người tham gia")
    parser.add_argument("--session-id", required=True, help="Mã phiên thu thập")
    parser.add_argument("--label", required=True, choices=ALLOWED_GESTURES, help="Nhãn cử chỉ")
    parser.add_argument("--samples", type=int, default=100, help="Số mẫu mục tiêu")
    parser.add_argument("--interval", type=float, default=150.0, help="Khoảng cách lấy mẫu ms")
    parser.add_argument("--output", default="data/raw/landmarks_dataset.csv", help="CSV output")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera")
    parser.add_argument("--lighting", default="normal", help="Điều kiện ánh sáng")
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
