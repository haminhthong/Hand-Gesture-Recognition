"""Wrapper tương thích cho CLI thu thập landmark trong package chính."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hand_gesture_controller.data_collection import (  # noqa: E402
    ALLOWED_GESTURES,
    CSV_HEADER,
    collect_landmarks,
    init_dataset_csv,
    main,
)

__all__ = [
    "ALLOWED_GESTURES",
    "CSV_HEADER",
    "collect_landmarks",
    "init_dataset_csv",
    "main",
]


if __name__ == "__main__":
    main()
