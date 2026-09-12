"""Điểm khởi chạy chính (entrypoint) của ứng dụng Hand Gesture HCI Controller."""

import os
import sys

# Đưa thư mục src vào sys.path để khởi chạy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hand_gesture_controller.app import main

if __name__ == "__main__":
    main()
