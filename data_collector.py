"""File tương thích cho CLI thu thập landmark MediaPipe."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from hand_gesture_controller.data_collection import main

if __name__ == "__main__":
    main()
