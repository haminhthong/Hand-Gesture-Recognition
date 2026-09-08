"""Unit tests xác thực tính năng Chuẩn hóa xoay mặt phẳng (In-plane Rotation Normalization) của LandmarkPreprocessor."""

import numpy as np
import pytest

from hand_gesture_controller.preprocessing import LandmarkPreprocessor


def test_rotation_normalization_aligns_middle_mcp_vertically():
    """Kiểm tra khi xoay bàn tay theo các góc khác nhau, preprocessor đưa vector Cổ tay -> Middle MCP về thẳng đứng."""
    preprocessor = LandmarkPreprocessor(mirror_left_hand=False, normalize_rotation=True)

    # Khởi tạo bàn tay cơ sở: Cổ tay tại (0,0,0), Middle MCP tại (0, 1, 0)
    base_coords = np.zeros((21, 3), dtype=np.float32)
    base_coords[0] = [0.0, 0.0, 0.0]
    base_coords[9] = [0.0, -1.0, 0.0]  # Thẳng đứng lên trên (-Y)
    base_coords[8] = [0.2, -1.2, 0.0]  # Index tip

    # Xoay bàn tay cơ sở đi 45 độ (pi/4)
    theta = np.pi / 4.0
    rot_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ], dtype=np.float32)

    rotated_coords = base_coords @ rot_matrix.T

    feat_base = preprocessor.transform(base_coords)
    feat_rotated = preprocessor.transform(rotated_coords)

    # Sau khi normalize rotation, hai vector đặc trưng phải tương đương nhau
    np.testing.assert_allclose(feat_base, feat_rotated, rtol=1e-4, atol=1e-4)


def test_rotation_normalization_disabled_by_default():
    """Mặc định normalize_rotation là False, giữ nguyên hướng xoay."""
    preprocessor = LandmarkPreprocessor(mirror_left_hand=False, normalize_rotation=False)

    coords = np.zeros((21, 3), dtype=np.float32)
    coords[0] = [0.0, 0.0, 0.0]
    coords[9] = [1.0, 0.0, 0.0]  # Nằm ngang (+X)

    feat = preprocessor.transform(coords)
    # Middle MCP X coordinate (index 9*3 = 27) vẫn là 1.0 (palm_size = 1.0)
    assert feat[27] == pytest.approx(1.0)
    assert feat[28] == pytest.approx(0.0)
