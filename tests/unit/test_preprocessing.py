"""Unit tests cho mô-đun preprocessing."""

import numpy as np
import pytest

from hand_gesture_controller.preprocessing import LandmarkPreprocessor


def test_preprocessor_shape_validation():
    preprocessor = LandmarkPreprocessor()
    invalid_coords = np.zeros((20, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"Kích thước coords phải là \(21, 3\)"):
        preprocessor.transform(invalid_coords)


def test_preprocessor_scale_translation_invariance():
    preprocessor = LandmarkPreprocessor(mirror_left_hand=False)

    coords_base = np.zeros((21, 3), dtype=np.float32)
    coords_base[0] = [1.0, 2.0, 3.0]  # Wrist
    coords_base[9] = [1.0, 4.0, 3.0]  # Middle MCP (distance = 2.0)
    coords_base[8] = [3.0, 2.0, 3.0]  # Index Tip

    feat1 = preprocessor.transform(coords_base)

    # Shift wrist by (+10, +20, +30) and scale by 5.0
    coords_transformed = coords_base * 5.0 + np.array([10.0, 20.0, 30.0], dtype=np.float32)
    feat2 = preprocessor.transform(coords_transformed)

    # Preprocessed features must be invariant under translation and uniform scaling
    np.testing.assert_allclose(feat1, feat2, rtol=1e-5, atol=1e-5)


def test_preprocessor_mirror_left_hand():
    preprocessor = LandmarkPreprocessor(mirror_left_hand=True)

    coords = np.zeros((21, 3), dtype=np.float32)
    coords[9] = [0.0, 1.0, 0.0]
    coords[8] = [0.5, 0.5, 0.0]  # Positive X for right hand

    feat_right = preprocessor.transform(coords, handedness="Right")
    feat_left = preprocessor.transform(coords, handedness="Left")

    # Left hand X coordinates should be mirrored (negated)
    # Index 8 X coordinate is at index 8*3 = 24
    assert feat_right[24] == pytest.approx(0.5)
    assert feat_left[24] == pytest.approx(-0.5)
