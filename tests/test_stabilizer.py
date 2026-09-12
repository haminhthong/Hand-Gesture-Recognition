"""Tests cho GestureStabilizer (Time-Based Hysteresis)."""

from hand_gesture_controller.schemas import StaticPrediction
from hand_gesture_controller.stabilizer import GestureStabilizer


def make_pred(label: str, confidence: float = 0.9, rejected: bool = False) -> StaticPrediction:
    return StaticPrediction(
        label=label,
        confidence=confidence,
        probabilities={label: confidence},
        rejected=rejected,
        source="mock",
    )


def test_stabilizer_activation_dwell():
    # Activation dwell: 120ms (0.12s)
    stabilizer = GestureStabilizer(activation_dwell_ms=120.0, release_dwell_ms=100.0)

    # t = 0.0s: Cử chỉ Peace mới xuất hiện
    g1 = stabilizer.update(make_pred("Peace"), timestamp=0.0)
    assert g1.label == "NoAction"  # Chưa đủ 120ms

    # t = 0.08s: Duy trì 80ms
    g2 = stabilizer.update(make_pred("Peace"), timestamp=0.08)
    assert g2.label == "NoAction"  # Vẫn chưa đủ 120ms

    # t = 0.13s: Duy trì 130ms >= 120ms -> Kích hoạt!
    g3 = stabilizer.update(make_pred("Peace"), timestamp=0.13)
    assert g3.label == "Peace"


def test_stabilizer_release_dwell():
    # Release dwell: 100ms (0.10s)
    stabilizer = GestureStabilizer(activation_dwell_ms=120.0, release_dwell_ms=100.0)

    # Kích hoạt Peace
    stabilizer.update(make_pred("Peace"), timestamp=0.0)
    stabilizer.update(make_pred("Peace"), timestamp=0.15)
    assert stabilizer.update(make_pred("Peace"), timestamp=0.20).label == "Peace"

    # Mất Peace trong 50ms (< 100ms release dwell): vẫn giữ Peace (hysteresis chống mất tạm thời)
    g_lost_brief = stabilizer.update(make_pred("NoAction"), timestamp=0.25)
    assert g_lost_brief.label == "Peace"

    # Mất Peace tới 120ms (> 100ms release dwell) -> Trạng thái được giải phóng!
    g_released = stabilizer.update(make_pred("NoAction"), timestamp=0.33)
    assert g_released.label == "NoAction"


def test_stabilizer_jitter_noise_suppression():
    stabilizer = GestureStabilizer(activation_dwell_ms=120.0, release_dwell_ms=100.0)

    # Duy trì Select
    stabilizer.update(make_pred("Select"), timestamp=0.0)
    stabilizer.update(make_pred("Select"), timestamp=0.15)
    assert stabilizer.update(make_pred("Select"), timestamp=0.20).label == "Select"

    # 1 frame nhiễu xuất hiện "Stop" ở t=0.22s
    noise_frame = stabilizer.update(make_pred("Stop"), timestamp=0.22)
    # Stop chưa đủ 120ms activation dwell và Select chưa mất đủ 100ms release dwell
    # Do đó nhãn ổn định vẫn là Select!
    assert noise_frame.label == "Select"


def test_stabilizer_reset():
    stabilizer = GestureStabilizer()
    stabilizer.update(make_pred("Fist"), timestamp=0.0)
    stabilizer.update(make_pred("Fist"), timestamp=0.15)
    assert stabilizer.update(make_pred("Fist"), timestamp=0.20).label == "Fist"

    stabilizer.reset()
    assert stabilizer.update(make_pred("NoAction"), timestamp=0.21).label == "NoAction"
