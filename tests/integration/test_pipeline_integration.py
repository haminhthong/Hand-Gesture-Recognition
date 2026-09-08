"""Integration tests kiểm tra luồng liên thông giữa Detector, Smoother, EventMapper và ObjectManager."""

from types import SimpleNamespace

from hand_gesture_controller.event_mapper import GestureEvent, GestureEventMapper
from hand_gesture_controller.gesture_detector import GestureDetector
from hand_gesture_controller.gesture_smoother import GestureSmoother
from hand_gesture_controller.object_manager import DraggableObject, DraggableObjectManager


def create_synthetic_fist_landmarks():
    """Tạo bộ điểm mốc giả lập cử chỉ Fist (gập 5 ngón tay)."""
    wrist = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    # Tất cả các đầu ngón nằm gần cổ tay (gập lại)
    landmarks_list = [wrist] + [SimpleNamespace(x=0.5, y=0.7, z=0.0) for _ in range(20)]
    return SimpleNamespace(landmark=landmarks_list)


def test_synthetic_pipeline_integration():
    detector = GestureDetector()
    smoother = GestureSmoother(window_size=3, minimum_votes=2)
    mapper = GestureEventMapper()
    manager = DraggableObjectManager()

    obj = DraggableObject(200, 200, 100, 100, (255, 0, 0), "TestObj")
    manager.add_object(obj)
    manager.visible = True

    landmarks = create_synthetic_fist_landmarks()

    # Chạy 3 khung hình giả lập
    for _ in range(3):
        raw_gesture, color = detector.detect_static_gesture(landmarks)
        smoothed_label, _ = smoother.update(raw_gesture, color)
        motion_res, _ = detector.detect_motion_gesture(landmarks)
        event = mapper.map_gesture_to_event(smoothed_label, motion_res[0])
        manager.update_event(landmarks, event, 640, 480)

    # Đảm bảo Fist được nhận diện thành công qua pipeline mà không crash
    assert raw_gesture == "Fist"
    assert smoothed_label == "Fist"
    assert event == GestureEvent.NONE
