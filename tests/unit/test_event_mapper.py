"""Unit tests cho mô-đun event_mapper."""

from hand_gesture_controller.event_mapper import GestureEvent, GestureEventMapper


def test_event_mapper_motion_gestures():
    mapper = GestureEventMapper()
    assert mapper.map_gesture_to_event("Select", "On/Off") == GestureEvent.TOGGLE_CANVAS
    assert mapper.map_gesture_to_event("Stop", "SOS") == GestureEvent.EMERGENCY_SOS


def test_event_mapper_static_drag_sequence():
    mapper = GestureEventMapper()

    # Initial Select -> START_DRAG
    evt1 = mapper.map_gesture_to_event("Select", "Still")
    assert evt1 == GestureEvent.START_DRAG
    assert mapper.is_dragging is True

    # Subsequent Select -> DRAG
    evt2 = mapper.map_gesture_to_event("Select", "Still")
    assert evt2 == GestureEvent.DRAG
    assert mapper.is_dragging is True

    # Transition away from Select -> STOP_DRAG
    evt3 = mapper.map_gesture_to_event("Fist", "Still")
    assert evt3 == GestureEvent.STOP_DRAG
    assert mapper.is_dragging is False


def test_event_mapper_actions():
    mapper = GestureEventMapper()
    assert mapper.map_gesture_to_event("Options", "Still") == GestureEvent.CHANGE_COLOR
    assert mapper.map_gesture_to_event("Stop", "Still") == GestureEvent.DELETE_OBJECT
    assert mapper.map_gesture_to_event("Peace", "Still") == GestureEvent.OPEN_MENU
    assert mapper.map_gesture_to_event("Fist", "Still") == GestureEvent.NONE


def test_event_mapper_accepts_yaml_style_cooldown_keys():
    mapper = GestureEventMapper({"change_color": 1.0, "delete_object": 0.0})

    assert mapper.map_gesture_to_event("Options", "Still", timestamp=1.0) == GestureEvent.CHANGE_COLOR
    assert mapper.map_gesture_to_event("Unknown", "Still", timestamp=1.1) == GestureEvent.NONE
    assert mapper.map_gesture_to_event("Options", "Still", timestamp=1.9) == GestureEvent.NONE
    assert mapper.map_gesture_to_event("Unknown", "Still", timestamp=2.0) == GestureEvent.NONE
    assert mapper.map_gesture_to_event("Options", "Still", timestamp=2.1) == GestureEvent.CHANGE_COLOR


def test_event_mapper_preserves_default_cooldowns_for_unspecified_events():
    mapper = GestureEventMapper({"change_color": 1.0})

    assert mapper.cooldowns[GestureEvent.CHANGE_COLOR] == 1.0
    assert mapper.cooldowns[GestureEvent.TOGGLE_CANVAS] == 0.8
    assert mapper.cooldowns[GestureEvent.EMERGENCY_SOS] == 1.0
