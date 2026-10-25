from gaze.control.router import route


def test_only_safety_phrases_are_matched_here():
    assert route("never mind").kind == "cancel" and route("yes please").kind == "confirm"
    assert route("stop dictation").kind == "dictate_off" and route("recalibrate").kind == "calibrate"
    assert route("again").kind == "unknown" and route("do the same here").kind == "unknown"  # Jev's to read
