from gaze.control.lips import HOLD_S, WINDOW_S, LipGate

LOOK = {"state": "tracking", "confidence": 1.0}


def feed(gate, t0, t1, mouth, fps=30, **msg):
    """Tracker messages from t0 to t1; mouth(t) gives the mouth-open ratio (None = no face)."""
    t = t0
    while t < t1:
        m = mouth(t)
        gate.update({**LOOK, **msg, **({"mouth": m} if m is not None else {})}, now=t)
        t += 1 / fps
    return t


def talking(t):
    return 0.05 + 0.15 * (int(t * 8) % 2)  # opens and closes ~4 times a second


def test_closed_while_lips_still_open_as_soon_as_they_move():
    g = LipGate()
    t = feed(g, 0, 3, lambda t: 0.02)
    assert not g.would_open(t) and g.reason == "lips still"
    t = feed(g, t, t + 0.2, talking)  # within a fifth of a second of talking
    assert g.would_open(t)


def test_needs_you_facing_the_screen():
    g = LipGate()
    t = feed(g, 0, 2, talking, confidence=0.0)  # head turned away
    assert not g.would_open(t) and g.reason == "not looking"


def test_holds_through_pauses_then_closes():
    g = LipGate()
    t = feed(g, 0, 2, talking)
    t = feed(g, t, t + HOLD_S - 0.3, lambda t: 0.02)
    assert g.would_open(t)
    t = feed(g, t, t + WINDOW_S + 0.4, lambda t: 0.02)  # the lip window trails the last movement
    assert not g.would_open(t)


def test_fails_open_without_tracker():
    g = LipGate()
    feed(g, 0, 1, lambda t: 0.02)
    assert g.would_open(5.0) and g.reason.startswith("no tracker")


def test_shadow_and_off_never_block():
    for mode in ("shadow", "off"):
        g = LipGate(mode)
        t = feed(g, 0, 2, lambda t: 0.02)
        assert g.is_open(t)


def test_someone_else_only_when_sure(monkeypatch):
    import gaze.control.lips as lips

    g = LipGate()
    t = feed(g, 0, 4, lambda t: 0.02)
    monkeypatch.setattr(lips.time, "time", lambda: t)
    assert g.someone_else(1, 3)  # face visible, lips still the whole time
    g2 = LipGate()
    t2 = feed(g2, 0, 4, lambda t: talking(t) if 2 < t < 2.5 else 0.02)
    monkeypatch.setattr(lips.time, "time", lambda: t2)
    assert not g2.someone_else(1, 3)  # lips moved during it
    g3 = LipGate()
    t3 = feed(g3, 0, 4, lambda t: None)  # no face: nobody at the computer, so not you
    monkeypatch.setattr(lips.time, "time", lambda: t3)
    assert g3.someone_else(1, 3)
    assert not g3.face_recent(t3)
    monkeypatch.setattr(lips.time, "time", lambda: t3 + 10)  # tracker stopped sending: fail open
    assert not g3.someone_else(1, 3) and g3.face_recent(t3 + 10)


def test_opens_for_a_steady_open_mouth():
    g = LipGate()
    t = feed(g, 0, 3, lambda t: 0.01)
    t = feed(g, t, t + 1.2, lambda t: 0.15)  # opened and held: no change after the first frame
    assert g.would_open(t)


def test_a_flicker_of_movement_during_someone_elses_speech_isnt_you(monkeypatch):
    import gaze.control.lips as lips

    g = LipGate()
    t = feed(g, 0, 6, lambda t: talking(t) if 3.0 < t < 3.15 else 0.02)  # a quick chew in a 3 s utterance
    monkeypatch.setattr(lips.time, "time", lambda: t)
    assert g.someone_else(1.5, 4.5)
    g2 = LipGate()
    t2 = feed(g2, 0, 6, lambda t: talking(t) if 2 < t < 4 else 0.02)  # talking through most of it
    monkeypatch.setattr(lips.time, "time", lambda: t2)
    assert not g2.someone_else(1.5, 4.5)


def test_lips_recent_for_cutting_a_turn_short(monkeypatch):
    import gaze.control.lips as lips

    g = LipGate()
    t = feed(g, 0, 2, talking)
    t = feed(g, t, t + 3, lambda t: 0.02)  # then still for 3 s (a TV keeps talking)
    monkeypatch.setattr(lips.time, "time", lambda: t)
    assert g.face_recent() and not g.lips_recent(2.0)  # face there, lips still: stop streaming
    assert g.lips_recent(4.0)


def test_audio_is_kept_only_around_your_lip_movement(monkeypatch):
    import gaze.control.lips as lips

    g = LipGate()
    t = feed(g, 0, 2, lambda t: 0.02)  # still: someone else talking
    t = feed(g, t, t + 1.5, talking)  # you speak (2 .. 3.5 s)
    t = feed(g, t, t + 2, lambda t: 0.02)  # still again
    monkeypatch.setattr(lips.time, "time", lambda: t)
    assert not g.moved_near(1.0)  # before you spoke: muted
    assert g.moved_near(2.5) and g.moved_near(3.9)  # while you speak, and a short pause after
    assert g.moved_near(1.85)  # just before your lips opened: kept (camera lag)
    assert not g.moved_near(5.0)  # long after: muted


def test_voice_mutes_chunks_without_lip_movement():
    from gaze.control.voice import Voice

    class Gate:
        mode = "on"

        def moved_near(self, t):
            return 2 <= t <= 4

    v = Voice(lambda *a: None, lambda s: None, gate=Gate())
    assert v._yours(3.0) and not v._yours(1.0) and not v._yours(5.0)
    Gate.mode = "shadow"  # shadow mode never changes what's sent
    assert v._yours(1.0)
