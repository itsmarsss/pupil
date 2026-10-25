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
    t3 = feed(g3, 0, 4, lambda t: None)  # no face seen: can't be sure
    monkeypatch.setattr(lips.time, "time", lambda: t3)
    assert not g3.someone_else(1, 3)
