import numpy as np

from gaze.core.types import GazePoint, Sample, TrackingState
from gaze.health.blinks import BlinkDetector
from gaze.health.tracker import HealthTracker


class _Face:
    def __init__(self, openness):
        self.eye_openness = (openness, openness)


def stream(health, seconds, openness, t0=0.0, fps=30, face=True):
    """Feed `seconds` of frames; openness(t) -> eye openness; face=False: nobody in view."""
    t = t0
    while t < t0 + seconds:
        state = TrackingState.TRACKING if face else TrackingState.NO_FACE
        health.on_sample(Sample(int(t * 1e9), None, _Face(openness(t)) if face else None,
                                GazePoint(int(t * 1e9), state)))
        t += 1 / fps
    return t


def blinking(every_s, length_s=0.15, open_=0.30, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return lambda t: (0.04 if (t % every_s) < length_s else open_) + rng.normal(0, noise)


def test_blink_detector_counts_blinks_not_noise_or_long_closures():
    d, blinks, t = BlinkDetector(), [], 0.0
    f = blinking(4.0)
    while t < 40:
        b = d.update(t, f(t)) if not 20 < t < 22 else d.update(t, 0.03)  # a 2 s closure at 20-22 s: not a blink
        if b:
            blinks.append(b)
        t += 1 / 30
    assert 8 <= len(blinks) <= 10  # one every 4 s, minus the closed stretch
    assert all(0.1 <= b.duration <= 0.25 for b in blinks)


def test_blink_rate_and_fatigue_signals():
    h = HealthTracker()
    stream(h, 90, blinking(4.0))  # 15 a minute: relaxed
    s = h.snapshot()
    assert 13 <= s["blink_rate"] <= 17 and s["level"] == "low"
    h2 = HealthTracker()
    stream(h2, 90, blinking(15.0, length_s=0.4))  # 4 a minute, slow blinks: staring, tired
    s2 = h2.snapshot()
    assert s2["blink_rate"] <= 5 and s2["fatigue"] > s["fatigue"]
    assert any("blink" in r for r in h2.due_reminders())  # a nudge...
    assert not any("blink" in r for r in h2.due_reminders())  # ...but not again right away


def test_looking_away_long_enough_is_a_break():
    h = HealthTracker()
    t = stream(h, 60, blinking(4.0))
    t = stream(h, 25, blinking(4.0), t0=t, face=False)  # away 25 s
    stream(h, 30, blinking(4.0), t0=t)
    assert h.breaks == 1 and h.snapshot()["minutes_on_screen"] < 1
