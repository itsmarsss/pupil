import numpy as np

from pupil.core.types import GazePoint, Sample, TrackingState
from pupil.health.blinks import BlinkDetector
from pupil.health.tracker import HealthTracker


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


def test_eye_movements_show_slower_jumps_and_more_drift_when_tired():
    from pupil.health.eyemove import BASELINE_S, EyeMovement

    rng = np.random.default_rng(0)
    em, t, fid, x = EyeMovement(), 0.0, 0, 0.0
    ppd = 40.0  # points per degree

    def look(seconds, saccade_frames, jitter_deg):
        nonlocal t, fid, x
        end = t + seconds
        while t < end:
            fid += 1
            for _ in range(24):  # a 0.8 s fixation
                em.update(t, (x + rng.normal(0, jitter_deg)) * ppd, rng.normal(0, jitter_deg) * ppd, True, fid, ppd)
                t += 1 / 30
            target = 12.0 if x < 6 else 0.0  # then a 12-degree jump, over `saccade_frames` frames
            for k in range(1, saccade_frames + 1):
                em.update(t, (x + (target - x) * k / saccade_frames) * ppd, 0.0, True, None, ppd)
                t += 1 / 30
            x = target

    look(BASELINE_S + 5, saccade_frames=2, jitter_deg=0.15)  # fresh: quick jumps, steady fixations
    fresh = em.snapshot(t)
    assert 0.85 <= fresh["saccade_speed"] <= 1.15 and 0.8 <= fresh["gaze_drift"] <= 1.25
    look(130, saccade_frames=3, jitter_deg=0.35)  # tired: slower jumps, wobblier fixations
    tired = em.snapshot(t)
    assert tired["saccade_speed"] < 0.8 and tired["gaze_drift"] > 1.5
    assert tired["saccade_drag"] > 1.3 and 0.85 <= fresh["saccade_drag"] <= 1.15  # jumps take longer: dragging
