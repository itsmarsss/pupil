"""Lip-movement gate: speech only reaches the recognizer while you face the screen and your lips move.

Built to never lose your words (no false negatives):
- the mic always records into a local pre-roll buffer, and opening the gate sends that buffer first, so a gate
  that opens late only delays the transcript; it doesn't cut the start of what you said;
- it opens on a low bar (any lip movement clearly above your resting jitter) and closes slowly (HOLD_S after the
  lips stop, and never during a turn the recognizer is still transcribing);
- if the tracker stops sending (crashed, not started) it fails open, so a tracker problem can't mute you.

Saves Cartesia credits (billed per second of audio sent, silence included) and keeps bystanders out.
GAZE_LIP_GATE=shadow streams everything but logs each utterance the gate would have blocked, to check for misses.
"""

import threading
import time
from collections import deque

WINDOW_S = 0.4  # lip movement = how much the mouth opening changes over this window
MIN_MOTION = 0.04  # floor for "moving" (mouth-open ratio range); resting jitter is usually well below
NOISE_FACTOR = 3.0  # or this many times your measured resting jitter, whichever is higher
OPEN_MIN = 0.06  # or held open this much wider than at rest (a steady open mouth: measured talking p95 0.27, still 0.01)
HOLD_S = 1.5  # stay open this long after the lips last moved (pauses between words)
LOOK_HOLD_S = 1.5  # glancing away mid-sentence doesn't count as not looking
MIN_CONFIDENCE = 0.2  # tracker confidence: falls toward 0 when the head leaves the range it was calibrated facing
TRACKER_STALE_S = 1.0  # no tracker messages for this long: fail open


class LipGate:
    def __init__(self, mode="on"):
        self.mode = mode  # on | shadow | off
        self._lock = threading.Lock()
        self._mouth = deque()  # (t, mouth-open ratio)
        self._noise = MIN_MOTION / NOISE_FACTOR  # resting lip jitter, learned
        self._rest = 0.01  # resting mouth opening, learned
        self._last_msg = 0.0
        self._last_look = 0.0
        self._last_move = 0.0
        self._moves = deque(maxlen=2000)  # times the lips moved while looking
        self._faces = deque(maxlen=2000)  # times a face was visible
        self.reason = "starting"

    def update(self, msg, now=None):
        """One tracker message (see sinks/publisher.py)."""
        now = time.time() if now is None else now
        with self._lock:
            self._last_msg = now
            face = "mouth" in msg
            looking = face and not msg.get("busy") and (
                msg.get("state") == "uncalibrated"  # nothing to judge "looking" by yet: a visible face will do
                or (msg.get("state") in ("tracking", "blink") and msg.get("confidence", 1.0) >= MIN_CONFIDENCE))
            if looking:
                self._last_look = now
            if not face:
                self._mouth.clear()
                return
            self._faces.append(now)
            self._mouth.append((now, msg["mouth"]))
            while self._mouth and self._mouth[0][0] < now - WINDOW_S:
                self._mouth.popleft()
            values = [m for _, m in self._mouth]
            motion = max(values) - min(values)
            if motion > self.threshold or msg["mouth"] > self._rest + OPEN_MIN:
                if now - self._last_look < LOOK_HOLD_S:
                    self._last_move = now
                    self._moves.append(now)
            else:  # learn the resting jitter and opening from still frames only
                self._noise = 0.98 * self._noise + 0.02 * motion
                self._rest = 0.98 * self._rest + 0.02 * msg["mouth"]

    @property
    def threshold(self):
        return max(MIN_MOTION, NOISE_FACTOR * self._noise)

    def is_open(self, now=None):
        """Whether audio should go to the recognizer right now (always True when off or in shadow mode)."""
        return self.mode != "on" or self.would_open(now)

    def would_open(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            if now - self._last_msg > TRACKER_STALE_S:
                self.reason = "no tracker - open"
                return True
            if now - self._last_look > LOOK_HOLD_S:
                self.reason = "not looking"
                return False
            if now - self._last_move > HOLD_S:
                self.reason = "lips still"
                return False
            self.reason = "open"
            return True

    def someone_else(self, t0, t1, slack=0.5):
        """True only when sure a transcript of speech in [t0, t1] wasn't you: the tracker was running and saw your
        face for most of it, and your lips never moved. Any doubt (no tracker, face lost) counts as you."""
        a, b = t0 - slack, t1 + slack
        with self._lock:
            if time.time() - self._last_msg > TRACKER_STALE_S:
                return False
            if any(a <= t <= b for t in self._moves):
                return False
            seen = sum(1 for t in self._faces if a <= t <= b)
        return seen >= 0.5 * (b - a) * 15  # face in at least half the frames (tracker runs at ~15-30 fps)
