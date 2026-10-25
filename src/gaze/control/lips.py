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
# Tightened 2026-10-24 (was 0.04 / 3x / 0.06 / 1.5 s and "any movement within 0.5 s"): too forgiving, chewing
# or a laugh while a video played let its speech through. Measured: still 0.005-0.01, talking swings ~0.1-0.27.
MIN_MOTION = 0.06  # floor for "moving" (mouth-open ratio range); resting jitter is usually well below
NOISE_FACTOR = 4.0  # or this many times your measured resting jitter, whichever is higher
OPEN_MIN = 0.10  # or held open this much wider than at rest (talking p95 0.27, still 0.01; a smile stays below)
HOLD_S = 0.8  # stay open this long after the lips last moved (pauses between words)
ATTRIBUTE_SLACK_S = 0.2  # a transcript's window, widened this much each side, when checking it was you
MIN_MOVING_SHARE = 0.2  # ...your lips moving in at least this share of its face frames
MIN_MOVING_FRAMES = 3
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

    @property
    def open_above(self):
        """Mouth opening that counts as talking even without movement (learned rest + OPEN_MIN)."""
        return self._rest + OPEN_MIN

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

    def someone_else(self, t0, t1, slack=ATTRIBUTE_SLACK_S):
        """True unless your lips were moving (while you faced the screen) through a real part of the speech in
        [t0, t1]: at least MIN_MOVING_SHARE of the frames your face was seen, and MIN_MOVING_FRAMES. A flicker of
        movement (chewing, a laugh) while a video talks doesn't count; no face means nobody at the computer.
        Only a tracker that stopped sending fails open."""
        a, b = t0 - slack, t1 + slack
        with self._lock:
            if time.time() - self._last_msg > TRACKER_STALE_S:
                return False
            moving = sum(1 for t in self._moves if a <= t <= b)
            faces = sum(1 for t in self._faces if a <= t <= b)
        return moving < max(MIN_MOVING_FRAMES, MIN_MOVING_SHARE * faces)

    def lips_recent(self, seconds, now=None):
        """Lips seen moving (while facing the screen) within `seconds` (or no tracker: fail open)."""
        now = time.time() if now is None else now
        with self._lock:
            return now - self._last_msg > TRACKER_STALE_S or now - self._last_move < seconds

    def face_recent(self, now=None):
        """A face seen within LOOK_HOLD_S (or no tracker: fail open). Keeps a turn going only while you're there."""
        now = time.time() if now is None else now
        with self._lock:
            return now - self._last_msg > TRACKER_STALE_S or bool(self._faces and now - self._faces[-1] < LOOK_HOLD_S)
