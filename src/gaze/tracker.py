"""The running pipeline: frames → face → features → calibrated, filtered gaze points.

Inference runs on its own thread. Results go to listeners (called on that thread) and
into a short history, which calibration and click learning query by timestamp.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable

import numpy as np

from .calibration.profile import Profile
from .capture.camera import FrameSource
from .core.screen import Screen
from .core.types import FaceObservation, GazeFeatures, GazePoint, Sample, TrackingState
from .face.tracker import FaceTracker
from .filter.fixation import FixationDetector
from .filter.one_euro import OneEuroFilter
from .models.base import GazeModel

log = logging.getLogger(__name__)

DEFAULT_DISTANCE_MM = 550.0
HISTORY_S = 6.0

Listener = Callable[[Sample], None]


class Tracker:
    def __init__(
        self,
        source: FrameSource,
        face_tracker: FaceTracker,
        model: GazeModel,
        screen: Screen,
        profile: Profile | None = None,
        min_cutoff: float = 0.6,
        beta: float = 0.015,
    ):
        self.source = source
        self.face_tracker = face_tracker
        self.model = model
        self.screen = screen
        self.profile = profile
        self._filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self._fixations = FixationDetector()
        self._history: collections.deque[Sample] = collections.deque()
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []
        self._thread: threading.Thread | None = None
        self._running = False
        self._last_point: GazePoint | None = None
        self.distance_mm = DEFAULT_DISTANCE_MM
        self.fps = 0.0

    # ---- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        self.source.start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self.source.stop()

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    # ---- queries -------------------------------------------------------------------
    def samples_between(self, start_ns: int, end_ns: int) -> list[Sample]:
        with self._lock:
            return [s for s in self._history if start_ns <= s.t_ns <= end_ns]

    def wait_until(self, t_ns: int, timeout: float = 1.0) -> bool:
        """Block until a sample at or after `t_ns` exists (so a window is complete)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._history and self._history[-1].t_ns >= t_ns:
                    return True
            time.sleep(0.01)
        return False

    @property
    def pt_per_deg(self) -> float:
        return self.screen.deg_to_pt(1.0, self.distance_mm)

    # ---- loop ----------------------------------------------------------------------
    def _loop(self) -> None:
        last_t = 0
        fps_t, fps_n = time.monotonic(), 0
        while self._running:
            frame = self.source.latest(after_t_ns=last_t, timeout=0.5)
            if frame is None:
                continue
            last_t = frame.t_ns
            try:
                sample = self.process(frame)
            except Exception:
                log.exception("inference failed")
                continue
            fps_n += 1
            if time.monotonic() - fps_t >= 1.0:
                self.fps, fps_t, fps_n = fps_n / (time.monotonic() - fps_t), time.monotonic(), 0
            for fn in list(self._listeners):
                try:
                    fn(sample)
                except Exception:
                    log.exception("listener failed")

    def process(self, frame) -> Sample:
        face = self.face_tracker.process(frame)
        features = self.model.infer(frame, face) if face is not None else None
        point = self._point(frame.t_ns, face, features)
        sample = Sample(frame.t_ns, features, face, point)
        with self._lock:
            self._history.append(sample)
            cutoff = frame.t_ns - int(HISTORY_S * 1e9)
            while self._history and self._history[0].t_ns < cutoff:
                self._history.popleft()
        self._last_point = point
        return sample

    def _point(self, t_ns: int, face: FaceObservation | None, features: GazeFeatures | None) -> GazePoint:
        if face is None or features is None:
            self._filter.reset()
            self._fixations.reset()
            return GazePoint(t_ns, TrackingState.NO_FACE)
        if face.distance_mm:
            self.distance_mm = 0.9 * self.distance_mm + 0.1 * face.distance_mm
        if face.is_blinking:
            last = self._last_point
            if last is not None and last.valid:
                return GazePoint(t_ns, TrackingState.BLINK, last.x, last.y, last.raw_x, last.raw_y, None, self.distance_mm)
            return GazePoint(t_ns, TrackingState.BLINK, distance_mm=self.distance_mm)
        prediction = self.profile.predict(features.vector) if self.profile is not None else None
        if prediction is None:
            return GazePoint(t_ns, TrackingState.UNCALIBRATED, distance_mm=self.distance_mm)
        raw_x, raw_y = prediction
        x, y = self._filter(np.array([raw_x, raw_y]), t_ns)
        fixation = self._fixations.update(float(x), float(y), t_ns, self.pt_per_deg)
        return GazePoint(t_ns, TrackingState.TRACKING, float(x), float(y), raw_x, raw_y, fixation, self.distance_mm)
