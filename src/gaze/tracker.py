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

from .calibration.features import calibration_vector
from .core.device import load_device
from .geometry.eyeball import ScreenPlane, geo_block, reconstruct, to_array
from .calibration.profile import Profile
from .capture.camera import FrameSource
from .core.screen import Screen
from .core.types import FaceObservation, Frame, GazePoint, Sample, TrackingState
from .face.tracker import FaceTracker
from .filter.fixation import FixationDetector
from .filter.smoother import PRESETS, GazeSmoother
from .models.base import GazeModel

log = logging.getLogger(__name__)

DEFAULT_DISTANCE_MM = 550.0
HISTORY_S = 6.0

Listener = Callable[[Sample], None]
FrameListener = Callable[[Frame, Sample], None]


class Tracker:
    def __init__(
        self,
        source: FrameSource,
        face_tracker: FaceTracker,
        model: GazeModel,
        screen: Screen,
        profile: Profile | None = None,
        smoothing: str = "medium",
    ):
        self.source = source
        self.face_tracker = face_tracker
        self.model = model
        self.screen = screen
        self.profile = profile
        self.smoothing = smoothing
        self._filter = GazeSmoother(PRESETS[smoothing])
        device = load_device()
        self.plane = ScreenPlane(screen, device.camera_below_top_cm, device.camera_x_offset_cm)
        self._fixations = FixationDetector()
        self._history: collections.deque[Sample] = collections.deque()
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []
        self._frame_listeners: list[FrameListener] = []  # get the image too; frames aren't kept
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

    def switch_source(self, source: FrameSource) -> None:
        """Swap the camera while running (blocks while the new one opens): the old one is released first so
        the same device can be reopened, and face and filter state start fresh."""
        running = self._running
        self.stop()
        self.source = source
        self.face_tracker.reset()
        self._filter.reset()
        self._fixations.reset()
        self._last_point = None
        with self._lock:
            self._history.clear()
        if running:
            self.start()

    def add_listener(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def remove_listener(self, fn: Listener) -> None:
        if fn in self._listeners:
            self._listeners.remove(fn)

    def add_frame_listener(self, fn: FrameListener) -> None:
        self._frame_listeners.append(fn)

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

    def set_smoothing(self, name: str) -> None:
        self.smoothing = name
        self._filter = GazeSmoother(PRESETS[name])

    @property
    def last_point(self) -> GazePoint | None:
        """The newest gaze point (read from the UI thread; a single reference, so no lock needed)."""
        return self._last_point

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
            for fn in list(self._frame_listeners):
                try:
                    fn(frame, sample)
                except Exception:
                    log.exception("frame listener failed")

    def process(self, frame) -> Sample:
        face = self.face_tracker.process(frame)
        features = self.model.infer(frame, face) if face is not None else None
        calib, geo, rec = None, None, None
        if features is not None:
            rec = reconstruct(face)  # eyeball model, from the unwarped face
            block = geo_block(rec.measure, self.plane) if rec is not None else None
            if block is not None:
                calib, geo = calibration_vector(features.vector, face, block), to_array(rec.measure)
        point = self._point(frame.t_ns, face, calib, geo)
        sample = Sample(frame.t_ns, features, face, point, calib, geo)
        if rec is not None:
            sample.extra["reconstruction"] = rec  # 3D detail for visualisation
        with self._lock:
            self._history.append(sample)
            cutoff = frame.t_ns - int(HISTORY_S * 1e9)
            while self._history and self._history[0].t_ns < cutoff:
                self._history.popleft()
        self._last_point = point
        return sample

    def _point(self, t_ns: int, face: FaceObservation | None, calib: np.ndarray | None,
               geo: np.ndarray | None = None) -> GazePoint:
        if face is None or calib is None:
            self._filter.reset()
            self._fixations.reset()
            return GazePoint(t_ns, TrackingState.NO_FACE)
        if face.distance_mm:
            self.distance_mm = 0.9 * self.distance_mm + 0.1 * face.distance_mm
        if face.is_blinking:
            last = self._last_point
            if last is not None and last.valid:
                return GazePoint(t_ns, TrackingState.BLINK, last.x, last.y, last.raw_x, last.raw_y, None,
                                 self.distance_mm, last.confidence)
            return GazePoint(t_ns, TrackingState.BLINK, distance_mm=self.distance_mm)
        profile = self.profile
        prediction = profile.predict(calib, geo) if profile is not None else None
        if prediction is None:
            return GazePoint(t_ns, TrackingState.UNCALIBRATED, distance_mm=self.distance_mm)
        # Outside the calibrated head-pose range the fit extrapolates; keep it on screen.
        raw_x, raw_y = self.screen.clamp(*prediction)
        confidence = profile.confidence(calib)
        x, y = self._filter(raw_x, raw_y, t_ns, self.pt_per_deg)
        fixation = self._fixations.update(float(x), float(y), t_ns, self.pt_per_deg)
        return GazePoint(t_ns, TrackingState.TRACKING, float(x), float(y), raw_x, raw_y, fixation,
                         self.distance_mm, confidence)
