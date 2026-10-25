"""Background services of the app: learning from clicks, snapping to UI elements, recording gaze.

Each is a small object with explicit dependencies, called from the AppKit main thread; slow work
(refits, Accessibility lookups) runs on its own thread.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ..calibration.implicit import evaluate_click, window
from ..calibration.profile import CLICK
from ..core.paths import recordings_dir
from ..sinks.recorder import GazeRecorder
from ..sinks.snapper import Element, Snapper
from ..tracker import Tracker

log = logging.getLogger(__name__)

REFIT_EVERY_CLICKS = 8
SNAP_RADIUS_DEG = 2.5


class ClickLearner:
    """Turns clicks into calibration samples (when gaze was already near them) and refits now and then."""

    def __init__(self, tracker: Tracker, enabled: bool = True):
        self.tracker = tracker
        self.enabled = enabled
        self.seen = self.accepted = 0
        self._since_fit = 0
        self._fitting = False

    def on_click(self, click_pt: tuple[float, float]) -> None:
        profile = self.tracker.profile
        if not self.enabled or profile is None or not profile.calibrated:
            return
        start, end = window(time.monotonic_ns())
        decision = evaluate_click(self.tracker.samples_between(start, end), click_pt, self.tracker.pt_per_deg)
        self.seen += 1
        if not decision.accepted:
            log.debug("click ignored: %s", decision.reason)
            return
        self.accepted += 1
        profile.add(decision.features, click_pt, CLICK, measures=decision.measures)
        self._since_fit += 1
        if self._since_fit >= REFIT_EVERY_CLICKS and not self._fitting:
            self._since_fit = 0
            self._fitting = True
            threading.Thread(target=self._refit, args=(profile,), daemon=True).start()

    def _refit(self, profile) -> None:
        try:
            profile.fit()
            profile.save()
        finally:
            self._fitting = False


class SnapService:
    """Finds the UI element under the current fixation; one Accessibility lookup per fixation, off the
    main thread."""

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.enabled = False
        self._snapper: Snapper | None = None
        self._result: tuple[int, Element | None] | None = None  # (fixation id, element)
        self._pending: int | None = None

    def toggle(self) -> bool:
        if not self.enabled:
            if self._snapper is None:
                self._snapper = Snapper()
            if not self._snapper.available:
                self._snapper.request_permission()
                log.info("snapping needs Accessibility access for this terminal; it starts once granted")
        self.enabled = not self.enabled
        self._result = None
        return self.enabled

    def element_for(self, point) -> Element | None:
        if not self.enabled or self._snapper is None or point.fixation_id is None:
            return None
        if self._result is not None and self._result[0] == point.fixation_id:
            return self._result[1]
        if self._pending is None:
            self._pending = point.fixation_id
            radius = self.tracker.screen.deg_to_pt(SNAP_RADIUS_DEG, self.tracker.distance_mm)
            threading.Thread(target=self._lookup, args=(point.fixation_id, (point.x, point.y), radius),
                             daemon=True).start()
        return None

    def _lookup(self, fixation_id: int, gaze, radius: float) -> None:
        try:
            self._result = (fixation_id, self._snapper.snap(gaze, radius))
        except Exception:
            log.exception("snap lookup failed")
            self._result = (fixation_id, None)
        finally:
            self._pending = None


class Recording:
    """Writes every gaze sample (and clicks) to a CSV while on."""

    def __init__(self, tracker: Tracker):
        self.tracker = tracker
        self.recorder: GazeRecorder | None = None

    @property
    def on(self) -> bool:
        return self.recorder is not None

    def start(self, path: Path | None = None) -> None:
        path = path or recordings_dir() / time.strftime("gaze-%Y%m%d-%H%M%S.csv")
        self.recorder = GazeRecorder(path)
        self.tracker.add_listener(self.recorder.on_sample)
        log.info("recording gaze to %s", path)

    def stop(self) -> None:
        if self.recorder is not None:
            self.tracker.remove_listener(self.recorder.on_sample)
            self.recorder.close()
            log.info("saved %d rows to %s", self.recorder.rows, self.recorder.path)
            self.recorder = None

    def toggle(self) -> bool:
        self.stop() if self.on else self.start()
        return self.on

    def click(self, click_pt: tuple[float, float]) -> None:
        if self.recorder is not None:
            self.recorder.event("click", *click_pt)
