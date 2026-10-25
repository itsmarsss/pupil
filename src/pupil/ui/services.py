"""Background services of the app: learning from clicks, snapping to UI elements, recording gaze.

Each is a small object with explicit dependencies, called from the AppKit main thread; slow work
(refits, Accessibility lookups) runs on its own thread.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from pathlib import Path

from ..calibration.implicit import evaluate_click, window
from ..calibration.profile import CLICK
from ..core.env import env
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


class VoiceLog:
    """Events from the voice controller (gaze.control): mic state, what it heard, what it decided.
    Written from the publisher's thread, read by the dashboard."""

    def __init__(self, keep: int = 30):
        self._lock = threading.Lock()
        self._log: collections.deque = collections.deque(maxlen=keep)
        self._state: dict = {}
        self._lips: collections.deque = collections.deque(maxlen=12)  # lip-gate verdicts, newest last
        retention = min(20, max(1, int(env("APIFY_RESEARCH_RETENTION", "5"))))
        self._research: collections.deque = collections.deque(maxlen=retention)

    def add(self, msg: dict) -> None:
        kind, text = msg.get("event"), msg.get("text", "")
        with self._lock:
            if kind in ("mic", "pending"):
                self._state[kind] = text or None
            elif kind == "target":
                self._state["target"] = msg.get("target")
            elif kind == "lip_gate":
                self._state["lip_gate"] = {k: v for k, v in msg.items() if k not in ("event", "text")}
            elif kind == "heat":  # candidates for the last "this"/"that", with their share of the gaze heat
                self._state["heat"] = msg.get("options")
            elif kind == "models":  # which model each role calls, and whether it's being called now
                self._state["models"] = msg.get("models")
            elif kind == "lips":
                self._lips.append({k: v for k, v in msg.items() if k not in ("event", "text")})
            elif kind == "research":
                research = msg.get("research")
                if isinstance(research, dict):
                    self._state["research"] = research
                    if research.get("phase") in ("complete", "partial", "failed"):
                        run_id = research.get("run_id")
                        if not self._research or run_id != self._research[-1].get("run_id"):
                            self._research.append(research)
            else:  # heard / status / tier
                if kind == "heard":
                    self._state["heard"] = text
                    self._state.pop("pending", None)
                self._log.append({"t": round(time.time(), 2), "kind": kind, "text": text})
            self._state["seen"] = time.time()

    def snapshot(self) -> dict | None:
        with self._lock:
            if not self._state:
                return None  # no controller connected
            return {**self._state, "log": list(self._log)[-12:], "lips": list(self._lips),
                    "research_history": list(self._research),
                    "connected": time.time() - self._state["seen"] < 30}

    def clear_research(self) -> None:
        with self._lock:
            self._research.clear()
            self._state.pop("research", None)
