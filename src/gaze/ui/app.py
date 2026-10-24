"""Menu-bar app: launch flow, calibration, validation, overlay and click learning.

Everything here runs on the AppKit main thread. The tracker runs inference on its own
thread; we read its latest point from a 60 Hz timer instead of calling into AppKit
from that thread.
"""
from __future__ import annotations

import logging
import signal
import threading
import time
from pathlib import Path

import numpy as np
import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSEvent,
    NSEventMaskLeftMouseDown,
    NSMenu,
    NSMenuItem,
    NSObject,
    NSStatusBar,
    NSTimer,
    NSVariableStatusItemLength,
)
from PyObjCTools import AppHelper

from ..calibration.implicit import evaluate_click, window
from ..calibration.profile import CLICK, EXPLICIT, MOTION, Profile, ProfileMeta
from ..calibration.targets import TargetTiming, grid, shuffled
from ..core.paths import APP_SUPPORT, recordings_dir
from ..core.types import TrackingState
from ..eval.metrics import ValidationResult, append_result, score
from ..eval.session import SessionWriter
from ..sinks.recorder import GazeRecorder
from ..sinks.snapper import Element, Snapper
from ..tracker import Tracker
from .overlay import GazeRing
from .target_screen import ESC, RETURN, SPACE, TargetScreen, TargetSequence

log = logging.getLogger(__name__)

LAUNCH_CHECK_MAX_DEG = 4.0  # saved calibration is reused if the quick check beats this
REFIT_EVERY_CLICKS = 8
MIN_SAMPLES_PER_TARGET = 8
SNAP_RADIUS_DEG = 2.5
MOTION_TIMING = TargetTiming(dwell_s=6.5, settle_s=0.8, tail_s=0.1)
MOTION_CHUNK = 15  # score the head-motion sweep in half-second pieces


class _MenuTarget(NSObject):
    """Bridges menu actions (Objective-C selectors) to Python callables."""

    def initWithActions_(self, actions):
        self = objc.super(_MenuTarget, self).init()
        if self is not None:
            self.actions = actions
        return self

    def run_(self, sender):
        self.actions[sender.tag()]()


class GazeApp:
    def __init__(
        self,
        tracker: Tracker,
        profile_name: str = "default",
        n_targets: int = 9,
        recalibrate: bool = False,
        results_csv: Path | None = None,
        learn_from_clicks: bool = True,
        snap: bool = False,
        record_path: Path | None = None,
        session_dir: Path | None = None,
        measure: bool = False,
    ):
        self.tracker = tracker
        self.screen = tracker.screen
        self.profile_name = profile_name
        self.n_targets = n_targets
        self.force_recalibrate = recalibrate
        self.results_csv = results_csv or APP_SUPPORT / "validation.csv"
        self.learn_from_clicks = learn_from_clicks
        self.overlay_on = True
        self.snap_on = snap
        self._snapper: Snapper | None = None
        self._snap: tuple[int, Element | None] | None = None  # (fixation id, element)
        self._snap_pending: int | None = None
        self.recorder: GazeRecorder | None = None
        self._record_path = record_path
        self._session_dir = session_dir
        self._pending_measure = measure  # run the measurement once the launch flow finishes
        self.session: SessionWriter | None = None

        self.ring: GazeRing | None = None
        self.targets: TargetScreen | None = None
        self.timing = TargetTiming()
        self._busy = False  # calibration/validation on screen
        self._last_valid_ns = 0
        self._click_stats = {"ok": 0, "seen": 0}
        self._clicks_since_fit = 0
        self._fitting = False
        self._status_item = None
        self._status_line = None
        self._menu_items: dict[str, NSMenuItem] = {}
        self._timers = []
        self._last_result: ValidationResult | None = None
        self._shut_down = False

    # ---- startup -------------------------------------------------------------------
    def run(self) -> None:
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.ring = GazeRing(self.screen)
        self.targets = TargetScreen(self.screen)
        self._build_menu()
        if self.snap_on:
            self.snap_on = False
            self._toggle_snap()
        if self._record_path is not None:
            self._start_recording(self._record_path)
        if self._session_dir is not None:
            self.session = SessionWriter(self._session_dir, {
                "screen_pt": [self.screen.width_pt, self.screen.height_pt],
                "screen_mm": [self.screen.width_mm, self.screen.height_mm],
                "timing": vars(self.timing),
                "motion_timing": vars(MOTION_TIMING),
                "mode": getattr(getattr(self.tracker.source, "mode", None), "name", ""),
                "model": self.tracker.model.model_id,
                "camera": self.tracker.source.source_id,
            })
            self.tracker.add_frame_listener(self.session.on_frame)
            log.info("saving calibration/validation frames to %s", self._session_dir)
        self.tracker.start()
        self._timers.append(NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 60, True, self._tick))
        self._timers.append(NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1.0, True, self._update_status))
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskLeftMouseDown, self._on_click)
        # start the launch flow once the run loop is going
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.1, False, lambda _t: self._launch())
        # Python only runs signal handlers while the main thread executes Python code;
        # the 1 s status timer guarantees that, so Ctrl+C and kill quit cleanly.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self._quit())
        try:
            AppHelper.runEventLoop()
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        """Idempotent: stopping the AppKit loop may exit the process without unwinding,
        so every quit path calls this first."""
        if self._shut_down:
            return
        self._shut_down = True
        self.tracker.stop()
        self._stop_recording()
        if self.session is not None:
            self.session.close()
            log.info("session: %d frames in %s", self.session.frames, self.session.root)
        if self.tracker.profile is not None and self.tracker.profile.calibrated:
            self.tracker.profile.save()

    def _launch(self) -> None:
        model_id = self.tracker.model.model_id
        camera = self.tracker.source.source_id
        screen_pt = (self.screen.width_pt, self.screen.height_pt)
        profile = None if self.force_recalibrate else Profile.load(self.profile_name)
        if profile is not None and profile.meta.matches(model_id, camera, screen_pt) and profile.calibrated:
            log.info("loaded profile %r (%d targets, %d from clicks)", self.profile_name, profile.n_groups, profile.count(CLICK))
            self.tracker.profile = profile
            self._validate(n=3, launch_check=True)
            return
        if profile is not None and profile.meta.matches(model_id, camera, screen_pt):
            self.tracker.profile = profile  # keep its click samples
        self._intro()

    # ---- screens -------------------------------------------------------------------
    def _intro(self) -> None:
        self._busy = True
        self.ring.hide()

        def lines():
            last = self.tracker.samples_between(time.monotonic_ns() - 500_000_000, time.monotonic_ns())
            faces = [s.face for s in last if s.face is not None]
            if not faces:
                status = "Looking for your face…  make sure it's lit and in view of the camera"
            else:
                d = np.median([f.distance_mm for f in faces if f.distance_mm]) / 10
                hint = "  ·  a bit close" if d < 40 else "  ·  a bit far" if d > 85 else ""
                status = f"Face found  ·  {d:.0f} cm from the camera{hint}"
            return [
                "Look at each dot until it shrinks away",
                f"{self.n_targets} dots, then a few seconds turning your head (about "
                f"{self.n_targets * self.timing.dwell_s + MOTION_TIMING.dwell_s:.0f} s). Sit the way you normally do.",
                status,
                "Press Space to start  ·  Esc to quit",
            ]

        def on_key(c):
            if c == SPACE:
                timer.invalidate()
                self._calibrate()
            elif c == ESC:
                timer.invalidate()
                self._quit()

        self.targets.show(lines(), on_key=on_key)
        timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.25, True, lambda _t: self.targets.set_lines(lines()))

    def _profile_for(self, dim: int) -> Profile:
        """The current profile, or a new one if there is none or its features differ."""
        p = self.tracker.profile
        if p is None or p.X.shape[1] != dim:
            meta = ProfileMeta(
                self.profile_name,
                self.tracker.model.model_id,
                self.tracker.source.source_id,
                (self.screen.width_pt, self.screen.height_pt),
            )
            p = Profile(meta, dim)
        return p

    def _calibrate(self) -> None:
        self._busy = True
        self.ring.hide()
        collected: dict[tuple, np.ndarray] = {}
        points = shuffled(grid(self.n_targets, self.screen.width_pt, self.screen.height_pt))

        def on_target(target, onset_ns, end_ns):
            self._label(None)
            feats = self._features_for(onset_ns, self.timing)
            if len(feats) >= MIN_SAMPLES_PER_TARGET:
                collected[target] = feats

        def on_done(cancelled):
            if cancelled:
                self._after_cancel()
                return
            if len(collected) < len(points) - len(points) // 3:
                self._message(
                    ["Couldn't see your eyes clearly", f"Got {len(collected)} of {len(points)} dots. Check lighting and that your face is in view.", "Press Space to try again  ·  Esc to quit"],
                    {SPACE: self._calibrate, ESC: self._quit},
                )
                return
            self._head_motion(collected)

        TargetSequence(self.targets, points, self.timing, on_target, on_done,
                       on_start=lambda t, onset: self._label("calib", t, onset)).start()

    def _head_motion(self, dots: dict[tuple, np.ndarray]) -> None:
        """Hold gaze on one dot while turning the head, so the fit learns to separate
        head movement from eye movement."""
        centre = (self.screen.width_pt / 2, self.screen.height_pt / 2)
        sweep: list[np.ndarray] = []

        def on_target(target, onset_ns, end_ns):
            self._label(None)
            feats = self._features_for(onset_ns, MOTION_TIMING)
            if len(feats):
                sweep.append(feats)

        def on_done(cancelled):
            if cancelled:
                self._after_cancel()
                return
            self._finish_calibration(dots, centre, sweep[0] if sweep else None)

        lines = ["Keep looking at the dot", "and slowly turn your head: left, right, up, down"]
        TargetSequence(self.targets, [centre], MOTION_TIMING, on_target, on_done,
                       on_start=lambda t, onset: self._label("motion", t, onset), lines=lines).start()

    def _finish_calibration(self, dots: dict[tuple, np.ndarray], centre, sweep: np.ndarray | None) -> None:
        profile = self._profile_for(next(iter(dots.values())).shape[1])
        profile.clear(EXPLICIT, MOTION)  # click samples are kept
        for target, feats in dots.items():
            profile.add(feats, target, EXPLICIT)
        if sweep is not None and len(sweep) >= MIN_SAMPLES_PER_TARGET:
            profile.add(sweep, centre, MOTION, chunk=MOTION_CHUNK)
        report = profile.fit()
        if report is None:
            self._message(["Calibration failed", "Press Space to try again"], {SPACE: self._calibrate, ESC: self._quit})
            return
        profile.save()
        self.tracker.profile = profile
        self._validate(n=3, launch_check=False)

    def _label(self, phase, target=None, onset_ns: int = 0) -> None:
        if self.session is not None:
            self.session.label(phase, target, onset_ns)

    def _features_for(self, onset_ns: int, timing: TargetTiming) -> np.ndarray:
        start = onset_ns + int(timing.settle_s * 1e9)
        end = onset_ns + int((timing.dwell_s - timing.tail_s) * 1e9)
        self.tracker.wait_until(end, timeout=0.2)
        samples = self.tracker.samples_between(start, end)
        feats = [s.calib for s in samples if s.calib is not None and not s.face.is_blinking]
        return np.stack(feats) if feats else np.zeros((0, 0), np.float32)

    def _validate(self, n: int = 9, launch_check: bool = False, phase: str = "valid", then=None,
                  lines: list[str] | None = None) -> None:
        """Show `n` dots and score gaze against them. `then(result)` replaces the results
        screen, for chaining checks; `phase` labels the frames in a saved session."""
        self._busy = True
        self.ring.hide()
        points = shuffled(grid(n, self.screen.width_pt, self.screen.height_pt))
        per_target: dict[tuple, list] = {}
        totals = {"frames": 0}
        distances = []

        def on_target(target, onset_ns, end_ns):
            self._label(None)
            start = onset_ns + int(self.timing.settle_s * 1e9)
            end = onset_ns + int((self.timing.dwell_s - self.timing.tail_s) * 1e9)
            self.tracker.wait_until(end, timeout=0.2)
            samples = self.tracker.samples_between(start, end)
            totals["frames"] += len(samples)
            per_target[target] = [(s.point.raw_x, s.point.raw_y) for s in samples if s.point.state is TrackingState.TRACKING]
            distances.extend(s.point.distance_mm for s in samples if s.point.distance_mm)

        def on_done(cancelled):
            if cancelled:
                self._after_cancel()
                return
            dist = float(np.median(distances)) if distances else self.tracker.distance_mm
            result = score(per_target, self.screen, dist, totals["frames"])
            self._last_result = result
            log.info("validation (%d targets): %s", n, result.summary())
            try:
                kind = "launch" if launch_check else phase if phase != "valid" else "manual"
                append_result(self.results_csv, result, targets=n, kind=kind,
                              mode=getattr(getattr(self.tracker.source, "mode", None), "name", ""),
                              model=self.tracker.model.model_id,
                              fit=self.tracker.profile.report.chosen if self.tracker.profile and self.tracker.profile.report else "")
            except OSError:
                log.exception("could not write results")
            if launch_check and not (result.accuracy_deg <= LAUNCH_CHECK_MAX_DEG):
                log.info("saved calibration is off (%.1f°); recalibrating", result.accuracy_deg)
                self._intro()
                return
            if then is not None:
                then(result)
                return
            acc = "no data" if np.isnan(result.accuracy_deg) else f"{result.accuracy_deg:.1f}°  (about {result.accuracy_pt:.0f} pt)"
            lines = [f"Accuracy {acc}", result.summary(), "Return to start  ·  Space to recalibrate"]
            self._message(lines, {RETURN: self._start_overlay, SPACE: self._calibrate, ESC: self._start_overlay},
                          auto_continue=None if n > 3 else 2.5)

        TargetSequence(self.targets, points, self.timing, on_target, on_done,
                       on_start=lambda t, onset: self._label(phase, t, onset), lines=lines).start()

    def _measure(self) -> None:
        """Two 9-dot checks: head still, then head moving naturally."""
        if self.tracker.profile is None or not self.tracker.profile.calibrated:
            return
        self._busy = True
        self.ring.hide()
        results = {}

        def show_results(moving):
            results["moving"] = moving
            fmt = lambda r: "no data" if np.isnan(r.accuracy_deg) else f"{r.accuracy_deg:.1f}° (p95 {r.accuracy_p95_deg:.1f}°, jitter {r.precision_rms_deg:.2f}°)"  # noqa: E731
            self._message(
                ["Measurement done", f"Head still:  {fmt(results['still'])}", f"Head moving:  {fmt(moving)}",
                 f"Distance {results['still'].distance_mm / 10:.0f} cm  ·  Return to continue"],
                {RETURN: self._start_overlay, ESC: self._start_overlay},
            )

        def moving_intro(still):
            results["still"] = still
            self._message(
                ["Now move your head naturally", "Follow the dots with your eyes while turning and tilting your head a little.",
                 "Press Space to start"],
                {SPACE: lambda: self._validate(9, phase="valid_moving", then=show_results), ESC: self._start_overlay},
            )

        self._message(
            ["Accuracy measurement", "First 9 dots: keep your head as still as you can. Then 9 more while moving it.",
             "Press Space to start  ·  Esc to cancel"],
            {SPACE: lambda: self._validate(9, phase="valid_still", then=moving_intro), ESC: self._start_overlay},
        )

    def _message(self, lines, keys: dict, auto_continue: float | None = None) -> None:
        state = {"done": False}

        def go(fn):
            if not state["done"]:
                state["done"] = True
                fn()

        self.targets.show(lines, on_key=lambda c: c in keys and go(keys[c]))
        if auto_continue:
            NSTimer.scheduledTimerWithTimeInterval_repeats_block_(auto_continue, False, lambda _t: go(keys[RETURN]))

    def _after_cancel(self) -> None:
        if self.tracker.profile is not None and self.tracker.profile.calibrated:
            self._start_overlay()
        else:
            self._quit()

    def _start_overlay(self) -> None:
        if self._pending_measure:
            self._pending_measure = False
            self._measure()
            return
        self.targets.hide()
        self._busy = False

    # ---- per-frame ----------------------------------------------------------------
    def _tick(self, _timer) -> None:
        if self._busy or not self.overlay_on:
            self.ring.hide()
            return
        p = self.tracker._last_point
        now = time.monotonic_ns()
        if p is not None and p.valid:
            self._last_valid_ns = now
            target = self._snapped(p)
            if target is not None:
                self.ring.show_at(*target.centre, fixating=True)
            else:
                self.ring.show_at(p.x, p.y, fixating=p.fixation_id is not None, confidence=p.confidence)
        elif now - self._last_valid_ns > 400_000_000:  # brief dropouts (blinks) keep the ring
            self.ring.hide()

    # ---- snapping -------------------------------------------------------------------
    def _snapped(self, p) -> Element | None:
        """Element for the current fixation; looked up once per fixation off the main thread."""
        if not self.snap_on or self._snapper is None or p.fixation_id is None:
            return None
        if self._snap is not None and self._snap[0] == p.fixation_id:
            return self._snap[1]
        if self._snap_pending is None:
            self._snap_pending = p.fixation_id
            radius = self.screen.deg_to_pt(SNAP_RADIUS_DEG, self.tracker.distance_mm)
            threading.Thread(target=self._lookup, args=(p.fixation_id, (p.x, p.y), radius), daemon=True).start()
        return None

    def _lookup(self, fixation_id: int, gaze, radius: float) -> None:
        try:
            self._snap = (fixation_id, self._snapper.snap(gaze, radius))
        except Exception:
            log.exception("snap lookup failed")
            self._snap = (fixation_id, None)
        finally:
            self._snap_pending = None

    # ---- recording ------------------------------------------------------------------
    def _start_recording(self, path: Path | None = None) -> None:
        path = path or recordings_dir() / time.strftime("gaze-%Y%m%d-%H%M%S.csv")
        self.recorder = GazeRecorder(path)
        self.tracker.add_listener(self.recorder.on_sample)
        log.info("recording gaze to %s", path)

    def _stop_recording(self) -> None:
        if self.recorder is not None:
            self.tracker.remove_listener(self.recorder.on_sample)
            self.recorder.close()
            log.info("saved %d rows to %s", self.recorder.rows, self.recorder.path)
            self.recorder = None

    # ---- click learning --------------------------------------------------------------
    def _on_click(self, event) -> None:
        if self.recorder is not None:
            loc = NSEvent.mouseLocation()
            self.recorder.event("click", float(loc.x), self.screen.to_cocoa_y(float(loc.y)))
        profile = self.tracker.profile
        if not self.learn_from_clicks or self._busy or profile is None or not profile.calibrated:
            return
        loc = NSEvent.mouseLocation()
        click = (float(loc.x), self.screen.to_cocoa_y(float(loc.y)))
        t = time.monotonic_ns()
        start, end = window(t)
        decision = evaluate_click(self.tracker.samples_between(start, end), click, self.tracker.pt_per_deg)
        self._click_stats["seen"] += 1
        if not decision.accepted:
            log.debug("click ignored: %s", decision.reason)
            return
        self._click_stats["ok"] += 1
        profile.add(decision.features, click, CLICK)
        self._clicks_since_fit += 1
        if self._clicks_since_fit >= REFIT_EVERY_CLICKS and not self._fitting:
            self._clicks_since_fit = 0
            self._fitting = True
            threading.Thread(target=self._refit, args=(profile,), daemon=True).start()

    def _refit(self, profile: Profile) -> None:
        try:
            profile.fit()
            profile.save()
        finally:
            self._fitting = False

    # ---- menu ------------------------------------------------------------------------
    def _build_menu(self) -> None:
        actions = [self._calibrate, lambda: self._validate(9), self._measure, self._toggle_overlay,
                   self._toggle_clicks, self._toggle_snap, self._toggle_recording, self._quit]
        self._menu_target = _MenuTarget.alloc().initWithActions_(actions)
        menu = NSMenu.alloc().init()
        self._status_line = menu.addItemWithTitle_action_keyEquivalent_("Starting…", None, "")
        self._status_line.setEnabled_(False)
        menu.addItem_(NSMenuItem.separatorItem())
        for tag, (key, title) in enumerate(
            [("calibrate", "Recalibrate"), ("validate", "Check accuracy"), ("measure", "Measure (head still + moving)"),
             ("overlay", "Show gaze ring"),
             ("clicks", "Learn from clicks"), ("snap", "Snap to buttons"), ("record", "Record gaze"), ("quit", "Quit")]
        ):
            if key == "quit":
                menu.addItem_(NSMenuItem.separatorItem())
            item = menu.addItemWithTitle_action_keyEquivalent_(title, "run:", "")
            item.setTarget_(self._menu_target)
            item.setTag_(tag)
            self._menu_items[key] = item
        self._menu_items["overlay"].setState_(1)
        self._menu_items["clicks"].setState_(1 if self.learn_from_clicks else 0)
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self._status_item.button().setTitle_("◉")
        self._status_item.setMenu_(menu)

    def _update_status(self, _timer) -> None:
        p = self.tracker._last_point
        state = p.state.value.replace("_", " ") if p is not None else "starting"
        parts = [state.capitalize(), f"{self.tracker.fps:.0f} fps", f"{self.tracker.distance_mm / 10:.0f} cm"]
        prof = self.tracker.profile
        if prof is not None and prof.calibrated:
            parts.append(f"{prof.count(CLICK)} clicks learned")
        if self._last_result is not None and not np.isnan(self._last_result.accuracy_deg):
            parts.append(f"last check {self._last_result.accuracy_deg:.1f}°")
        self._status_line.setTitle_("  ·  ".join(parts))

    def _toggle_overlay(self) -> None:
        self.overlay_on = not self.overlay_on
        self._menu_items["overlay"].setState_(1 if self.overlay_on else 0)

    def _toggle_clicks(self) -> None:
        self.learn_from_clicks = not self.learn_from_clicks
        self._menu_items["clicks"].setState_(1 if self.learn_from_clicks else 0)

    def _toggle_snap(self) -> None:
        if not self.snap_on:
            if self._snapper is None:
                self._snapper = Snapper()
            if not self._snapper.available:
                self._snapper.request_permission()
                log.info("snapping needs Accessibility access for this terminal; it starts once granted")
        self.snap_on = not self.snap_on
        self._snap = None
        self._menu_items["snap"].setState_(1 if self.snap_on else 0)

    def _toggle_recording(self) -> None:
        if self.recorder is None:
            self._start_recording()
        else:
            self._stop_recording()
        self._menu_items["record"].setState_(1 if self.recorder is not None else 0)

    def _quit(self) -> None:
        self._shutdown()
        AppHelper.stopEventLoop()
