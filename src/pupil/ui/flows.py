"""Fullscreen flows: intro, calibration (dots + head movement), checks, re-centre and measurement.

`Flows` owns what's on the fullscreen target screen: whether one is up (`busy`) and which phase and target
it shows (`phase`, for saved sessions and the dashboard). Everything runs on the AppKit main thread.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
from AppKit import NSTimer

from ..calibration.profile import CLICK, EXPLICIT, MOTION, Profile, ProfileMeta
from ..calibration.targets import TargetTiming, grid, shuffled
from ..core.lid import LidSensor
from ..core.types import TrackingState
from ..eval.metrics import ValidationResult, append_result, score
from ..tracker import Tracker
from .overlay import GazeRing
from .target_screen import ESC, RETURN, SPACE, TargetScreen, TargetSequence

log = logging.getLogger(__name__)

LAUNCH_CHECK_MAX_DEG = 4.0  # saved calibration is reused if the quick check beats this
MIN_SAMPLES_PER_TARGET = 8
MOTION_TIMING = TargetTiming(dwell_s=6.5, settle_s=0.8, tail_s=0.1)
MOTION_CHUNK = 15  # score the head-motion sweep in half-second pieces

Pair = tuple[np.ndarray, np.ndarray]  # (calibration features, eyeball measurements) of a target's frames


class Flows:
    def __init__(self, tracker: Tracker, targets: TargetScreen, ring: GazeRing, lid: LidSensor, *,
                 profile_name: str, n_targets: int, results_csv: Path, quit: Callable[[], None],
                 session_label: Callable[[str | None, tuple | None, int], None] = lambda *_: None):
        self.tracker, self.targets, self.ring, self.lid = tracker, targets, ring, lid
        self.screen = tracker.screen
        self.profile_name, self.n_targets, self.results_csv = profile_name, n_targets, results_csv
        self.base_profile = profile_name
        self.quit, self.session_label = quit, session_label
        self.timing = TargetTiming()
        self.busy = False  # a fullscreen screen is up
        self.phase: tuple[str, tuple | None] = ("idle", None)  # what it shows
        self.last_result: ValidationResult | None = None
        self.pending_measure = False  # run the measurement once the launch flow finishes

    # ---- helpers ----------------------------------------------------------------------
    def label(self, phase: str | None, target=None, onset_ns: int = 0) -> None:
        self.phase = (phase or "idle", tuple(target) if target is not None else None)
        self.session_label(phase, target, onset_ns)

    def _begin(self) -> None:
        self.busy = True
        self.ring.hide()

    def _samples(self, onset_ns: int, timing: TargetTiming):
        start, end = timing.window_ns(onset_ns)
        self.tracker.wait_until(end, timeout=0.2)
        return self.tracker.samples_between(start, end)

    def _features(self, onset_ns: int, timing: TargetTiming) -> Pair:
        """Calibration features and eyeball measurements of the usable frames in a target's window."""
        samples = [s for s in self._samples(onset_ns, timing)
                   if s.calib is not None and s.geo is not None and not s.face.is_blinking]
        if not samples:
            return np.zeros((0, 0), np.float32), np.zeros((0, 0), np.float32)
        return np.stack([s.calib for s in samples]), np.stack([s.geo for s in samples])

    def _tracked_points(self, samples) -> list[tuple[float, float]]:
        return [(s.point.raw_x, s.point.raw_y) for s in samples if s.point.state is TrackingState.TRACKING]

    def message(self, lines, keys: dict, auto_continue: float | None = None) -> None:
        self.label(None)  # no target on screen: stop saving session frames
        state = {"done": False}

        def go(fn):
            if not state["done"]:
                state["done"] = True
                fn()

        self.targets.show(lines, on_key=lambda c: c in keys and go(keys[c]))
        if auto_continue:
            NSTimer.scheduledTimerWithTimeInterval_repeats_block_(auto_continue, False, lambda _t: go(keys[RETURN]))

    def when_idle(self, fn: Callable[[], None]) -> None:
        """Remote commands arrive any time; ignore them while a screen is already up."""
        if not self.busy:
            fn()

    def after_cancel(self) -> None:
        self.label(None)
        if self.tracker.profile is not None and self.tracker.profile.calibrated:
            self.done()
        else:
            self.quit()

    def done(self) -> None:
        """Leave the fullscreen screens and show the ring."""
        self.label(None)
        if self.pending_measure:
            self.pending_measure = False
            self.measure()
            return
        self.targets.hide()
        self.busy = False

    # ---- launch ------------------------------------------------------------------------
    def launch(self, force_recalibrate: bool) -> None:
        """Reuse a saved calibration after a quick check, or start calibrating."""
        model_id = self.tracker.model.model_id
        camera = self.tracker.source.source_id
        screen_pt = (self.screen.width_pt, self.screen.height_pt)
        self.profile_name = self._profile_name_for(camera)
        profile = None if force_recalibrate else Profile.load(self.profile_name, plane=self.tracker.plane)
        matches = profile is not None and profile.meta.matches(model_id, camera, screen_pt)
        if matches and profile.calibrated:
            log.info("loaded profile %r (%d targets, %d from clicks)", self.profile_name, profile.n_groups,
                     profile.count(CLICK))
            self.tracker.profile = profile
            self.validate(n=3, launch_check=True)
            return
        self.tracker.profile = profile if matches else None  # a matching one keeps its click samples
        self.intro()

    def _profile_name_for(self, camera: str) -> str:
        """One profile per camera: the base name belongs to the first camera calibrated, others get their own."""
        if Profile.saved_camera(self.base_profile) in (None, camera):
            return self.base_profile
        slug = re.sub(r"[^a-z0-9]+", "-", camera.lower()).strip("-") or "camera"
        return f"{self.base_profile}-{slug}"

    def intro(self) -> None:
        self._begin()

        def lines():
            last = self.tracker.samples_between(time.monotonic_ns() - 500_000_000, time.monotonic_ns())
            faces = [s.face for s in last if s.face is not None]
            if not faces:
                status = "Looking for your face…  make sure it's lit and in view of the camera"
            else:
                d = np.median([f.distance_mm for f in faces if f.distance_mm]) / 10
                hint = "  ·  a bit close" if d < 40 else "  ·  a bit far" if d > 85 else ""
                status = f"Face found  ·  {d:.0f} cm from the camera{hint}"
            if getattr(self.tracker.source, "downgraded", False):
                w, h = self.tracker.source.delivered
                status += f"  ·  camera is only giving {w}x{h}: close other apps using it"
            seconds = self.n_targets * self.timing.dwell_s + MOTION_TIMING.dwell_s
            return [
                "Look at each dot until it shrinks away",
                f"{self.n_targets} dots, then a few seconds turning your head (about {seconds:.0f} s). "
                "Sit the way you normally do.",
                status,
                "Press Space to start  ·  Esc to quit",
            ]

        def on_key(c):
            if c in (SPACE, ESC):
                timer.invalidate()
                (self.calibrate if c == SPACE else self.quit)()

        self.targets.show(lines(), on_key=on_key)
        timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.25, True, lambda _t: self.targets.set_lines(lines()))

    # ---- calibration ----------------------------------------------------------------------
    def calibrate(self) -> None:
        self._begin()
        collected: dict[tuple, Pair] = {}
        points = shuffled(grid(self.n_targets, self.screen.width_pt, self.screen.height_pt))

        def on_target(target, onset_ns, end_ns):
            self.label(None)
            feats, geos = self._features(onset_ns, self.timing)
            if len(feats) >= MIN_SAMPLES_PER_TARGET:
                collected[target] = (feats, geos)

        def on_done(cancelled):
            if cancelled:
                self.after_cancel()
            elif len(collected) < len(points) - len(points) // 3:
                self.message(
                    ["Couldn't see your eyes clearly",
                     f"Got {len(collected)} of {len(points)} dots. Check lighting and that your face is in view.",
                     "Press Space to try again  ·  Esc to quit"],
                    {SPACE: self.calibrate, ESC: self.quit},
                )
            else:
                self._head_motion(collected)

        TargetSequence(self.targets, points, self.timing, on_target, on_done,
                       on_start=lambda t, onset: self.label("calib", t, onset)).start()

    def _head_motion(self, dots: dict[tuple, Pair]) -> None:
        """Hold gaze on one dot while turning the head, so the fit learns to separate head movement from
        eye movement."""
        centre = (self.screen.width_pt / 2, self.screen.height_pt / 2)
        sweep: list[Pair] = []

        def on_target(target, onset_ns, end_ns):
            self.label(None)
            feats, geos = self._features(onset_ns, MOTION_TIMING)
            if len(feats):
                sweep.append((feats, geos))

        def on_done(cancelled):
            if cancelled:
                self.after_cancel()
            else:
                self._finish_calibration(dots, centre, sweep[0] if sweep else None)

        lines = ["Keep looking at the dot", "and slowly turn your head: left, right, up, down"]
        TargetSequence(self.targets, [centre], MOTION_TIMING, on_target, on_done,
                       on_start=lambda t, onset: self.label("motion", t, onset), lines=lines).start()

    def _profile_for(self, dim: int) -> Profile:
        """The current profile, or a new one if there is none or its features differ."""
        p = self.tracker.profile
        if p is None or p.X.shape[1] != dim:
            meta = ProfileMeta(self.profile_name, self.tracker.model.model_id, self.tracker.source.source_id,
                               (self.screen.width_pt, self.screen.height_pt))
            p = Profile(meta, dim)
        p.plane = self.tracker.plane
        return p

    def _finish_calibration(self, dots: dict[tuple, Pair], centre, sweep: Pair | None) -> None:
        profile = self._profile_for(next(iter(dots.values()))[0].shape[1])
        profile.clear(EXPLICIT, MOTION)  # click samples are kept
        profile.meta.lid_deg = self.lid.angle()
        profile.meta.offset_pt = (0.0, 0.0)  # a fresh calibration needs no drift correction
        for target, (feats, geos) in dots.items():
            profile.add(feats, target, EXPLICIT, measures=geos)
        if sweep is not None and len(sweep[0]) >= MIN_SAMPLES_PER_TARGET:
            profile.add(sweep[0], centre, MOTION, chunk=MOTION_CHUNK, measures=sweep[1])
        if profile.fit() is None:
            self.message(["Calibration failed", "Press Space to try again"], {SPACE: self.calibrate, ESC: self.quit})
            return
        profile.save()
        self.tracker.profile = profile
        self.validate(n=3)

    # ---- checks ------------------------------------------------------------------------
    def validate(self, n: int = 9, launch_check: bool = False, phase: str = "valid", then=None,
                 lines: list[str] | None = None) -> None:
        """Show `n` dots and score gaze against them. `then(result)` replaces the results screen, for
        chaining checks; `phase` labels the frames in a saved session."""
        self._begin()
        points = shuffled(grid(n, self.screen.width_pt, self.screen.height_pt))
        per_target: dict[tuple, list] = {}
        totals = {"frames": 0}
        distances: list[float] = []

        def on_target(target, onset_ns, end_ns):
            self.label(None)
            samples = self._samples(onset_ns, self.timing)
            totals["frames"] += len(samples)
            per_target[target] = self._tracked_points(samples)
            distances.extend(s.point.distance_mm for s in samples if s.point.distance_mm)

        def on_done(cancelled):
            if cancelled:
                self.after_cancel()
                return
            dist = float(np.median(distances)) if distances else self.tracker.distance_mm
            result = score(per_target, self.screen, dist, totals["frames"])
            self.last_result = result
            log.info("validation (%d targets): %s", n, result.summary())
            self._log_result(result, n, "launch" if launch_check else phase if phase != "valid" else
                             "quick" if n <= 3 else "manual")
            if launch_check and not (result.accuracy_deg <= LAUNCH_CHECK_MAX_DEG):
                self._offer_recovery(result)
            elif then is not None:
                then(result)
            else:
                self._show_result(result, n)

        TargetSequence(self.targets, points, self.timing, on_target, on_done,
                       on_start=lambda t, onset: self.label(phase, t, onset), lines=lines).start()

    def _log_result(self, result: ValidationResult, n: int, kind: str) -> None:
        profile = self.tracker.profile
        try:
            append_result(self.results_csv, result, targets=n, kind=kind,
                          mode=getattr(getattr(self.tracker.source, "mode", None), "name", ""),
                          model=self.tracker.model.model_id,
                          fit=profile.report.chosen if profile is not None and profile.report else "")
        except OSError:
            log.exception("could not write results")

    def _offer_recovery(self, result: ValidationResult) -> None:
        log.info("saved calibration is off (%.1f°)", result.accuracy_deg)
        off = "no data" if np.isnan(result.accuracy_deg) else f"{result.accuracy_deg:.1f}°"
        self.message(
            [f"Your saved calibration is off ({off})",
             "If you've just moved a little, a 2-second re-centre may be enough.",
             "Space: re-centre (1 dot)  ·  Return: full calibration  ·  Esc: use it anyway"],
            # Esc dismisses, as everywhere else: quitting here twice took voice control down with it.
            {SPACE: self.recentre, RETURN: self.intro, ESC: self.done},
        )

    def _show_result(self, result: ValidationResult, n: int) -> None:
        acc = ("no data" if np.isnan(result.accuracy_deg)
               else f"{result.accuracy_deg:.1f}°  (about {result.accuracy_pt:.0f} pt)")
        title = f"Quick check ({n} dots): {acc}" if n <= 3 else f"Accuracy ({n} dots): {acc}"
        hint = "  ·  Measure from the ◉ menu for a full check" if n <= 3 else ""
        self.message([title, result.summary() + hint, "Return to start  ·  Space to recalibrate"],
                     {RETURN: self.done, SPACE: self.calibrate, ESC: self.done},
                     auto_continue=None if n > 3 else 2.5)

    def recentre(self) -> None:
        """Drift correction: one centre dot; shift every prediction so it lands on the dot."""
        profile = self.tracker.profile
        if profile is None or not profile.calibrated:
            self.intro()
            return
        self._begin()
        centre = (self.screen.width_pt / 2, self.screen.height_pt / 2)
        got: list[tuple[float, float]] = []

        def on_target(target, onset_ns, end_ns):
            self.label(None)
            got.extend(self._tracked_points(self._samples(onset_ns, self.timing)))

        def on_done(cancelled):
            if cancelled:
                self.after_cancel()
                return
            if len(got) < MIN_SAMPLES_PER_TARGET:
                self.message(["Couldn't see your eyes", "Space: try again  ·  Return: full calibration"],
                             {SPACE: self.recentre, RETURN: self.intro, ESC: self.done})
                return
            before = profile.meta.offset_pt
            try:
                after = profile.recentre(np.array(got), centre)
            except ValueError as e:
                log.info("re-centre refused: %s", e)
                self.message(["Too far off for a re-centre", "Press Space for a full calibration"],
                             {SPACE: self.intro, ESC: self.done})
                return
            profile.save()
            log.info("re-centred by %.0f pt (offset now %.0f, %.0f)",
                     float(np.hypot(after[0] - before[0], after[1] - before[1])), *after)
            self.validate(n=3)

        TargetSequence(self.targets, [centre], self.timing, on_target, on_done,
                       on_start=lambda t, onset: self.label("recentre", t, onset), lines=["Look at the dot"]).start()

    def measure(self) -> None:
        """Two 9-dot checks: head still, then head moving naturally."""
        if self.tracker.profile is None or not self.tracker.profile.calibrated:
            return
        self._begin()
        results: dict[str, ValidationResult] = {}

        def fmt(r: ValidationResult) -> str:
            if np.isnan(r.accuracy_deg):
                return "no data"
            return f"{r.accuracy_deg:.1f}° (p95 {r.accuracy_p95_deg:.1f}°, jitter {r.precision_rms_deg:.2f}°)"

        def show_results(moving):
            still = results["still"]
            self.message(["Measurement done", f"Head still:  {fmt(still)}", f"Head moving:  {fmt(moving)}",
                          f"Distance {still.distance_mm / 10:.0f} cm  ·  Return to continue"],
                         {RETURN: self.done, ESC: self.done})

        def moving_intro(still):
            results["still"] = still
            self.message(["Now move your head naturally",
                          "Follow the dots with your eyes while turning and tilting your head a little.",
                          "Press Space to start"],
                         {SPACE: lambda: self.validate(9, phase="valid_moving", then=show_results), ESC: self.done})

        self.message(["Accuracy measurement",
                      "First 9 dots: keep your head as still as you can. Then 9 more while moving it.",
                      "Press Space to start  ·  Esc to cancel"],
                     {SPACE: lambda: self.validate(9, phase="valid_still", then=moving_intro), ESC: self.done})
