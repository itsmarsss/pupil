"""Menu-bar app: wires the tracker to the ring, the fullscreen flows, services, menu and outputs.

Everything here runs on the AppKit main thread. The tracker runs inference on its own thread; the ring
reads its latest point from a 60 Hz timer instead of calling into AppKit from that thread.
"""
from __future__ import annotations

import json
import logging
import signal
import time
from pathlib import Path

import numpy as np
from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSEvent, NSEventMaskLeftMouseDown, NSTimer
from PyObjCTools import AppHelper

from ..calibration.profile import CLICK
from ..calibration.targets import TargetTiming
from ..core.lid import LidMonitor
from ..core.paths import APP_SUPPORT
from ..eval.session import SessionWriter
from ..tracker import Tracker
from .flows import MOTION_TIMING, Flows
from .menu import MenuBar
from .overlay import GazeRing
from .services import ClickLearner, Recording, SnapService, VoiceLog
from .target_screen import TargetScreen

log = logging.getLogger(__name__)

LID_MOVED_DEG = 4.0  # hinge change since calibration that flags the fit as suspect
SMOOTHING_ORDER = ["off", "light", "medium", "strong"]


class GazeApp:
    def __init__(
        self,
        tracker: Tracker,
        profile_name: str = "default",
        n_targets: int = 16,
        recalibrate: bool = False,
        results_csv: Path | None = None,
        learn_from_clicks: bool = True,
        snap: bool = False,
        record_path: Path | None = None,
        session_dir: Path | None = None,
        measure: bool = False,
        publish: bool = False,
        dashboard_port: int | None = None,
    ):
        self.tracker = tracker
        self.screen = tracker.screen
        self.options = dict(profile_name=profile_name, n_targets=n_targets, recalibrate=recalibrate,
                            results_csv=results_csv or APP_SUPPORT / "validation.csv", snap=snap,
                            record_path=record_path, session_dir=session_dir, measure=measure,
                            publish=publish, dashboard_port=dashboard_port)
        self.overlay_on = True
        self.lid = LidMonitor(hz=30.0, on_change=self._on_lid)  # hinge angle at the camera's rate
        self.clicks = ClickLearner(tracker, learn_from_clicks)
        self.snapping = SnapService(tracker)
        self.recording = Recording(tracker)
        self.voice = VoiceLog()  # what the voice controller heard and did, for the dashboard
        self.session: SessionWriter | None = None
        self.dashboard = None
        self.flows: Flows | None = None
        self.ring: GazeRing | None = None
        self.menu: MenuBar | None = None
        self._last_valid_ns = 0
        self._shut_down = False

    # ---- lifecycle --------------------------------------------------------------------
    def run(self) -> None:
        o = self.options
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.ring = GazeRing(self.screen)
        if o["session_dir"] is not None:
            self._start_session(o["session_dir"])
        self.flows = Flows(self.tracker, TargetScreen(self.screen), self.ring, self.lid.sensor,
                           profile_name=o["profile_name"], n_targets=o["n_targets"], results_csv=o["results_csv"],
                           quit=self.quit, session_label=self._session_label)
        self.flows.pending_measure = o["measure"]
        self._build_menu()
        if o["snap"]:
            self._toggle_snap()
        if o["record_path"] is not None:
            self.recording.start(o["record_path"])
            self.menu.set_checked("record", True)
        if o["dashboard_port"] is not None:
            self._start_dashboard(o["dashboard_port"])
        if o["publish"]:
            self._start_publisher()
        self.tracker.start()
        self.lid.start()
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 60, True, self._tick)
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1.0, True, self._update_status)
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskLeftMouseDown, self._on_click)
        # start the launch flow once the run loop is going
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.1, False, lambda _t: self.flows.launch(o["recalibrate"]))
        # Python only runs signal handlers while the main thread executes Python code; the 1 s status timer
        # guarantees that, so Ctrl+C and kill quit cleanly.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.quit())
        try:
            AppHelper.runEventLoop()
        finally:
            self._shutdown()

    def quit(self) -> None:
        self._shutdown()
        AppHelper.stopEventLoop()

    def _shutdown(self) -> None:
        """Idempotent: stopping the AppKit loop may exit the process without unwinding, so every quit path
        calls this first."""
        if self._shut_down:
            return
        self._shut_down = True
        self.tracker.stop()
        self.lid.stop()
        if self.dashboard is not None:
            self.dashboard.stop()
        self.recording.stop()
        self._close_session()
        if self.tracker.profile is not None and self.tracker.profile.calibrated:
            self.tracker.profile.save()

    # ---- outputs ----------------------------------------------------------------------
    def _start_session(self, root: Path) -> None:
        self.session = SessionWriter(root, {
            "screen_pt": [self.screen.width_pt, self.screen.height_pt],
            "screen_mm": [self.screen.width_mm, self.screen.height_mm],
            "timing": vars(TargetTiming()),
            "motion_timing": vars(MOTION_TIMING),
            "mode": getattr(getattr(self.tracker.source, "mode", None), "name", ""),
            "model": self.tracker.model.model_id,
            "camera": self.tracker.source.source_id,
            "n_targets": self.options["n_targets"],
            "lid_deg": self.lid.angle,
        })
        self.session.context["lid"] = self.lid.angle
        self.tracker.add_frame_listener(self.session.on_frame)
        log.info("saving calibration/validation frames to %s", root)

    def _session_label(self, phase, target, onset_ns) -> None:
        if self.session is not None:
            self.session.label(phase, target, onset_ns)

    def _close_session(self) -> None:
        if self.session is None:
            return
        delivered = getattr(self.tracker.source, "delivered", None)
        if delivered:  # record what the camera actually sent
            meta_path = self.session.root / "meta.json"
            meta = json.loads(meta_path.read_text())
            meta["delivered"] = list(delivered)
            meta_path.write_text(json.dumps(meta, indent=2))
        self.session.close()
        log.info("session: %d frames in %s", self.session.frames, self.session.root)

    def _remote_commands(self) -> dict:
        """Commands from the dashboard or the voice controller, run on the main thread."""
        f = self.flows

        def on_main(fn):
            return lambda: AppHelper.callAfter(fn)

        return {
            "recalibrate": on_main(lambda: f.when_idle(f.calibrate)),
            "recentre": on_main(lambda: f.when_idle(f.recentre)),
            "check": on_main(lambda: f.when_idle(lambda: f.validate(9))),
            "validate": on_main(lambda: f.when_idle(lambda: f.validate(9))),
            "measure": on_main(lambda: f.when_idle(f.measure)),
            "cycle_smoothing": on_main(self._cycle_smoothing),
            "toggle_overlay": on_main(self._toggle_overlay),
        }

    def _start_dashboard(self, port: int) -> None:
        from ..dashboard.feeds import LiveFeed
        from ..dashboard.server import DashboardServer

        feed = LiveFeed(self.tracker, context=self._dashboard_context, lid=lambda: self.lid.angle)
        self.dashboard = DashboardServer(feed, commands=self._remote_commands(), port=port)
        self.dashboard.start()

    def _dashboard_context(self) -> dict:
        phase, target = self.flows.phase
        return {"phase": phase, "target": target, "busy": self.flows.busy, "lid_deg": self.lid.angle,
                "smoothing": self.tracker.smoothing, "overlay": self.overlay_on, "voice": self.voice.snapshot()}

    def _start_publisher(self) -> None:
        from ..sinks.publisher import GazePublisher

        publisher = GazePublisher(busy=lambda: self.flows.busy)
        self.tracker.add_listener(publisher.on_sample)
        commands = self._remote_commands()
        publisher.listen({k: commands[k] for k in ("recalibrate", "recentre", "validate")}, on_event=self.voice.add)
        log.info("publishing gaze to the controller over UDP")

    # ---- per frame ------------------------------------------------------------------------
    def _tick(self, _timer) -> None:
        if self.flows.busy or not self.overlay_on:
            self.ring.hide()
            return
        p = self.tracker.last_point
        now = time.monotonic_ns()
        if p is not None and p.valid:
            self._last_valid_ns = now
            confidence = min(p.confidence, 0.3) if self.lid_moved else p.confidence
            element = self.snapping.element_for(p)
            if element is not None:
                self.ring.show_at(*element.centre, fixating=True, confidence=confidence)
            else:
                self.ring.show_at(p.x, p.y, fixating=p.fixation_id is not None, confidence=confidence)
        elif now - self._last_valid_ns > 400_000_000:  # brief dropouts (blinks) keep the ring
            self.ring.hide()

    def _on_click(self, _event) -> None:
        loc = NSEvent.mouseLocation()
        click = (float(loc.x), self.screen.to_cocoa_y(float(loc.y)))
        self.recording.click(click)
        if not self.flows.busy:
            self.clicks.on_click(click)

    def _on_lid(self, angle: float) -> None:
        """Lid moved (called from the monitor's thread): saved sessions record it per frame."""
        if self.session is not None:
            self.session.context["lid"] = angle

    @property
    def lid_moved(self) -> float | None:
        """Degrees the lid has moved since calibration, if beyond LID_MOVED_DEG."""
        prof = self.tracker.profile
        if prof is None or prof.meta.lid_deg is None or self.lid.angle is None:
            return None
        delta = self.lid.angle - prof.meta.lid_deg
        return delta if abs(delta) > LID_MOVED_DEG else None

    # ---- menu ------------------------------------------------------------------------
    def _build_menu(self) -> None:
        f = self.flows
        self.menu = MenuBar([
            ("calibrate", "Recalibrate", f.calibrate),
            ("recentre", "Re-centre (1 dot)", f.recentre),
            ("validate", "Check accuracy", lambda: f.validate(9)),
            ("measure", "Measure (head still + moving)", f.measure),
            ("overlay", "Show gaze ring", self._toggle_overlay),
            ("smoothing", f"Smoothing: {self.tracker.smoothing}", self._cycle_smoothing),
            ("clicks", "Learn from clicks", self._toggle_clicks),
            ("snap", "Snap to buttons", self._toggle_snap),
            ("record", "Record gaze", self._toggle_recording),
            ("-", "", None),
            ("quit", "Quit", self.quit),
        ])
        self.menu.set_checked("overlay", self.overlay_on)
        self.menu.set_checked("clicks", self.clicks.enabled)

    def _update_status(self, _timer) -> None:
        p = self.tracker.last_point
        parts = [(p.state.value.replace("_", " ") if p is not None else "starting").capitalize(),
                 f"{self.tracker.fps:.0f} fps", f"{self.tracker.distance_mm / 10:.0f} cm"]
        prof = self.tracker.profile
        if prof is not None and prof.calibrated:
            parts.append(f"{prof.count(CLICK)} clicks learned")
        last = self.flows.last_result
        if last is not None and not np.isnan(last.accuracy_deg):
            parts.append(f"last check {last.accuracy_deg:.1f}°")
        if self.lid_moved:
            parts.append(f"lid moved {self.lid_moved:+.0f}° since calibration: check accuracy")
        self.menu.set_status("  ·  ".join(parts))

    def _toggle_overlay(self) -> None:
        self.overlay_on = not self.overlay_on
        self.menu.set_checked("overlay", self.overlay_on)

    def _cycle_smoothing(self) -> None:
        name = SMOOTHING_ORDER[(SMOOTHING_ORDER.index(self.tracker.smoothing) + 1) % len(SMOOTHING_ORDER)]
        self.tracker.set_smoothing(name)
        self.menu.set_title("smoothing", f"Smoothing: {name}")

    def _toggle_clicks(self) -> None:
        self.clicks.enabled = not self.clicks.enabled
        self.menu.set_checked("clicks", self.clicks.enabled)

    def _toggle_snap(self) -> None:
        self.menu.set_checked("snap", self.snapping.toggle())

    def _toggle_recording(self) -> None:
        self.menu.set_checked("record", self.recording.toggle())
