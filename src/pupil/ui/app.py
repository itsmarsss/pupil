"""Menu-bar app: wires the tracker to the ring, the fullscreen flows, services, menu and outputs.

Everything here runs on the AppKit main thread. The tracker runs inference on its own thread; the ring
reads its latest point from a 60 Hz timer instead of calling into AppKit from that thread.
"""
from __future__ import annotations

import json
import logging
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from AppKit import (NSURL, NSApplication, NSApplicationActivationPolicyAccessory, NSEvent, NSEventMaskLeftMouseDown,
                    NSTimer, NSWorkspace)
from PyObjCTools import AppHelper

from ..calibration.profile import CLICK
from ..calibration.targets import TargetTiming
from ..capture.camera import CameraSource, list_cameras
from ..core.lid import LidMonitor
from ..core.paths import APP_SUPPORT
from ..core.types import TrackingState
from ..eval.session import SessionWriter
from ..health.tracker import HealthTracker
from ..tracker import Tracker
from .flows import MOTION_TIMING, Flows
from .menu import SEPARATOR, Action, Choice, Header, MenuBar, Toggle
from .overlay import PupilRing
from .services import ClickLearner, Recording, SnapService, VoiceLog
from .target_screen import TargetScreen
from ..core.env import env

log = logging.getLogger(__name__)

LID_MOVED_DEG = 4.0  # hinge change since calibration that flags the fit as suspect
SMOOTHING_ORDER = ["off", "light", "medium", "strong"]
HEAT_WINDOW_NS = 600_000_000  # the on-screen heat map shows this much recent gaze
HEAT_MAX_POINTS = 24
DASHBOARD_PORT = 8766


class PupilApp:
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
        voice: bool = False,
        voice_typed: bool = False,
    ):
        self.tracker = tracker
        self.screen = tracker.screen
        self.options = dict(profile_name=profile_name, n_targets=n_targets, recalibrate=recalibrate,
                            results_csv=results_csv or APP_SUPPORT / "validation.csv", snap=snap,
                            record_path=record_path, session_dir=session_dir, measure=measure,
                            publish=publish, dashboard_port=dashboard_port, voice=voice, voice_typed=voice_typed)
        self.overlay_on = True
        self.lid = LidMonitor(hz=30.0, on_change=self._on_lid)  # hinge angle at the camera's rate
        self.clicks = ClickLearner(tracker, learn_from_clicks)
        self.snapping = SnapService(tracker)
        self.recording = Recording(tracker)
        self.voice = VoiceLog()  # what the voice controller heard and did, for the dashboard
        self.health = HealthTracker(log_dir=APP_SUPPORT / "health",  # blinks, eye fatigue, breaks
                                    pt_per_deg=lambda: self.tracker.pt_per_deg)
        self._health_now: dict = {}
        self.notch = None  # the user-facing status (ui/notch.py)
        self.session: SessionWriter | None = None
        self.dashboard = None
        self.controller = None  # voice control (pupil.control), when on
        self.flows: Flows | None = None
        self.ring: PupilRing | None = None
        self.menu: MenuBar | None = None
        self._last_valid_ns = 0
        self._switching_camera = False
        self._shut_down = False

    # ---- lifecycle --------------------------------------------------------------------
    def run(self) -> None:
        o = self.options
        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        self.ring = PupilRing(self.screen)
        self.ring.heat = env("DISPLAY", "heat") == "heat"  # ring | heat (◉ → Look display)
        if o["session_dir"] is not None:
            self._start_session(o["session_dir"])
        self.flows = Flows(self.tracker, TargetScreen(self.screen), self.ring, self.lid.sensor,
                           profile_name=o["profile_name"], n_targets=o["n_targets"], results_csv=o["results_csv"],
                           quit=self.quit, session_label=self._session_label)
        self.flows.pending_measure = o["measure"]
        self._build_menu()
        if env("NOTCH", "1") != "0":
            from .notch import Notch

            self.notch = Notch(on_toggle_voice=self._toggle_voice if not o["voice_typed"] else None)
            NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 30, True, self._notch_tick)
        if o["snap"]:
            self.snapping.toggle()
        if o["record_path"] is not None:
            self.recording.start(o["record_path"])
        if o["dashboard_port"] is not None:
            self._open_dashboard()  # starts it, then opens it at the port it got
        if o["publish"]:
            self._start_publisher()
        if o["voice"]:
            self._start_voice(o["voice_typed"])
        self.tracker.add_listener(self.health.on_sample)
        self.tracker.start()
        self.lid.start()
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 60, True, self._tick)
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1.0, True, self._health_tick)
        NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskLeftMouseDown, self._on_click)
        # start the launch flow once the run loop is going
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(0.1, False, lambda _t: self._launch())
        # Python only runs signal handlers while the main thread executes Python code; the 1 s health timer
        # guarantees that, so Ctrl+C and kill quit cleanly.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.quit())
        try:
            AppHelper.runEventLoop()
        finally:
            self._shutdown()

    def _launch(self) -> None:
        """The launch flow; an exception in an NSTimer block vanishes silently, which once left the app sitting
        uncalibrated with no screen up, so log it and fall back to a fresh calibration."""
        try:
            self.flows.launch(self.options["recalibrate"])
        except Exception:
            log.exception("launch failed; starting a fresh calibration")
            self.flows.launch(True)

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
            "clear_research": on_main(self.voice.clear_research),
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
                "smoothing": self.tracker.smoothing, "overlay": self.overlay_on, "voice": self.voice.snapshot(),
                "gaze_sigma_pt": round(self.gaze_sigma_pt(), 1), "health": self._health_now}

    def _register_settings(self) -> None:
        """What the voice agent may change for the user (control/settings.py); changes run on the main thread."""
        from ..control import settings

        def on_main(fn):
            return lambda value: AppHelper.callAfter(fn, value) or None

        def camera(name):
            names = list_cameras()
            self._switch_camera(names.index(name))

        settings.register("camera", "which camera tracks the eyes (each keeps its own calibration)",
                          lambda: self.tracker.source.source_id, on_main(camera),
                          lambda: list_cameras() if isinstance(self.tracker.source, CameraSource) else [])
        settings.register("gaze_display", "how gaze is drawn on screen", self._display, on_main(self._set_display),
                          lambda: ["ring", "heat", "off"])
        settings.register("smoothing", "how much gaze is smoothed (more: steadier, slower)",
                          lambda: self.tracker.smoothing, on_main(self.tracker.set_smoothing), lambda: SMOOTHING_ORDER)
        settings.register("health_reminders", "blink and 20-20-20 break reminders",
                          lambda: "on" if self.health.reminders_on else "off",
                          lambda v: setattr(self.health, "reminders_on", v == "on"), lambda: ["on", "off"])
        settings.register("learn_from_clicks", "improve calibration from where you click",
                          lambda: "on" if self.clicks.enabled else "off",
                          on_main(lambda v: setattr(self.clicks, "enabled", v == "on")), lambda: ["on", "off"])

    def _start_voice(self, typed: bool) -> None:
        """Voice control in this process: it reads gaze from the tracker and drives the flows directly."""
        try:
            from ..control.controller import VoiceController
            from ..control.sources import TrackerGaze
        except ImportError as e:
            log.error("voice control needs its extras (uv sync --extra control): %s", e)
            return
        commands = self._remote_commands()
        gaze = TrackerGaze(busy=lambda: self.flows.busy, report=self.voice.add, sigma=self.gaze_sigma_pt,
                           commands={k: commands[k] for k in ("recalibrate", "recentre", "validate")})
        self.tracker.add_listener(gaze.on_sample)
        self._register_settings()
        self.controller = VoiceController(gaze, typed=typed)
        self.controller.start()
        NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 30, True, lambda _t: self.controller.tick())
        log.info("voice control on%s", " (typed)" if typed else "")

    def _start_publisher(self) -> None:
        from ..sinks.publisher import PupilPublisher

        publisher = PupilPublisher(busy=lambda: self.flows.busy)
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
            if self.ring.heat:  # the full heat map: the last HEAT_WINDOW of raw gaze, blurred by its accuracy
                recent = self.tracker.samples_between(now - HEAT_WINDOW_NS, now)
                points = [(s.point.raw_x, s.point.raw_y, s.point.confidence) for s in recent
                          if s.point.valid and s.point.raw_x is not None]
                self.ring.show_cloud(points[-HEAT_MAX_POINTS:], self.gaze_sigma_pt())
                return
            element = self.snapping.element_for(p)
            if element is not None and not self.ring.heat:  # the heat map shows raw gaze: the blur is the point
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

    def _health_tick(self, _timer) -> None:
        """Once a second: eye-health reminders, as a notification and (with voice control) a caption at the gaze."""
        self._health_now = self.health.snapshot()  # for the dashboard (every frame) without recomputing it
        for message in self.health.due_reminders():
            log.info("health: %s", message)
            subprocess.Popen(["osascript", "-e", f'display notification "{message}" with title "Eyes"'])
            if self.controller is not None:
                self.controller.indicator.say(message)
                self.controller.indicator._caption_end = time.time() + 6

    def _notch_tick(self, _timer) -> None:
        try:
            self.notch.update(self._notch_model(), self._gaze_now(), hidden=self.flows.busy)
        except Exception:  # an exception in an NSTimer block vanishes silently: say so, once
            if not getattr(self, "_notch_failed", False):
                self._notch_failed = True
                log.exception("notch update failed")

    def _gaze_now(self):
        p = self.tracker.last_point
        return (p.x, p.y) if p is not None and p.valid else None

    def _notch_model(self):
        from ..control import models
        from .notch_content import NotchModel

        p = self.tracker.last_point
        prof = self.tracker.profile
        tracking = ("calibrating" if self.flows.busy else "starting" if p is None
                    else "no face" if p.state == TrackingState.NO_FACE
                    else "uncalibrated" if p.state == TrackingState.UNCALIBRATED or prof is None or not prof.calibrated
                    else "tracking")
        last = self.flows.last_result
        m = NotchModel(tracking=tracking,
                       accuracy_deg=float(last.accuracy_deg) if last is not None and np.isfinite(last.accuracy_deg) else None,
                       distance_cm=self.tracker.distance_mm / 10 if self.tracker.distance_mm else None,
                       fps=self.tracker.fps)
        c = self.controller
        if c is not None:
            status = c._indicator_status()
            ind = c.indicator
            m.voice_on = c.enabled
            m.state, m.label = (status.state, status.label) if status else (None, "")
            m.heard = c.hud.heard.rstrip("…") if c.hud.heard else ""
            m.caption = (ind._caption, ind._caption_start, ind._caption_end) if ind._caption else None
            m.pending = c.pending_text or ("let the agent continue" if c.agent.paused else None)
            m.mic = c.hud.mic_device or ""
            m.activity = list(c.activity)
            active = next((r for r in models.snapshot() if r.get("active") and r["role"] != "stt"), None)
            m.model = None if active is None else "Jev" if active["role"] == "jev" else active["model"]
        return m

    def gaze_sigma_pt(self) -> float:
        """How wide the gaze heat map is right now: the last measured accuracy (launch check, validation or
        re-centre check) at the current distance, as a Gaussian width in screen points."""
        from ..control.heat import DEFAULT_SIGMA_DEG, sigma_from_accuracy

        last = self.flows.last_result if self.flows is not None else None
        deg = last.accuracy_deg if last is not None and np.isfinite(last.accuracy_deg) else DEFAULT_SIGMA_DEG
        if self.lid_moved:
            deg *= 1.5  # the lid has moved since calibration: less sure
        return sigma_from_accuracy(self.screen.deg_to_pt(deg, self.tracker.distance_mm))

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
        self.menu = MenuBar(self._menu_items, status=self._status_line)

    def _menu_items(self) -> list:
        """The ◉ menu, rebuilt each time it opens."""
        f = self.flows
        idle = lambda: not f.busy and not self._switching_camera  # noqa: E731
        items = [
            Header("Calibration"),
            Action("Recalibrate", f.calibrate, idle),
            Action("Re-centre (1 dot)", f.recentre, idle),
            Action("Check accuracy", lambda: f.validate(9), idle),
            Action("Measure (head still + moving)", f.measure, idle),
            SEPARATOR,
            Header("View"),
            Choice("Look display", lambda: [(title, self._display() == key, lambda key=key: self._set_display(key))
                                            for key, title in (("ring", "Ring"), ("heat", "Heat map (shows accuracy)"),
                                                               ("off", "Off"))]),
            Choice("Smoothing", lambda: [(name.capitalize(), name == self.tracker.smoothing,
                                          lambda name=name: self.tracker.set_smoothing(name))
                                         for name in SMOOTHING_ORDER]),
            Toggle("Snap to buttons", lambda: self.snapping.enabled, self.snapping.toggle),
            Toggle("Notch", lambda: self.notch is not None and self.notch.enabled, self._toggle_notch),
            Action("Open dashboard", self._open_dashboard),
            SEPARATOR,
            Header("Input"),
            Choice("Camera", self._camera_options),
        ]
        if not self.options["voice_typed"]:
            items.append(Toggle("Voice control", lambda: self.controller is not None and self.controller.enabled,
                                self._toggle_voice))
        voice = self.controller.voice if self.controller is not None else None
        if voice is not None and self.controller.enabled:
            items.append(Choice("Microphone", lambda: self._mic_options(voice)))
            items.append(Toggle("Voice isolation (default mic)", lambda: voice.isolation,
                                lambda: voice.set_isolation(not voice.isolation)))
        speaker = self.controller.speaker if self.controller is not None else None
        if speaker is not None and speaker.available:
            items.append(Choice("Agent voice", lambda: self._voice_options(speaker)))
        items += [
            SEPARATOR,
            Header("Health"),
            Action(lambda: self.health.status_line(), lambda: None, lambda: False),
            Toggle("Blink & break reminders", lambda: self.health.reminders_on,
                   lambda: setattr(self.health, "reminders_on", not self.health.reminders_on)),
            SEPARATOR,
            Header("Learning & data"),
            Toggle("Learn from clicks", lambda: self.clicks.enabled, self._toggle_clicks),
            Toggle("Record gaze", lambda: self.recording.on, self.recording.toggle),
            SEPARATOR,
            Action("Quit", self.quit),
        ]
        return items

    def _status_line(self) -> str:
        if self._switching_camera:
            return "Switching camera…"
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
        return "  ·  ".join(parts)

    def _toggle_notch(self) -> None:
        if self.notch is not None:
            self.notch.enabled = not self.notch.enabled

    def _toggle_overlay(self) -> None:
        self.overlay_on = not self.overlay_on

    def _display(self) -> str:
        return "off" if not self.overlay_on else "heat" if self.ring.heat else "ring"

    def _set_display(self, key: str) -> None:
        self.overlay_on = key != "off"
        self.ring.heat = key == "heat"

    def _cycle_smoothing(self) -> None:
        name = SMOOTHING_ORDER[(SMOOTHING_ORDER.index(self.tracker.smoothing) + 1) % len(SMOOTHING_ORDER)]
        self.tracker.set_smoothing(name)

    def _toggle_clicks(self) -> None:
        self.clicks.enabled = not self.clicks.enabled

    def _open_dashboard(self) -> None:
        if self.dashboard is None:
            self._start_dashboard(self.options["dashboard_port"] or DASHBOARD_PORT)
        NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(self.dashboard.url))

    # ---- inputs --------------------------------------------------------------------------
    def _camera_options(self) -> list:
        source = self.tracker.source
        if not isinstance(source, CameraSource):  # replaying a file
            return [(source.source_id, True, lambda: None)]
        return [(name, name == source.source_id, lambda i=i: self._switch_camera(i))
                for i, name in enumerate(list_cameras())]

    def _switch_camera(self, index: int) -> None:
        """Reopen the tracker on another camera (off the main thread: opening one takes a second), then load
        that camera's calibration or start one; each camera keeps its own profile."""
        old = self.tracker.source
        if self.flows.busy or self._switching_camera or list_cameras()[index] == old.source_id:
            return
        prof = self.tracker.profile
        if prof is not None and prof.calibrated:
            prof.save()
        self._switching_camera = True

        def work():
            try:
                self.tracker.switch_source(CameraSource(mode=old.mode.name, camera=index))
                ok = True
            except RuntimeError as e:
                log.error("%s", e)
                self.tracker.switch_source(old)  # it was released cleanly, so it reopens
                ok = False
            AppHelper.callAfter(done, ok)

        def done(ok):
            self._switching_camera = False
            log.info("camera: %s", self.tracker.source.source_id)
            if ok:
                self.flows.launch(False)

        threading.Thread(target=work, name="switch-camera", daemon=True).start()

    def _mic_options(self, voice) -> list:
        from ..control import config as control_config
        from ..control.voice import input_devices  # voice extras are only there when voice control is

        try:
            names = input_devices()
        except Exception:  # PortAudio may be re-initialising on the voice thread
            names = [voice.device] if voice.device else []
        follow = voice.choice is None
        auto = "Automatic: system default" + (", MacBook mic instead of Bluetooth"
                                              if control_config.AVOID_BLUETOOTH_MIC else "")
        if follow and voice.device:
            auto += f" (now {voice.device})"
        return [(auto, follow, lambda: voice.use(None))] + [
            (name, not follow and name == voice.device, lambda name=name: voice.use(name)) for name in names]

    def _voice_options(self, speaker) -> list:
        voices = speaker.menu_voices()
        if not voices:
            return [("Loading voices…", False, lambda: None)]
        return [(f"{v['name']} · {v['tagline']}" if v["tagline"] else v["name"], v["id"] == speaker.voice,
                 lambda v=v: speaker.set_voice(v["id"])) for v in voices]

    def _toggle_voice(self) -> None:
        if self.controller is None:
            self._start_voice(typed=False)
        else:
            self.controller.set_enabled(not self.controller.enabled)
