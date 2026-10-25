"""Look at something, say what to do.

    gaze control                # eye tracking (starts `gaze --publish`) + voice commands
    gaze control --typed        # type commands in the terminal instead (dev)
    gaze control --gaze mouse   # the mouse cursor stands in for gaze
    gaze control --calibrate    # recalibrate the tracker on start

Tiers: 0 = exact phrases (router.py), 1 = Jev gate (gate.py), 2 = Claude computer-use agent (agent.py).
"""

import argparse
import atexit
import queue
import subprocess
import sys
import threading
import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from . import actions, commands, config, macos, router, snapping
from .agent import Agent
from .gate import YES, Gate
from .gaze_client import GazeClient, MouseGaze
from .overlay import Overlay
from .screen import Screen
from .voice import TypedInput, Voice

# Gate routes -> the direct action they map to.
_ROUTE_ACTIONS = {
    "click": ("click", {"clicks": 1, "button": "left"}),
    "double_click": ("click", {"clicks": 2, "button": "left"}),
    "right_click": ("click", {"clicks": 1, "button": "right"}),
    "scroll_down": ("scroll", {"direction": "down", "amount": 4}),
    "scroll_up": ("scroll", {"direction": "up", "amount": 4}),
}


class App:
    def __init__(self, args):
        self.args = args
        self.qt = QApplication(sys.argv)
        self.qt.setQuitOnLastWindowClosed(False)  # we only have the overlay; never quit on a window closing
        macos.run_as_accessory()

        self.screen = Screen()
        self.geometry = self.qt.primaryScreen().geometry()
        self.events = queue.Queue()  # (kind, payload) from worker threads; handled on the UI thread
        self.gaze = GazeClient() if args.gaze == "eyes" else MouseGaze()
        self.gaze.start()
        if args.gaze == "eyes" and not args.no_tracker:
            self._start_tracker()
        self.overlay = Overlay(self.geometry)
        self.overlay.draw_gaze = args.gaze == "mouse"  # the tracker draws its own gaze ring
        self.gate = Gate()
        if not self.gate.available:
            print(f"Fast tier (Jev) off: {self.gate.error}. Unmatched commands go straight to Claude.")
        self.agent = Agent(self.screen, self.gate, self._status)
        self.flash_until = 0.0
        self.pending = None  # direct action waiting for "confirm"

        self.snap = (None, [])  # (target, candidates) for the live gaze, refreshed by _snap_loop
        threading.Thread(target=self._snap_loop, daemon=True).start()
        self.commands = queue.Queue()  # commands run one at a time, off the UI thread
        threading.Thread(target=self._command_loop, daemon=True).start()

        def on_utterance(text, t0, t1):
            print(f"[{time.strftime('%H:%M:%S')}] heard: \"{text}\"", flush=True)
            self.events.put(("speech", (text, t0, t1)))

        def on_mic_state(state):
            if state in ("listening", "transcribing"):
                print(f"[{time.strftime('%H:%M:%S')}] {state}...", flush=True)
            self.events.put(("mic", state))

        if args.typed:
            TypedInput(on_utterance).start()
            self.overlay.mic = "typing in terminal"
        else:
            Voice(on_utterance, on_mic_state, lambda name: self.events.put(("mic_device", name))).start()
            self.overlay.mic = "loading"

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def run(self):
        if not macos.has_accessibility():
            self.overlay.status = "Grant Accessibility permission to your terminal (System Settings > Privacy & Security)."
        self.overlay.show()
        macos.float_overlay(self.overlay)
        return self.qt.exec()

    def _start_tracker(self):
        """The tracker runs as its own AppKit app (menu bar ◉, calibration screens) next to our Qt one."""
        from ..core.paths import GAZE_MODEL_FILE, model_path

        cmd = [sys.executable, "-m", "gaze.cli", "--publish"] + (["--recalibrate"] if self.args.calibrate else [])
        if not model_path(GAZE_MODEL_FILE).exists() and "--appearance" not in self.args.tracker_args:
            print("No models/gaze_base.mnn: tracking with per-person eye patches (--appearance patches).", flush=True)
            cmd += ["--appearance", "patches"]
        tracker = subprocess.Popen(cmd + self.args.tracker_args)
        atexit.register(tracker.terminate)

    def _status(self, text):
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        self.events.put(("status", text))

    # --- UI thread ---

    def _snap_loop(self):
        while True:
            point = self.gaze.point()
            self.snap = snapping.snap(*point, self.screen.w_pt, self.screen.h_pt) if point else (None, [])
            time.sleep(0.15)

    def _tick(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "speech":
                self._on_speech(*payload)
            elif kind == "mic":
                self.overlay.mic = payload
            elif kind == "mic_device":
                self.overlay.mic_device = payload
            elif kind == "status":
                self.overlay.status = payload
            elif kind == "flash":
                self.overlay.flash = payload
                self.flash_until = time.time() + 0.4

        o = self.overlay
        o.gaze = self.gaze.point()
        o.target, o.candidates = self.snap
        o.warning = ("" if self.gaze.connected and not self.gaze.busy
                     else "calibrating" if self.gaze.busy else "no gaze - is the tracker running?")
        if o.flash and time.time() > self.flash_until:
            o.flash = None
        o.update()

    def _on_speech(self, text, t0, t1):
        cmd = router.route(text)
        if cmd is None:
            return
        self.overlay.heard = cmd.text
        if cmd.kind == "cancel":  # handled immediately, even while a command is running
            self.agent.cancel()
            self.pending = None
            self.overlay.status = "Cancelled."
            return
        if cmd.kind == "calibrate":
            self.calibrate()
            return
        self.commands.put((cmd, t0, t1))

    # --- command worker thread ---

    def _command_loop(self):
        while True:
            cmd, t0, t1 = self.commands.get()
            try:
                self._handle(cmd, t0, t1)
            except Exception as e:  # keep the worker alive; show what went wrong
                self._status(f"Error: {e}")

    def _handle(self, cmd, t0, t1):
        if cmd.kind == "confirm":
            if self.agent.resume():
                self._status("Confirmed - continuing...")
            elif self.pending:
                kind, args, gaze, target = self.pending
                self.pending = None
                self._do(kind, args, gaze, target)
            else:
                self._status("Nothing to confirm.")
            return
        if self.agent.busy:
            self._status('Still working - say "stop" to cancel.')
            return
        if self.gaze.busy:
            self._status("Calibrating - finish the dots first.")
            return

        gaze = self.gaze.point_during(t0, t1)  # where they looked while speaking, not now
        target, candidates = snapping.snap(*gaze, self.screen.w_pt, self.screen.h_pt) if gaze else (None, [])

        if cmd.kind == "unknown":
            self._route_with_gate(cmd.args["request"], gaze, target, candidates)
            return

        # Exact phrase, but several elements are close together: let Jev pick which one "this" means.
        close = [c for c in candidates if c.distance(*gaze) <= config.SNAP_RADIUS / 2] if gaze else []
        if cmd.kind in ("click", "type") and len(close) > 1 and self.gate.available:
            try:
                picked, conf, _ = self.gate.pick_target(cmd.text, gaze, candidates)
                if picked and conf >= YES:
                    target = picked
            except Exception:
                pass  # fall back to the nearest element
        self._do(cmd.kind, cmd.args, gaze, target)

    def _route_with_gate(self, request, gaze, target, candidates):
        app = macos.frontmost_app()
        if not self.gate.available:
            self.agent.start(request, gaze, target, candidates, app)
            return
        try:
            d = self.gate.decide(request, gaze, candidates, app)
        except Exception as e:
            self._status(f"Jev unavailable ({e}); asking Claude...")
            self.agent.start(request, gaze, target, candidates, app)
            return
        self._status(d.summary())

        # gate._decide already applied the truth table; here we just carry out its decision.
        if d.route == "ignore":
            self._status(f"Ignored - {d.summary()}")
            return
        risky = d.risky >= config.GATE_RISKY_MIN
        if d.route == "open_app" and d.app_path:
            commands.open_app(d.app_path)
            self._status(f"Opening {d.app} - {d.summary()}")
            return
        if d.route == "shortcut" and d.shortcut:
            desc, combo = commands.SHORTCUTS[d.shortcut]
            if risky:
                self.pending = ("key", {"combo": combo}, gaze, None)
                self._status(f'"{desc}" might lose work. Say "confirm" to do it.')
                return
            actions.press(combo)
            self._status(f"{desc} ({combo}) - {d.summary()}")
            return
        if d.route in _ROUTE_ACTIONS:
            kind, args = _ROUTE_ACTIONS[d.route]
            if risky:
                self.pending = (kind, args, gaze, d.target)
                where = d.target.describe() if d.target else "here"
                self._status(f'This might be irreversible ({kind} {where}). Say "confirm" to do it.')
                return
            self._do(kind, args, gaze, d.target)
            return
        # Needs several steps or writing, or nothing direct matched: Claude takes it.
        if not self.agent.available:
            self._status(f"Skipped: needs Claude - {d.summary()}. Add ANTHROPIC_API_KEY to .env to enable Claude.")
            return
        self._status(f"Asking Claude ({d.why})...")
        self.agent.start(request, gaze, d.target or target, candidates, app)

    def _do(self, kind, args, gaze, target):
        if gaze is None and target is None:
            self._status("I can't tell where you're looking.")
            return
        x, y = target.center if target else gaze
        where = target.describe() if target else "there"
        if kind == "click":
            actions.click(x, y, args["clicks"], args["button"])
            self.events.put(("flash", (x, y)))
            self._status(f"Clicked {where}")
        elif kind == "scroll":
            actions.scroll(*(gaze or (x, y)), args["direction"], args["amount"])
            self._status(f"Scrolled {args['direction']}")
        elif kind == "key":
            actions.press(args["combo"])
            self._status(f"Pressed {args['combo']}")
        elif kind == "type":
            actions.click(x, y)
            actions.type_text(args["text"])
            self.events.put(("flash", (x, y)))
            self._status(f"Typed into {where}")

    # --- calibration ---

    def calibrate(self):
        """The tracker owns calibration; ask it to show its dots."""
        if not self.gaze.calibratable:
            self.overlay.status = "Using the mouse as gaze - nothing to calibrate."
            return
        self.gaze.send({"cmd": "recalibrate"})
        self.overlay.status = "Calibrating - look at each dot until it shrinks away."


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gaze control")
    parser.add_argument("--gaze", choices=["eyes", "mouse"], default="eyes",
                        help="eyes = webcam eye tracking (default); mouse = the cursor stands in for gaze")
    parser.add_argument("--calibrate", action="store_true", help="recalibrate the tracker on start (eyes only)")
    parser.add_argument("--typed", action="store_true", help="type commands in the terminal instead of speaking")
    parser.add_argument("--no-tracker", action="store_true",
                        help="don't start the tracker; use one already running with `gaze --publish`")
    parser.add_argument("--tracker-args", nargs=argparse.REMAINDER, default=[],
                        help="pass the rest to the tracker, e.g. --tracker-args --targets 25 --snap")
    return App(parser.parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
