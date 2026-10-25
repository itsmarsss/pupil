"""Look at something, say what to do: voice control running inside the gaze app.

Tiers: 0 = exact phrases (router.py), 1 = Jev gate (gate.py), 2 = Claude computer-use agent (agent.py).
"This" is where the user looked while speaking, snapped to the nearest Accessibility element. Work happens on
background threads; `tick()` (called ~30 Hz on the AppKit main thread) applies their results to the HUD.
"""
from __future__ import annotations

import queue
import threading
import time

from . import actions, commands, config, macos, router, snapping
from .agent import Agent
from .gate import YES, Gate
from .hud import Hud
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


class VoiceController:
    def __init__(self, gaze, typed: bool = False):
        """gaze: control.sources.TrackerGaze (eyes) or MouseGaze."""
        self.gaze = gaze
        self.screen = Screen()
        self.hud = Hud()
        self.hud.draw_gaze = not gaze.calibratable  # the tracker draws its own ring
        self.events: queue.Queue = queue.Queue()  # (kind, payload) from worker threads, applied in tick()
        self.gate = Gate()
        if not self.gate.available:
            self._status(f"Fast tier (Jev) off: {self.gate.error}. Unmatched commands go straight to Claude.")
        self.agent = Agent(self.screen, self.gate, self._status)
        self.flash_until = 0.0
        self.pending = None  # direct action waiting for "confirm"
        self.snap = (None, [])  # (target, candidates) for the live gaze, refreshed by _snap_loop
        self.commands: queue.Queue = queue.Queue()  # commands run one at a time, off the UI thread
        self._typed = typed
        self.voice: Voice | None = None  # the mic thread (None when typing)
        self.enabled = True

    def start(self) -> None:
        threading.Thread(target=self._snap_loop, name="snap", daemon=True).start()
        threading.Thread(target=self._command_loop, name="voice-commands", daemon=True).start()
        if self._typed:
            TypedInput(self._on_utterance).start()
            self.hud.mic = "typing in terminal"
        else:
            self.voice = Voice(self._on_utterance, self._on_mic_state, lambda name: self.events.put(("mic_device", name)))
            self.voice.start()
            self.hud.mic = "loading"
        if not macos.has_accessibility():
            self.hud.status = "Grant Accessibility permission to your terminal (System Settings > Privacy & Security)."
        self.hud.show()

    def set_enabled(self, on: bool) -> None:
        """Pause or resume: off releases the mic, hides the HUD and stops looking for targets."""
        self.enabled = on
        if self.voice is not None:
            self.voice.set_enabled(on)
        if on:
            self.hud.show()
        else:
            self.agent.cancel()
            self._set_pending(None)
            self.hud.hide()

    # --- callbacks from the voice threads ---

    def _on_utterance(self, text, t0, t1):
        if not self.enabled:
            return
        print(f"[{time.strftime('%H:%M:%S')}] heard: \"{text}\"", flush=True)
        self.gaze.report("heard", text)
        self.events.put(("speech", (text, t0, t1)))

    def _on_mic_state(self, state):
        self.gaze.report("mic", state)
        self.events.put(("mic", state))

    def _status(self, text):
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        self.gaze.report("status", text)
        self.events.put(("status", text))

    def _set_pending(self, pending, text=""):
        """A direct action waiting for "confirm" (None clears it); the dashboard shows it."""
        self.pending = pending
        self.gaze.report("pending", text if pending else "")

    # --- main thread ---

    def tick(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "speech":
                self._on_speech(*payload)
            elif kind == "mic":
                self.hud.mic = payload
            elif kind == "mic_device":
                self.hud.mic_device = payload
            elif kind == "status":
                self.hud.status = payload
            elif kind == "flash":
                self.hud.flash = payload
                self.flash_until = time.time() + 0.4
        h = self.hud
        h.gaze = self.gaze.point()
        h.target, h.candidates = self.snap
        h.warning = ("" if self.gaze.connected and not self.gaze.busy
                     else "calibrating" if self.gaze.busy else "no gaze: is the camera on?")
        if h.flash and time.time() > self.flash_until:
            h.flash = None
        h.update()

    def _snap_loop(self):
        while True:
            if not self.enabled:
                self.snap = (None, [])
                time.sleep(0.5)
                continue
            point = self.gaze.point()
            self.snap = snapping.snap(*point, self.screen.w_pt, self.screen.h_pt) if point else (None, [])
            time.sleep(0.15)

    def _on_speech(self, text, t0, t1):
        cmd = router.route(text)
        if cmd is None:
            return
        self.hud.heard = cmd.text
        if cmd.kind == "cancel":  # handled immediately, even while a command is running
            self.agent.cancel()
            self._set_pending(None)
            self.hud.status = "Cancelled."
            return
        if cmd.kind in ("calibrate", "recentre", "check"):
            self._tracker_command(cmd.kind)
            return
        self.commands.put((cmd, t0, t1))

    def _tracker_command(self, kind):
        """The tracker owns calibration screens: recalibrate, re-centre (1 dot) or check accuracy (9 dots)."""
        if not self.gaze.calibratable:
            self.hud.status = "Using the mouse as gaze: nothing to calibrate."
            return
        cmd = {"calibrate": "recalibrate", "recentre": "recentre", "check": "validate"}[kind]
        self.gaze.send({"cmd": cmd})
        self._status({"recalibrate": "Calibrating: look at each dot until it shrinks away.",
                      "recentre": "Re-centring: look at the dot.",
                      "validate": "Checking accuracy: follow the dots."}[cmd])

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
                self._status("Confirmed: continuing...")
            elif self.pending:
                kind, args, gaze, target = self.pending
                self._set_pending(None)
                self._do(kind, args, gaze, target)
            else:
                self._status("Nothing to confirm.")
            return
        if self.agent.busy:
            self._status('Still working: say "stop" to cancel.')
            return
        if self.gaze.busy:
            self._status("Calibrating: finish the dots first.")
            return

        gaze = self.gaze.point_during(t0, t1)  # where they looked while speaking, not now
        target, candidates = snapping.snap(*gaze, self.screen.w_pt, self.screen.h_pt) if gaze else (None, [])
        self._report_target(target)
        confidence = self.gaze.confidence_during(t0, t1)

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
        if cmd.kind in ("click", "type") and confidence < config.GAZE_MIN_CONFIDENCE:
            # The tracker wasn't sure (head turned past what calibration covered): ask instead of acting.
            where = target.describe() if target else "there"
            self._set_pending((cmd.kind, cmd.args, gaze, target), f"{cmd.kind} {where}")
            self._status(f'Not sure where you\'re looking ({cmd.kind} {where}?). Say "confirm", or "re-centre".')
            return
        self._do(cmd.kind, cmd.args, gaze, target)

    def _report_target(self, target):
        if target is None:
            self.gaze.report("target", target=None)
        else:
            self.gaze.report("target", target={"x": target.x, "y": target.y, "w": target.w, "h": target.h,
                                                "label": target.describe()})

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
            self._status(f"Ignored: {d.summary()}")
            return
        risky = d.risky >= config.GATE_RISKY_MIN
        if d.route == "open_app" and d.app_path:
            commands.open_app(d.app_path)
            self._status(f"Opening {d.app}: {d.summary()}")
            return
        if d.route == "shortcut" and d.shortcut:
            desc, combo = commands.SHORTCUTS[d.shortcut]
            if risky:
                self._set_pending(("key", {"combo": combo}, gaze, None), desc)
                self._status(f'"{desc}" might lose work. Say "confirm" to do it.')
                return
            actions.press(combo)
            self._status(f"{desc} ({combo}): {d.summary()}")
            return
        if d.route in _ROUTE_ACTIONS:
            kind, args = _ROUTE_ACTIONS[d.route]
            if risky:
                where = d.target.describe() if d.target else "here"
                self._set_pending((kind, args, gaze, d.target), f"{kind} {where}")
                self._status(f'This might be irreversible ({kind} {where}). Say "confirm" to do it.')
                return
            self._do(kind, args, gaze, d.target)
            return
        # Needs several steps or writing, or nothing direct matched: Claude takes it.
        if not self.agent.available:
            self._status(f"Skipped: needs Claude ({d.summary()}). Add ANTHROPIC_API_KEY to .env to enable Claude.")
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
