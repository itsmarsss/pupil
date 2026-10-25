"""Look at something, say what to do: voice control running inside the gaze app.

Tiers: 0 = exact phrases (router.py), 1 = Jev gate (gate.py), 2 = Claude computer-use agent (agent.py).
"This" is where the user looked while speaking, snapped to the nearest Accessibility element. The gate may also
ask a question first (not enough to act on) or split a command into a few actions run one after another.
Work happens on background threads; `tick()` (called ~30 Hz on the AppKit main thread) applies their results
to the HUD. Every command is logged as an episode (episodes.py) so routing can be measured and replayed.
"""
from __future__ import annotations

import queue
import threading
import time

from . import actions, commands, config, macos, router, snapping
from .agent import Agent
from .episodes import EpisodeLog, element_dict
from .gate import CLARIFY_PROMPTS, MAX_CANDIDATES, YES, Gate
from .hud import Hud
from .lips import LipGate
from .screen import Screen
from .voice import CartesiaVoice, TypedInput, Voice

# Gate routes -> the direct action they map to.
_ROUTE_ACTIONS = {
    "click": ("click", {"clicks": 1, "button": "left"}),
    "double_click": ("click", {"clicks": 2, "button": "left"}),
    "right_click": ("click", {"clicks": 1, "button": "right"}),
    "scroll_down": ("scroll", {"direction": "down", "amount": 4}),
    "scroll_up": ("scroll", {"direction": "up", "amount": 4}),
}
CLARIFY_WINDOW_S = 10.0  # how long an answer to "Which one?" is taken as the answer
MAX_CLAUSES = 3  # longer chains go to the agent whole
SETTLE_S = {"open_app": 1.5}  # wait after a compound step before deciding the next one; default below
DEFAULT_SETTLE_S = 0.5
CLIP_PAD_S = 0.3  # audio saved with each episode: the utterance plus this either side


class VoiceController:
    def __init__(self, gaze, typed: bool = False):
        """gaze: control.sources.TrackerGaze (eyes) or MouseGaze."""
        self._ep: dict | None = None  # episode the worker is handling (one at a time); first: _status reads it
        self.gaze = gaze
        self.screen = Screen()
        self.hud = Hud()
        self.hud.draw_gaze = not gaze.calibratable  # the tracker draws its own ring
        self.events: queue.Queue = queue.Queue()  # (kind, payload) from worker threads, applied in tick()
        self.gate = Gate()
        if not self.gate.available:
            self._status(f"Fast tier (Jev) off: {self.gate.error}. Unmatched commands go straight to Claude.")
        self.agent = Agent(self.screen, self.gate, self._status)
        self.episodes = EpisodeLog(source="typed" if typed else "voice")
        self.flash_until = 0.0
        self.pending = None  # direct action waiting for "confirm"
        self.clarifying = None  # (request, missing, deadline, episode): the gate asked a question
        self.snap = (None, [])  # (target, candidates) for the live gaze, refreshed by _snap_loop
        self.commands: queue.Queue = queue.Queue()  # commands run one at a time, off the UI thread
        self._typed = typed
        self.voice: Voice | None = None  # the mic thread (None when typing)
        self.enabled = True
        self.lips = LipGate(config.LIP_GATE) if gaze.calibratable and config.LIP_GATE != "off" else None
        gaze.lips = self.lips
        self._lip_state = None  # last lip-gate state sent to the dashboard

    def start(self) -> None:
        threading.Thread(target=self._snap_loop, name="snap", daemon=True).start()
        threading.Thread(target=self._command_loop, name="voice-commands", daemon=True).start()
        if self._typed:
            TypedInput(self._on_utterance).start()
            self.hud.mic = "typing in terminal"
        else:
            voice = CartesiaVoice if config.STT == "cartesia" else Voice
            self.voice = voice(self._on_utterance, self._on_mic_state, lambda name: self.events.put(("mic_device", name)),
                               on_partial=lambda text: self.events.put(("partial", text)), gate=self.lips)
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
            self.clarifying = None
            self.hud.hide()

    # --- callbacks from the voice threads ---

    def _on_utterance(self, text, t0, t1):
        if not self.enabled:
            return
        stamp = time.strftime('%H:%M:%S')
        someone_else = self.lips.someone_else(t0, t1) if self.lips else False
        if self.lips:  # the dashboard's Lips plot marks each utterance with the verdict
            self.gaze.report("lips", t0=round(self.gaze.frame_time(t0), 3), t1=round(self.gaze.frame_time(t1), 3),
                             moving=not someone_else, why="lips didn't move" if someone_else else "")
        clip = self.voice.clip(t0 - CLIP_PAD_S, t1 + CLIP_PAD_S) if self.voice and config.SAVE_AUDIO else None
        if someone_else:
            if self.lips.mode == "on":
                print(f"[{stamp}] ignored (your lips didn't move): \"{text}\"", flush=True)
                self.events.put(("partial", ""))
                self.events.put(("dropped", (text, t0, t1, clip)))
                return
            print(f"[{stamp}] [shadow] gate would have BLOCKED: \"{text}\"", flush=True)
        elif self.lips and self.lips.mode == "shadow":
            print(f"[{stamp}] [shadow] gate would have let through: \"{text}\"", flush=True)
        print(f"[{stamp}] heard: \"{text}\"", flush=True)
        self.gaze.report("heard", text)
        self.events.put(("speech", (text, t0, t1, clip)))

    def _on_mic_state(self, state):
        self.gaze.report("mic", state)
        self.events.put(("mic", state))

    def _status(self, text):
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        self.gaze.report("status", text)
        self.events.put(("status", text))
        if self._ep is not None:
            self._ep["status"] = text

    def _set_pending(self, pending, text=""):
        """A direct action waiting for "confirm" (None clears it); the dashboard shows it."""
        self.pending = pending
        if pending and self._ep is not None:
            self._ep["pending"] = True
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
            elif kind == "dropped":  # heard, but the lip gate is sure it wasn't you
                text, t0, t1, clip = payload
                ep = self.episodes.begin(text, t0, t1, "dropped")
                ep["route"] = "lip_gate"
                self.episodes.save_audio(ep, clip)
                self.episodes.finish(ep)
                self.hud.status = f'Dropped (lips didn\'t move): "{text}"'
            elif kind == "partial":  # words so far, while you're still talking
                self.hud.heard = f"{payload}…" if payload else ""
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
        if self.lips:
            is_open = self.lips.would_open()
            if self.lips.mode == "on":
                h.gate = "open" if is_open else f"waiting ({self.lips.reason})"
            state = (is_open, self.lips.reason, round(self.lips.threshold, 3), round(self.lips.open_above, 3))
            if state != self._lip_state:  # for the dashboard's Lips plot
                self._lip_state = state
                self.gaze.report("lip_gate", mode=self.lips.mode, open=is_open, reason=state[1],
                                 threshold=state[2], open_above=state[3])
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

    def _on_speech(self, text, t0, t1, clip=None):
        cmd = router.route(text)
        if cmd is None:
            return
        fixed = router.correct(cmd.text) if cmd.kind == "unknown" else None
        if fixed:  # near-miss of a known phrase: "closed time" -> "close tab"
            print(f"[{time.strftime('%H:%M:%S')}] treating \"{cmd.text}\" as \"{fixed}\"", flush=True)
            cmd = router.route(fixed, wake=False)
        ep = self.episodes.begin(cmd.text, t0, t1, cmd.kind)
        self.episodes.save_audio(ep, clip)
        if fixed:
            ep["heard_as"] = text
        self.hud.heard = cmd.text
        if cmd.kind == "cancel":  # handled immediately, even while a command is running
            self.agent.cancel()
            self._set_pending(None)
            if self.clarifying is not None:
                self.episodes.label(self.clarifying[3], "declined")
                self.clarifying = None
            self.hud.status = "Cancelled."
            ep["route"] = "cancel"
            self.episodes.finish(ep)
            return
        if cmd.kind in ("calibrate", "recentre", "check"):
            self._tracker_command(cmd.kind)
            ep["route"] = cmd.kind
            self.episodes.finish(ep)
            return
        if self.clarifying is not None and cmd.kind == "unknown":
            original, missing, deadline, asked = self.clarifying
            self.clarifying = None
            if time.time() < deadline:  # the answer to our question: re-decide with it, and never ask twice
                self.episodes.label(asked, "answered")
                request = cmd.text if missing == "cut_off" else f"{original} ({missing}: {cmd.text})"
                cmd = router.Command("unknown", {"request": request, "clarified": True}, request)
                ep["clarified"] = {"from": original, "missing": missing, "answer": ep["text"]}
                self.gaze.report("pending", "")
        self.commands.put((cmd, t0, t1, ep))

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
            cmd, t0, t1, ep = self.commands.get()
            self._ep = ep
            try:
                self._handle(cmd, t0, t1)
            except Exception as e:  # keep the worker alive; show what went wrong
                ep["error"] = str(e)
                self._status(f"Error: {e}")
            finally:
                self._ep = None
                self.episodes.finish(ep)

    def _handle(self, cmd, t0, t1):
        ep = self._ep
        if cmd.kind == "confirm":
            if self.agent.resume():
                ep["route"] = "agent_resume"
                self._status("Confirmed: continuing...")
            elif self.pending:
                kind, args, gaze, target = self.pending
                self._set_pending(None)
                ep["route"] = "confirm"
                self._do(kind, args, gaze, target)
            else:
                ep["route"] = "nothing_to_confirm"
                self._status("Nothing to confirm.")
            return
        if self.agent.busy:
            ep["route"] = "busy"
            self._status('Still working: say "stop" to cancel.')
            return
        if self.gaze.busy:
            ep["route"] = "calibrating"
            self._status("Calibrating: finish the dots first.")
            return

        gaze = self.gaze.point_during(t0, t1)  # where they looked while speaking, not now
        target, candidates = snapping.snap(*gaze, self.screen.w_pt, self.screen.h_pt) if gaze else (None, [])
        self._report_target(target)
        confidence = self.gaze.confidence_during(t0, t1)
        ep.update(gaze=[round(v, 1) for v in gaze] if gaze else None, confidence=round(confidence, 3),
                  target=element_dict(target, gaze),
                  elements=[element_dict(c, gaze) for c in candidates[:MAX_CANDIDATES]])

        if cmd.kind == "unknown":
            ep["route"] = "gate"
            request = cmd.args["request"]
            fixed = router.correct(request, [c.label for c in candidates if c.label])
            if fixed:  # near-miss of a label near the gaze: "click search repose" -> "click Search repositories"
                self._status(f'Treating "{request}" as "{fixed}"')
                ep["corrected_to"] = request = fixed
            self._route_with_gate(request, gaze, target, candidates, allow_clarify=not cmd.args.get("clarified"))
            return

        ep["route"] = cmd.kind
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

    def _start_agent(self, request, gaze, target, candidates, app):
        ep = self._ep
        ep["acted"] = True
        ep.setdefault("agent_request", request)
        self.agent.start(request, gaze, target, candidates, app, on_done=lambda text, steps, s: self.episodes.outcome(
            ep, agent_result=text, agent_steps=steps, agent_s=round(s, 1)))

    def _route_with_gate(self, request, gaze, target, candidates, allow_clarify=True):
        ep = self._ep
        app = macos.frontmost_app()
        ep["app"] = app
        if not self.gate.available:
            ep["decision"] = "agent"
            self._start_agent(request, gaze, target, candidates, app)
            return
        try:
            d = self.gate.decide(request, gaze, candidates, app, allow_clarify=allow_clarify)
        except Exception as e:
            self._status(f"Jev unavailable ({e}); asking Claude...")
            ep["decision"] = "agent"
            self._start_agent(request, gaze, target, candidates, app)
            return
        ep["gate"], ep["decision"] = d.answers, d.route
        self._status(d.summary())

        if d.route == "ignore":
            self._status(f"Ignored: {d.summary()}")
            return
        if d.route == "clarify":
            prompt = CLARIFY_PROMPTS.get(d.missing, "Can you say more?")
            self.clarifying = (request, d.missing, time.time() + CLARIFY_WINDOW_S, ep)
            self.gaze.report("pending", prompt)
            self._status(prompt)
            return
        if d.route == "compound":
            self._compound(request, gaze, target, candidates, app)
            return
        if self._apply(d, gaze):
            return
        # Needs writing, reading the screen, or nothing direct matched: Claude takes it.
        if not self.agent.available:
            self._status(f"Skipped: needs Claude ({d.summary()}). Add ANTHROPIC_API_KEY to .env to enable Claude.")
            return
        self._status(f"Asking Claude ({d.why})...")
        self._start_agent(request, gaze, d.target or target, candidates, app)

    def _apply(self, d, gaze) -> bool:
        """Carry out a direct decision (or hold it for "confirm" if risky). False if it isn't a direct one."""
        risky = d.risky >= config.GATE_RISKY_MIN
        if d.route == "open_app" and d.app_path:
            commands.open_app(d.app_path)
            self._acted("open_app", {"app": d.app})
            self._status(f"Opening {d.app}: {d.summary()}")
            return True
        if d.route == "shortcut" and d.shortcut:
            desc, combo = commands.SHORTCUTS[d.shortcut]
            if risky:
                self._set_pending(("key", {"combo": combo}, gaze, None), desc)
                self._status(f'"{desc}" might lose work. Say "confirm" to do it.')
                return True
            actions.press(combo)
            self._acted("key", {"combo": combo, "shortcut": d.shortcut})
            self._status(f"{desc} ({combo}): {d.summary()}")
            return True
        if d.route in _ROUTE_ACTIONS:
            kind, args = _ROUTE_ACTIONS[d.route]
            if risky:
                where = d.target.describe() if d.target else "here"
                self._set_pending((kind, args, gaze, d.target), f"{kind} {where}")
                self._status(f'This might be irreversible ({kind} {where}). Say "confirm" to do it.')
                return True
            self._do(kind, args, gaze, d.target)
            return True
        return False

    def _compound(self, request, gaze, target, candidates, app):
        """Several actions in a row. Each clause is decided just before it runs, against the screen the earlier
        ones left; the first one that isn't a direct action hands the rest of the request to Claude."""
        ep = self._ep
        clauses = router.split_clauses(request)
        ep["clauses"] = steps = []
        if not 2 <= len(clauses) <= MAX_CLAUSES:
            self._status(f"Asking Claude ({len(clauses)} parts)...")
            self._start_agent(request, gaze, target, candidates, app)
            return
        done = []
        for i, clause in enumerate(clauses):
            if i:  # the screen has changed: look again where the user is looking now
                time.sleep(SETTLE_S.get(steps[-1].get("route"), DEFAULT_SETTLE_S))
                gaze = self.gaze.point() or gaze
                target, candidates = snapping.snap(*gaze, self.screen.w_pt, self.screen.h_pt) if gaze else (None, [])
                app = macos.frontmost_app()
            step = {"clause": clause}
            steps.append(step)
            sub = router.route(clause)
            if sub is not None and sub.kind in ("click", "type", "scroll", "key"):
                step["route"] = sub.kind
                self._do(sub.kind, sub.args, gaze, target)
                done.append(clause)
                continue
            d = None
            if sub is not None and sub.kind == "unknown" and self.gate.available:
                try:
                    d = self.gate.decide(clause, gaze, candidates, app, allow_clarify=False, allow_compound=False)
                except Exception as e:
                    step["error"] = str(e)
            if d is not None:
                step.update(route=d.route, gate=d.answers)
                if d.route == "ignore":
                    continue
                before = self.pending
                if self._apply(d, gaze):
                    if self.pending is not before:  # risky: wait for "confirm" and drop the rest
                        rest = ", ".join(clauses[i + 1:])
                        self._status(f'Say "confirm" to {clause}' + (f" (skipping: {rest})" if rest else ""))
                        return
                    done.append(clause)
                    continue
            rest = " and then ".join(clauses[i:])
            if not self.agent.available:
                self._status(f'Did {len(done)} of {len(clauses)}; "{rest}" needs Claude (ANTHROPIC_API_KEY).')
                return
            note = f' (Already done: {"; ".join(done)}. Do the rest: {rest}.)' if done else ""
            self._status(f"Asking Claude for: {rest}")
            self._start_agent(request + note, gaze, target, candidates, app)
            return
        self._status(f"Done: {'; '.join(done)}")

    def _acted(self, kind, args, where=None):
        if self._ep is not None:
            self._ep["acted"] = True
            self._ep.setdefault("actions", []).append({"kind": kind, **args, **({"where": where} if where else {})})

    def _do(self, kind, args, gaze, target):
        if gaze is None and target is None:
            self._status("I can't tell where you're looking.")
            return
        x, y = target.center if target else gaze
        where = target.describe() if target else "there"
        self._acted(kind, args, where)
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
