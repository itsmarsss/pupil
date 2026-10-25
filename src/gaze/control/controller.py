"""Look at something, say what to do: voice control running inside the gaze app.

Tiers: 0 = exact phrases (router.py), 1 = Jev gate (gate.py), 2 = Claude computer-use agent (agent.py).
"This" is where the user looked while speaking, snapped to the nearest Accessibility element. The gate may also
ask a question first (not enough to act on) or split a command into a few actions run one after another.
Work happens on background threads; `tick()` (called ~30 Hz on the AppKit main thread) applies their results
to the HUD. Every command is logged as an episode (episodes.py) so routing can be measured and replayed.
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass

from . import actions, apps, commands, config, dictation, heat, macos, router, snapping, uitree
from .agent import Agent
from .episodes import EpisodeLog, element_dict
from .gate import CLARIFY_PROMPTS, MAX_CANDIDATES, NO, Gate
from .hud import Hud
from .lips import LipGate
from .screen import Screen
from .speech import Speaker
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
CHOOSE_WINDOW_S = 12.0  # how long "which one?" waits for an answer
RELATIVE_WINDOW_S = 20.0  # "no, the left one" refers to the last target this recently
NUMBERS = ["one", "two", "three"]
_ORDINALS = [("one", "1", "first"), ("two", "2", "second", "too", "to"), ("three", "3", "third")]  # as heard


@dataclass
class TargetReference:
    action: str
    target: object
    candidates: list
    args: dict
    app: str
    window: str
    t: float


def pick_option(text, options):
    """Which of `options` an answer means: a position ("the left one", "top"), a number ("two", "the first",
    "number one"; "one" alone is a pronoun in "the left one"), or a name ("the YouTube one"). None if unclear."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for side, key, sign in (("left", 0, 1), ("right", 0, -1), ("top", 1, 1), ("upper", 1, 1),
                            ("bottom", 1, -1), ("lower", 1, -1)):
        if side in words:
            return min(options, key=lambda el: sign * el.center[key])
    if "last" in words:
        return options[-1]
    number_only = words <= {"one", "number", "option", "the", "it", "s", "that"}
    for i, names in enumerate(_ORDINALS[:len(options)]):
        if words & set(names[1:]) or (names[0] in words and (names[0] != "one" or number_only)):
            return options[i]
    named = [el for el in options if el.label and any(w in el.label.lower() for w in words if len(w) > 3)]
    return named[0] if len(named) == 1 else None


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
        self.agent = Agent(
            self.screen, self.gate, self._status,
            on_provider_event=lambda state: self.gaze.report("research", research=state),
        )
        self.speaker = Speaker(on_state=lambda on: self.events.put(("speaking", on)))
        self.agent.on_reply = self.speaker.say  # Claude's answer (or what it's waiting to confirm), read aloud
        self.episodes = EpisodeLog(source="typed" if typed else "voice")
        self.flash_until = 0.0
        self.pending = None  # direct action waiting for "confirm"
        self.clarifying = None  # (request, missing, deadline, episode): the gate asked a question
        self._candidates: list = []  # elements near the gaze for the command being handled
        self.dictating = False  # dictation mode: every utterance is typed at the cursor
        self.choosing = None  # "which one?" asked: options, action, deadline
        self._cloud = None  # gaze heat for the command being handled
        self._last_target: TargetReference | None = None  # last target-bearing action, for corrections/follow-ups
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
        threading.Thread(target=dictation.warm, name="warm-dictation", daemon=True).start()
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
            self._last_target = None
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
        if state == "listening" and self.lips is None:  # no lips to watch: any speech at the mic barges in
            self.speaker.stop()

    def _status(self, text):
        print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)
        self.gaze.report("status", text)
        self.events.put(("status", text))
        if self._ep is not None:
            self._ep["status"] = text

    def _set_pending(self, pending, text=""):
        """A direct action waiting for "confirm" (None clears it); the dashboard shows it, and it's asked aloud."""
        self.pending = pending
        if pending and text:
            self.speaker.say(f"{text}? Say confirm.")
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
            elif kind == "speaking":
                self.hud.speaking = payload
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
        if self.speaker.speaking and self.lips is not None and self.lips.would_open():
            self.speaker.stop()  # barge-in: you started talking (lips, not sound: the speaker would trigger that)
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
            self.speaker.stop()
            self.choosing, self.hud.badges = None, []
            if self.dictating:
                self.dictating = False
                self.hud.status = "Dictation off."
            self.agent.cancel()
            self._set_pending(None)
            self._last_target = None
            if self.clarifying is not None:
                self.episodes.label(self.clarifying[3], "declined")
                self.clarifying = None
            self.hud.status = "Cancelled."
            ep["route"] = "cancel"
            self.episodes.finish(ep)
            return
        if cmd.kind in ("dictate_on", "dictate_off"):
            self.dictating = cmd.kind == "dictate_on"
            self.hud.status = ('Dictating: everything you say is typed. Say "stop dictation" to finish.'
                               if self.dictating else "Dictation off.")
            ep["route"] = cmd.kind
            self.episodes.finish(ep)
            return
        if cmd.kind == "relative" and (self._last_target is None
                                       or time.time() - self._last_target.t > RELATIVE_WINDOW_S):
            cmd = router.Command("unknown", {"request": text}, text)  # no recent target: "left" is just a command
        if self.choosing is not None and cmd.kind != "cancel":
            if time.time() < self.choosing["deadline"]:  # the answer to "which one?"
                cmd = router.Command("choice", {"text": text}, text)
            else:
                self.choosing, self.hud.badges = None, []
        if self.dictating and cmd.kind not in ("cancel", "confirm"):  # say it, and it's typed where the cursor is
            cmd = router.Command("dictation", {"text": text}, text)
            ep["kind"] = "dictation"
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
        self._last_target = None
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
        if cmd.kind == "dictation":
            text = dictation.format(cmd.args["text"])
            actions.type_text(text)
            ep.update(route="dictation", typed=text, acted=True)
            self._status(f"Typed: {text[:60]}")
            return
        if cmd.kind == "confirm":
            if self.agent.resume():
                ep["route"] = "agent_resume"
                self._status("Confirmed: continuing...")
            elif self.pending:
                kind, args, gaze, target = self.pending
                self._set_pending(None)
                ep["route"] = "confirm"
                if kind == "close":
                    self._close(target) or self._status(f"Couldn't find how to close {target.describe()}.")
                else:
                    self._do(kind, args, gaze, target)
            else:
                ep["route"] = "nothing_to_confirm"
                self._status("Nothing to confirm.")
            return
        if self.agent.busy:  # talking to Claude mid-task: it hears this at its next step, if it's meant for it
            if self.gate.available and cmd.kind == "unknown":
                try:
                    p, _ = self.gate.is_for_the_task(cmd.text, ep.get("agent_request") or self.agent.request)
                except Exception:
                    p = 1.0  # can't tell: better to pass it on than lose an instruction
                ep["steer_p"] = round(p, 3)
                if p < NO:
                    ep["route"] = "ignore"
                    self._status(f'Not for Claude: "{cmd.text}"')
                    return
            ep["route"] = "steer"
            self.agent.steer(cmd.text)
            self._status(f'Telling Claude: "{cmd.text}"')
            return
        if self.agent.paused and cmd.kind not in ("cancel", "confirm"):  # something else instead of "confirm"
            ep["route"] = "redirect"
            self.agent.redirect(cmd.text)
            self._status(f'Skipped that; telling Claude: "{cmd.text}"')
            return
        if self.gaze.busy:
            ep["route"] = "calibrating"
            self._status("Calibrating: finish the dots first.")
            return

        gaze = self.gaze.point_during(t0, t1)  # where they looked while speaking, not now
        cloud = self.gaze.cloud_during(t0, t1)  # ...and how sure that is: a heat map, not a point
        target, candidates = snapping.snap(*gaze, self.screen.w_pt, self.screen.h_pt) if gaze else (None, [])
        self._report_target(target)
        confidence = self.gaze.confidence_during(t0, t1)
        self._cloud, self._candidates = cloud, candidates
        ep.update(gaze=[round(v, 1) for v in gaze] if gaze else None, confidence=round(confidence, 3),
                  target=element_dict(target, gaze), cloud=cloud.summary(),
                  elements=[element_dict(c, gaze) for c in candidates[:MAX_CANDIDATES]])
        if cmd.kind == "choice":
            self._resolve_choice(cmd.args["text"], cloud)
            return
        if cmd.kind == "relative":
            self._relative(cmd.args["direction"])
            return
        if cmd.kind == "reference":
            self._reference(cmd.args["mode"], gaze, cloud, candidates)
            return

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
        if cmd.kind == "shortcut":
            self._shortcut(cmd.args["shortcut"])
            return
        # Exact phrase ("click this"): the element with the most gaze heat (x fit with the words, when Jev is
        # there to judge it); if two are too close to call, ask.
        if cmd.kind in ("click", "type") and cloud and len(candidates) > 1:
            els = candidates[:MAX_CANDIDATES]
            fit = None
            if self.gate.available:
                try:
                    fit = self.gate.element_fit(cmd.text, gaze, els)
                except Exception:
                    pass  # heat alone
            shares = heat.scores(els, cloud, fit)
            ranked = sorted(zip(shares, els), key=lambda x: -x[0])
            ep["shares"] = {el.describe(): round(s, 3) for s, el in ranked[:5]}
            self.gaze.report("heat", options=[{"x": el.x, "y": el.y, "w": el.w, "h": el.h, "label": el.label[:40],
                                               "share": round(s, 3)} for s, el in ranked[:4] if s >= 0.05])
            if heat.ambiguous(shares) and cmd.kind == "click":
                self._ask_which([(s, el) for s, el in ranked[:3] if s >= 0.1], "click", cmd.args)
                return
            target = ranked[0][1]
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
            ep, agent_result=text, agent_steps=steps, agent_s=round(s, 1),
            apify=list(self.agent.provider_metadata)))

    def _route_with_gate(self, request, gaze, target, candidates, allow_clarify=True):
        ep = self._ep
        app = macos.frontmost_app()
        ep["app"] = app
        if not self.gate.available:
            ep["decision"] = "agent"
            self._start_agent(request, gaze, target, candidates, app)
            return
        try:
            d = self.gate.decide(request, gaze, candidates, app, allow_clarify=allow_clarify, cloud=self._cloud)
        except Exception as e:
            self._status(f"Jev unavailable ({e}); asking Claude...")
            ep["decision"] = "agent"
            self._start_agent(request, gaze, target, candidates, app)
            return
        ep["gate"], ep["decision"] = d.answers, d.route
        self._candidates = candidates  # for _close: the close button that belongs to the target
        self._status(d.summary())

        if d.route == "ignore":
            self._status(f"Ignored: {d.summary()}")
            return
        if d.route == "choose":
            self._ask_which(d.options, d.action, {"clicks": 1, "button": "left"})
            return
        if d.route == "clarify":
            prompt = CLARIFY_PROMPTS.get(d.missing, "Can you say more?")
            self.clarifying = (request, d.missing, time.time() + CLARIFY_WINDOW_S, ep)
            self.gaze.report("pending", prompt)
            self._status(prompt)
            self.speaker.say(prompt)
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
            self._status(f"Opening {d.app}: {d.summary()}")
            self._acted("open_app", {"app": d.app})
            self._status(apps.open_path(d.app, d.app_path))  # adapters open their app scriptable
            return True
        if d.route == "shortcut" and d.shortcut:
            desc, combo = commands.SHORTCUTS[d.shortcut]
            if risky:
                self._set_pending(("key", {"combo": combo}, gaze, None), desc)
                self._status(f'"{desc}" might lose work. Say "confirm" to do it.')
                return True
            self._shortcut(d.shortcut, d.times)
            return True
        if d.route == "close" and d.target is not None:
            if risky:
                self._set_pending(("close", {}, gaze, d.target), f"close {d.target.describe()}")
                self._status(f'Closing {d.target.describe()} might lose work. Say "confirm" to do it.')
                return True
            return self._close(d.target)
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

    def _shortcut(self, shortcut, times=1):
        """A standard action, the way the frontmost app does it: its own menu command if it has one (shortcuts
        differ between apps: "previous tab" is Ctrl+Shift+Tab in Chrome, Cmd+Shift+[ in Finder), else the keys."""
        desc, combo = commands.SHORTCUTS[shortcut]
        phrases = commands.MENU_PHRASES.get(shortcut)
        for i in range(max(1, times)):
            via = uitree.press_menu_command(phrases) if phrases else None
            if via == "":
                self._status(f"{desc}: not available right now")
                return
            if via is None:
                actions.press(combo)
            if i + 1 < times:
                time.sleep(0.06)
        how = f"menu {via!r}" if via else combo
        self._acted("key", {"combo": combo, "shortcut": shortcut, "times": times, "via": how})
        self._status(f"{desc} ({how}{f' x{times}' if times > 1 else ''})")

    # --- which one? (heat too close to call) and "no, the left one" ---

    def _ask_which(self, options, action, args):
        """Badges on the top candidates and a spoken question; the next utterance picks one (_resolve_choice)."""
        if len(options) < 2:
            if options:
                self._act_on(action, args, options[0][1])
            return
        self.choosing = {"options": [el for _, el in options], "action": action, "args": args,
                         "deadline": time.time() + CHOOSE_WINDOW_S}
        self.hud.badges = [el for _, el in options]
        self.gaze.report("heat", options=[{"x": el.x, "y": el.y, "w": el.w, "h": el.h, "label": el.label[:40],
                                           "share": round(s, 3)} for s, el in options])
        self._ep["route"] = "choose"
        self._ep["options"] = {el.describe(): s for s, el in options}
        names = " or ".join(NUMBERS[i] for i in range(len(options)))
        self._status(f"Which one? Say {names} (or look at it and say \"this one\").")
        self.speaker.say(f"Which one? {names.capitalize()}?")

    def _resolve_choice(self, text, cloud):
        choosing, self.choosing = self.choosing, None
        self.hud.badges = []
        options = choosing["options"]
        pick = pick_option(text, options)
        if pick is None and cloud:  # "this one" / "that": wherever they're looking now, among the options
            shares = heat.scores(options, cloud)
            best = max(range(len(options)), key=shares.__getitem__)
            pick = options[best] if shares[best] >= 0.5 else None
        if pick is None:
            self._status("Didn't catch which one: nothing done.")
            return
        self._ep["route"] = "choice"
        self._act_on(choosing["action"], choosing["args"], pick)

    def _act_on(self, action, args, target):
        if action == "close":
            self._close(target) or self._status(f"Couldn't find how to close {target.describe()}.")
        else:
            kind, extra = _ROUTE_ACTIONS.get(action, ("click", {"clicks": 1, "button": "left"}))
            self._do(kind, {**extra, **(args if action == "click" else {})}, None, target)

    def _new_reference(self, action, target, args=None) -> TargetReference:
        app, window = uitree.front_context()
        return TargetReference(action, target, list(self._candidates), dict(args or {}), app, window, time.time())

    @staticmethod
    def _same_element(wanted, candidate) -> bool:
        return (candidate.role == wanted.role and candidate.label == wanted.label
                and candidate.parent == wanted.parent and candidate.parent_role == wanted.parent_role
                and candidate.subrole == wanted.subrole)

    def _resolve_reference(self, reference: TargetReference):
        if time.time() - reference.t > RELATIVE_WINDOW_S:
            return None, []
        app, window = uitree.front_context()
        if (reference.app and app != reference.app) or (reference.window and window != reference.window):
            return None, []
        fresh = snapping.candidates_near(*reference.target.center, self.screen.w_pt, self.screen.h_pt)
        matches = [candidate for candidate in fresh if self._same_element(reference.target, candidate)]
        target = min(matches, key=lambda candidate: candidate.distance(*reference.target.center), default=None)
        return target, fresh

    def _reference(self, mode, gaze, cloud, candidates):
        reference = self._last_target
        if reference is None or time.time() - reference.t > RELATIVE_WINDOW_S:
            self._status("Nothing recent to refer to.")
            return
        app, window = uitree.front_context()
        if (reference.app and app != reference.app) or (reference.window and window != reference.window):
            self._last_target = None
            self._status("That reference belongs to a different app or window.")
            return
        if mode == "same_here":
            if not cloud or not candidates:
                self._status("I can't tell where you mean by here.")
                return
            shares = heat.scores(candidates[:MAX_CANDIDATES], cloud)
            ranked = sorted(zip(shares, candidates[:MAX_CANDIDATES]), key=lambda item: -item[0])
            if heat.ambiguous(shares):
                self._ask_which([(share, target) for share, target in ranked[:3] if share >= 0.1],
                                reference.action, reference.args)
                return
            self._candidates = candidates
            self._act_on(reference.action, reference.args, ranked[0][1])
            return
        target, fresh = self._resolve_reference(reference)
        if target is None:
            self._last_target = None
            self._status("That target is no longer available.")
            return
        self._candidates = fresh
        if mode == "close_last":
            self._close(target) or self._status(f"Couldn't find how to close {target.describe()}.")
        else:
            self._act_on(reference.action, reference.args, target)

    def _relative(self, direction):
        """ "No, the left one" / "the next one": redo the last action on the target's neighbour."""
        last = self._last_target
        if last is None or time.time() - last.t > RELATIVE_WINDOW_S:
            self._status("Nothing recent to move from.")
            return
        target = last.target
        other = heat.neighbour(target, last.candidates, direction)
        if other is None:
            self._status(f"There's nothing {direction} of {target.describe()}.")
            return
        self._candidates = last.candidates
        if last.action == "close":  # the wrong one is already closed: bring it back first
            self._shortcut("reopen_tab")
            time.sleep(0.4)
        self._act_on(last.action, last.args, other)

    def _close(self, target) -> bool:
        """Close what the user means, in any app: the target if it is a close control; its own close control; the
        close control of the dialog or panel it's in; or a tab by selecting it and pressing Cmd+W (closes the current
        tab in browsers, editors, Finder, terminals). False otherwise: the agent takes it."""
        reference = self._new_reference("close", target)
        if target.is_close:  # it is the close control
            self._do("click", {"clicks": 1, "button": "left"}, None, target)
            self._last_target = reference
            self._status(f"Closed ({target.describe()})")
            return True
        own = [c for c in self._candidates if c.is_close and c.parent and c.parent == target.label]
        sibling = [c for c in self._candidates if c.is_close and target.parent and c.parent == target.parent]
        if own or sibling:  # its own close control, else the close control of the dialog or panel it's in
            self._do("click", {"clicks": 1, "button": "left"}, None, (own or sibling)[0])
            self._last_target = reference
            self._status(f"Closed {target.describe() if own else target.parent}")
            return True
        if target.is_tab:
            self._do("click", {"clicks": 1, "button": "left"}, None, target)
            time.sleep(0.15)  # let the tab become the current one
            actions.press("command+w")
            self._acted("key", {"combo": "command+w"})
            self._last_target = reference
            self._status(f"Closed {target.describe()}")
            return True
        if target.role == "AXWindow":
            button = snapping.window_close_button(*target.center)
            if button is not None:
                self._do("click", {"clicks": 1, "button": "left"}, None, button)
                self._last_target = reference
                self._status("Closed the window")
                return True
        # Not a close control, doesn't have one, not a tab: closing its whole window is too big a guess (a row in
        # a list isn't "the window"); the agent looks at the screen and decides.
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
            if sub is not None and sub.kind == "shortcut":
                step["route"] = "shortcut"
                self._shortcut(sub.args["shortcut"])
                done.append(clause)
                continue
            if sub is not None and sub.kind in ("click", "type", "scroll", "key"):
                step["route"] = sub.kind
                self._do(sub.kind, sub.args, gaze, target)
                done.append(clause)
                continue
            d = None
            if sub is not None and sub.kind == "unknown" and self.gate.available:
                try:
                    d = self.gate.decide(clause, gaze, candidates, app, allow_clarify=False, allow_compound=False)
                    self._candidates = candidates
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
        if target is not None and kind == "click":
            action = {(2, "left"): "double_click", (1, "right"): "right_click"}.get((args.get("clicks"), args.get("button")), "click")
            self._last_target = self._new_reference(action, target, args)
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
            time.sleep(0.1)  # focus lands before reading the field for context
            text = dictation.format(args["text"])
            actions.type_text(text)
            if self._ep is not None:
                self._ep["typed"] = text
            self.events.put(("flash", (x, y)))
            self._status(f"Typed into {where}")
