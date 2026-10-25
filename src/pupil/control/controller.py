"""Look at something, say what to do: voice control running inside Pupil.

Tiers: 0 = stop/confirm/dictation phrases (router.py), 1 = Jev gate (gate.py), 2 = the computer-use agent (agent.py).
"This" is where the user looked while speaking, snapped to the nearest Accessibility element. The gate may also
ask a question first (not enough to act on) or split a command into a few actions run one after another.
Work happens on background threads; `tick()` (called ~30 Hz on the AppKit main thread) applies their results
to the HUD. Every command is logged as an episode (episodes.py) so routing can be measured and replayed.
"""
from __future__ import annotations

import contextlib
import queue
import re
import subprocess
import urllib.parse
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from . import actions, apps, commands, config, console, dictation, extract, heat, macos, models, router, settings, snapping, uitree
from .agent import Agent
from .episodes import EpisodeLog, element_dict
from .gate import CLARIFY_PROMPTS, MAX_CANDIDATES, NO, YES, Gate
from .hud import Hud
from .indicator import Indicator, Status
from .lips import LipGate
from .memory import Memory
from .screen import Frames, Screen
from .research import Research
from .speech import Speaker
from .voice import CartesiaVoice, TypedInput, Voice, input_devices

# Gate routes -> the direct action they map to.
_ROUTE_ACTIONS = {
    "click": ("click", {"clicks": 1, "button": "left"}),
    "double_click": ("click", {"clicks": 2, "button": "left"}),
    "right_click": ("click", {"clicks": 1, "button": "right"}),
}
CLARIFY_WINDOW_S = 10.0  # how long an answer to "Which one?" is taken as the answer
MAX_CLAUSES = 3  # longer chains go to the agent whole
SETTLE_S = {"open_app": 1.5}  # wait after a compound step before deciding the next one; default below
DEFAULT_SETTLE_S = 0.5
RECENT_CONTEXT_S = 120  # what you said this recently is context for Jev's next decision
CONVERSATION_S = 300  # and this much of the conversation (said, done, replied) goes to the agent
FLASH_S = 0.9  # how long "✓ clicked" stays on the gaze indicator
CHOOSE_WINDOW_S = 12.0  # how long "which one?" waits for an answer
RELATIVE_WINDOW_S = 20.0  # "no, the left one" refers to the last target this recently
MAYBE_WINDOW_S = 10.0  # how long "do it" runs something Jev wasn't sure was a command
NUMBERS = ["one", "two", "three"]  # how "which one?" names the badges


@dataclass
class TargetReference:
    action: str
    target: object
    candidates: list
    args: dict
    app: str
    window: str
    t: float


BLOCKED = ("ignore", "maybe", "clarify")  # Jev's routes that don't act


def _describe_action(kind, args, where=None):
    """One action in words, for people ("Clicked Pull requests", "Opened Notes")."""
    if kind == "click":
        verb = {(2, "left"): "Double-clicked", (1, "right"): "Right-clicked"}.get(
            (args.get("clicks"), args.get("button")), "Clicked")
        return f"{verb} {where}" if where and where != "there" else verb
    if kind == "scroll":
        return f"Scrolled {args.get('direction', '')}".strip()
    if kind == "type":
        text = args.get("text", "")
        return f"Typed “{text[:40]}{'…' if len(text) > 40 else ''}”"
    if kind == "key":
        shortcut = args.get("shortcut")
        return commands.SHORTCUTS[shortcut][0] if shortcut in commands.SHORTCUTS else f"Pressed {args.get('combo', '')}"
    if kind == "open_app":
        if args.get("quit"):
            return f"{'Force quit' if args.get('force') else 'Quit'} {args['quit']}"
        return f"Opened {args.get('app') or args.get('url', '')}".strip()
    return "Done"


def _last_sentence(text):
    """The last sentence of something said in several ("is now beginning. Click join." -> "Click join."), or None."""
    parts = [p for p in re.split(r"(?<=[.?!])\s+", text.strip()) if p]
    return parts[-1] if len(parts) > 1 and len(parts[-1].split()) >= 2 else None


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
            self._status(f"Fast tier (Jev) off: {self.gate.error}. Unmatched commands go straight to the agent.")
        self.memory = Memory()
        self.gate.remembered = self.memory.latest_item()
        self.agent = Agent(self.screen, self.gate, self._status, on_provider_event=self._on_research)
        self.speaker = Speaker(on_state=lambda on: self.events.put(("speaking", on)))
        self.agent.on_reply = lambda text: self._say(text)  # the agent's answer (or what it's waiting on), aloud
        # web research in the background: the notch shows it, voice control stays free (research.py)
        self.research = Research(on_done=lambda job: self.events.put(("research_done", job)),
                                 on_provider_event=lambda state: self.gaze.report("research", research=state))
        self._research_eps: dict = {}  # job id -> the episode that asked for it
        self.agent.research = self.research  # the agent's research tool: status, results, start, wait
        self.episodes = EpisodeLog(source="typed" if typed else "voice")
        self.episodes.show = True  # one block per command in the terminal (console.py)
        self.flash_until = 0.0
        self.pending = None  # direct action waiting for "confirm"
        self.pending_text = None  # what it is, in words
        self.activity: deque = deque(maxlen=4)  # recent {t, said, did}, newest last: the notch's "recent"
        self.conversation: deque = deque(maxlen=24)  # (t, who, text): user / done / assistant, for context
        self.clarifying = None  # (request, missing, deadline, episode): the gate asked a question
        self.maybe = None  # (request, deadline, gaze, target, candidates, episode): maybe a command; "do it" runs it
        self._candidates: list = []  # elements near the gaze for the command being handled
        self.dictating = False  # dictation mode: every utterance is typed at the cursor
        self._agent_route = ("standard", "normal")  # Jev's model tier and step budget for the agent
        self.choosing = None  # "which one?" asked: options, action, deadline
        self._cloud = None  # gaze heat for the command being handled
        self._frame = None  # the screen while the command being handled was said (screen.Frame)
        self.frames: Frames | None = None  # the screen while you speak (not when typing)
        self._hearing = False  # the mic is in an utterance
        self._last_target: TargetReference | None = None  # last target-bearing action, for corrections/follow-ups
        self.indicator = Indicator()  # status pill trailing the gaze
        self._busy_with: Status | None = None  # the worker is waiting on Jev / the dictation model
        self._flash: tuple[Status, float] | None = None  # brief "done" / error, until the time given
        self.snap = (None, [])  # (target, candidates) for the live gaze, refreshed by _snap_loop
        self.commands: queue.Queue = queue.Queue()  # commands run one at a time, off the UI thread
        self._typed = typed
        self.voice: Voice | None = None  # the mic thread (None when typing)
        self.enabled = True
        use_lips = config.LIP_GATE != "off" and (gaze.calibratable or getattr(gaze, "lip_gate", False))
        self.lips = LipGate(config.LIP_GATE) if use_lips else None
        gaze.lips = self.lips
        self._lip_state = None  # last lip-gate state sent to the dashboard
        self._models_seen = -1  # models.version() last sent to the dashboard
        self._menu_choice: dict = {}  # (app, shortcut) -> the app's menu command for it (Jev's pick), or None
        self._tts_started_at = 0.0
        self._tts_echo_until = 0.0  # ignore mic pickup of our own speech until then
        self._last_tts_text = ""

    def _configure_models(self):
        """What each role will call, for the dashboard before anything has been called."""
        if not self._typed:
            models.configure("stt", f"Cartesia {config.CARTESIA_MODEL}" if config.STT == "cartesia"
                             else f"Whisper {Path(str(config.WHISPER_MODEL)).name} (on device)")
        if self.gate.available:
            models.configure("jev", config.GATE_MODEL)
        if extract.available():
            models.configure("extract", config.EXTRACT_MODEL)
        if config.DICTATION_FORMAT:
            models.configure("dictation", config.DICTATION_MODEL)
        if self.agent.available:
            models.configure("agent", config.AGENT_TIERS["standard"][0])
        if self.speaker.available:
            models.configure("tts", f"Cartesia {config.TTS_MODEL}")

    def start(self) -> None:
        self._configure_models()
        threading.Thread(target=self._snap_loop, name="snap", daemon=True).start()
        threading.Thread(target=self._command_loop, name="voice-commands", daemon=True).start()
        threading.Thread(target=dictation.warm, name="warm-dictation", daemon=True).start()
        threading.Thread(target=extract.warm, name="warm-extract", daemon=True).start()
        threading.Thread(target=self._warm_menus, name="warm-menus", daemon=True).start()
        if self._typed:
            TypedInput(self._on_utterance).start()
            self.hud.mic = "typing in terminal"
        else:
            voice = CartesiaVoice if config.STT == "cartesia" else Voice
            duck = lambda: self.lips is None and self.speaker.speaking
            self.voice = voice(self._on_utterance, self._on_mic_state, lambda name: self.events.put(("mic_device", name)),
                               on_partial=lambda text: self.events.put(("partial", text)), gate=self.lips,
                               stt_duck=duck)
            self.voice.start()
            self.hud.mic = "loading"
            self.frames = Frames(self.screen, self.gaze.point,
                                 lambda: self._hearing or (self.lips is not None and self.lips.would_open()))
            self.frames.start()
        if not macos.has_accessibility():
            self.hud.status = "Grant Accessibility permission to your terminal (System Settings > Privacy & Security)."
        self.hud.show()

    def _warm_menus(self):
        """Keep the frontmost app's menu index built (standard actions go through the app's own menu commands)."""
        while True:
            try:
                uitree.warm_menu_index()
            except Exception:
                pass  # an app without menus or Accessibility access: the command path reads them, or presses keys
            time.sleep(1.0)

    def _register_settings(self):
        """Voice control's own settings, for the agent's settings tool."""
        voice = self.voice

        def onoff(get, set_):
            return (lambda: "on" if get() else "off"), (lambda v: set_(v == "on")), (lambda: ["on", "off"])

        if voice is not None:
            def mics():
                try:
                    return ["automatic", *input_devices()]
                except Exception:
                    return ["automatic"]
            settings.register("microphone", "which mic listens (automatic: the default, but not Bluetooth)",
                              lambda: voice.device or "automatic",
                              lambda v: voice.use(None if v == "automatic" else v), mics)
            settings.register("voice_isolation", "Apple noise suppression on the default mic", *onoff(
                lambda: voice.isolation, voice.set_isolation))
        speaker = self.speaker
        settings.register("agent_voice", "the voice replies are spoken in (Cartesia)",
                          lambda: speaker.voice_name or speaker.voice,
                          lambda name: speaker.set_voice(next(v["id"] for v in speaker.voices if v["name"] == name)),
                          lambda: [v["name"] for v in speaker.voices])
        settings.register("speak_replies", "read the agent's replies aloud", *onoff(
            lambda: config.SPEAK, lambda on: setattr(config, "SPEAK", on)))
        settings.register("format_dictation", "clean up dictated text (symbols, punctuation)", *onoff(
            lambda: config.DICTATION_FORMAT, lambda on: setattr(config, "DICTATION_FORMAT", on)))
        if self.lips is not None:
            settings.register("lip_gate", "listen only while your lips move (shadow: log only)",
                              lambda: self.lips.mode, lambda v: setattr(self.lips, "mode", v),
                              lambda: ["on", "shadow", "off"])

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

    def _tts_echo(self, text: str) -> bool:
        """True when the mic is probably hearing our own TTS (no lip gate to tell them apart)."""
        if self.lips is not None:
            return False
        if not self.speaker.speaking and time.time() > self._tts_echo_until:
            return False
        heard = text.lower().strip(" .!?,\"'")
        if not heard:
            return True
        spoken = (self._last_tts_text or self.speaker.text).lower()
        if not spoken:
            return False
        if heard in spoken:
            return True
        for word in spoken.replace(":", " ").replace(",", " ").split():
            w = word.strip(".,!?:\"'")
            if len(heard) <= 5 and w:
                stem = heard[: min(3, len(heard))]
                if w.startswith(heard) or heard in w or w.startswith(stem) or heard.startswith(w[: min(3, len(w))]):
                    return True
        return False

    def _on_utterance(self, text, t0, t1):
        if not self.enabled:
            return
        if self._tts_echo(text):  # our own spoken reply, heard back through the mic
            console.dropped(text, "speaker echo")
            return
        someone_else = self.lips.someone_else(t0, t1) if self.lips else False
        if self.lips:  # the dashboard's Lips plot marks each utterance with the verdict
            self.gaze.report("lips", t0=round(self.gaze.frame_time(t0), 3), t1=round(self.gaze.frame_time(t1), 3),
                             moving=not someone_else, why="lips didn't move" if someone_else else "")
        clip = self.voice.clip(t0 - CLIP_PAD_S, t1 + CLIP_PAD_S) if self.voice and config.SAVE_AUDIO else None
        if someone_else:
            if self.lips.mode == "on":
                console.dropped(text, "lips still")
                self.events.put(("partial", ""))
                self.events.put(("dropped", (text, t0, t1, clip)))
                return
            console.status(f'[shadow] lip gate would have blocked: "{text}"')
        elif self.lips and self.lips.mode == "shadow":
            console.status(f'[shadow] lip gate would have let through: "{text}"')
        console.status(f'heard: "{text}"')
        self.gaze.report("heard", text)
        self.events.put(("speech", (text, t0, t1, clip)))

    def _on_mic_state(self, state):
        self._hearing = state == "listening"
        self.gaze.report("mic", state)
        self.events.put(("mic", state))

    def _status(self, text):
        console.status(text)  # the command's block in the terminal says what mattered (console.py)
        self.gaze.report("status", text)
        self.events.put(("status", text))
        if self._ep is not None:
            self._ep["status"] = text

    def _set_pending(self, pending, text=""):
        """A direct action waiting for "confirm" (None clears it); the dashboard shows it, and it's asked aloud."""
        self.pending = pending
        self.pending_text = text if pending else None  # for the notch
        if pending and text:
            self._say(f"{text}? Say confirm.")
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
                if payload:
                    self._tts_started_at = time.time()
                    self._last_tts_text = self.speaker.text
                else:
                    self._tts_echo_until = time.time() + 1.2
                    self.indicator.said()  # fade the caption
            elif kind == "partial":  # words so far, while you're still talking
                self.hud.heard = f"{payload}…" if payload else ""
            elif kind == "mic":
                self.hud.mic = payload
                if payload == "listening":
                    self.indicator.clear()  # you're talking: what it last said is stale
            elif kind == "mic_device":
                self.hud.mic_device = payload
            elif kind == "status":
                self.hud.status = payload
            elif kind == "research_done":
                self._research_done(payload)
            elif kind == "flash":
                self.hud.flash = payload
                self.flash_until = time.time() + 0.4
        if models.version() != self._models_seen:  # which model is being called, for the Voice & agent panel
            self._models_seen = models.version()
            self.gaze.report("models", models=models.snapshot())
        h = self.hud
        h.gaze = self.gaze.point()
        self.indicator.update(self._indicator_status(), h.gaze)
        if (self.speaker.speaking and self.lips is not None and self._tts_started_at
                and self.lips.moved_since(self._tts_started_at) and self.lips.would_open()):
            self.indicator.clear()
            self.speaker.stop()  # barge-in: lips moved after TTS started (not sound, not an old open gate)
        if self.lips:
            is_open = self.lips.would_open()
            if self.lips.mode == "on":
                h.gate = "open" if is_open else f"waiting ({self.lips.reason})"
            state = (is_open, self.lips.reason, round(self.lips.threshold, 3), round(self.lips.open_above, 3))
            if state != self._lip_state:  # for the dashboard's Lips plot
                self._lip_state = state
                self._register_settings()
            self.gaze.report("lip_gate", mode=self.lips.mode, open=is_open, reason=state[1],
                             threshold=state[2], open_above=state[3])
        h.target, h.candidates = self.snap
        if not self.gaze.calibratable:
            h.warning = ""
        else:
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
        self.indicator.clear()
        cmd = router.route(text)
        if cmd is None:
            return
        ep = self.episodes.begin(cmd.text, t0, t1, cmd.kind)
        self.episodes.save_audio(ep, clip)
        self.hud.heard = cmd.text
        if cmd.kind == "cancel":  # handled immediately, even while a command is running
            self._cancel()
            ep["route"] = "cancel"
            self.episodes.finish(ep)
            return
        if cmd.kind == "forget":
            self.memory.clear()
            self.gate.remembered = None
            ep["route"] = "forget"
            self._status("Forgot.")
            self.episodes.finish(ep)
            return
        if cmd.kind in ("dictate_on", "dictate_off"):
            self.dictating = cmd.kind == "dictate_on"
            self.hud.status = ('Dictating: everything you say is typed. Say "stop dictation" to finish.'
                               if self.dictating else "Dictation off.")
            ep["route"] = cmd.kind
            self.episodes.finish(ep)
            return
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

    def _cancel(self):
        """Stop everything: speech, the agent, anything waiting for an answer. Background research only when it's
        the only thing going on (otherwise "stop" is about what's in front of you)."""
        research = getattr(self, "research", None)
        busy = (self.agent.busy or self.agent.paused or self.pending or self.speaker.speaking or self.dictating
                or self.choosing is not None or self.clarifying is not None)
        if research is not None and not busy and research.cancel_all():
            self.hud.status = "Research stopped."
            return
        self.speaker.stop()
        self.choosing, self.hud.badges = None, []
        if self.dictating:
            self.dictating = False
            self.hud.status = "Dictation off."
        self.agent.cancel()
        self._set_pending(None)
        self._last_target = self.maybe = None
        if self.clarifying is not None:
            self.episodes.label(self.clarifying[3], "declined")
            self.clarifying = None
        self.hud.status = "Cancelled."

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
                self._flash = (Status("error", "error"), time.time() + FLASH_S * 2)
            finally:
                self._ep = None
                self.episodes.finish(ep)

    def _handle(self, cmd, t0, t1):
        ep = self._ep
        self._lap("started")
        self._frame = None
        if cmd.kind != "confirm":
            self.maybe = None  # said something else: whatever it was, it's moved on
        if cmd.kind == "dictation":
            with self._working("writing", "writing…"):
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
                elif kind == "quit":
                    self._really_quit(**args)
                else:
                    self._do(kind, args, gaze, target)
            elif self.maybe is not None and time.time() < self.maybe[1]:
                request, _, gaze, target, candidates, offered = self.maybe
                self.maybe = None
                ep["route"] = "maybe_confirmed"
                self.episodes.label(offered, "confirmed")  # Jev doubted a real command: worth learning from
                self._route_with_gate(request, gaze, target, candidates, allow_clarify=False, assume_command=True)
            else:
                ep["route"] = "nothing_to_confirm"
                self._status("Nothing to confirm.")
            return
        if self.agent.busy:  # talking to Claude mid-task: it hears this at its next step, if it's meant for it
            if self.gate.available and cmd.kind == "unknown":
                try:
                    with self._working("thinking", "thinking"):
                        p, _ = self.gate.is_for_the_task(cmd.text, ep.get("agent_request") or self.agent.request)
                except Exception:
                    p = 1.0  # can't tell: better to pass it on than lose an instruction
                ep["steer_p"] = round(p, 3)
                if p < NO:
                    ep["route"] = "ignore"
                    self._status(f'Not for the agent: "{cmd.text}"')
                    return
            ep["route"] = "steer"
            self._converse("user", cmd.text)
            self.agent.steer(cmd.text)
            self._status(f'Telling the agent: "{cmd.text}"')
            return
        if self.agent.paused and cmd.kind not in ("cancel", "confirm"):  # something else instead of "confirm"
            ep["route"] = "redirect"
            self.agent.redirect(cmd.text)
            self._status(f'Skipped that; telling the agent: "{cmd.text}"')
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
        self._cloud, self._candidates, self._confidence = cloud, candidates, confidence
        self._frame = self.frames.during(t0, t1, gaze) if self.frames else None
        self._lap("looked")  # gaze, elements under it, the frame
        ep.update(gaze=[round(v, 1) for v in gaze] if gaze else None, confidence=round(confidence, 3),
                  target=element_dict(target, gaze), cloud=cloud.summary(),
                  elements=[element_dict(c, gaze) for c in candidates[:MAX_CANDIDATES]])
        if cmd.kind == "choice":
            self._resolve_choice(cmd.args["text"], cloud)
            return
        if cmd.kind == "unknown":
            ep["route"] = "gate"
            self._route_with_gate(cmd.args["request"], gaze, target, candidates,
                                  allow_clarify=not cmd.args.get("clarified"))

    def _lap(self, name):
        """ms from the transcript arriving to `name`, in the episode log: where a slow command spent its time."""
        if self._ep is not None and "t" in self._ep:
            self._ep.setdefault("laps", {})[name] = round((time.time() - self._ep["t"]) * 1000)

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
        tier, budget = getattr(self, "_agent_route", ("standard", "normal"))
        ep.update(agent_tier=tier, agent_budget=budget)
        research = getattr(self, "_agent_research", False)
        ep["agent_research"] = research
        self.agent.start(request, gaze, target, candidates, app, tier=tier, budget=budget,
                         context=self._conversation_context(request), research=research,
                         frame=getattr(self, "_frame", None),
                         on_done=lambda text, steps, s: (
                             self.episodes.outcome(ep, agent_result=text, agent_steps=steps, agent_s=round(s, 1),
                                                   apify=list(self.agent.provider_metadata)),
                             self._note_activity(text or "Agent finished", ep, agent=True)))

    def _route_with_gate(self, request, gaze, target, candidates, allow_clarify=True, assume_command=False):
        ep = self._ep
        self._agent_route, self._agent_research = ("standard", "normal"), False  # until Jev says otherwise
        if not assume_command:  # (a confirmed "maybe" is already remembered)
            self._remember_command(request)
        app = macos.frontmost_app()
        ep["app"] = app
        if not self.gate.available:
            ep["decision"] = "agent"
            self._start_agent(request, gaze, target, candidates, app)
            return
        try:
            with self._working("thinking", "thinking"):
                research = getattr(self, "research", None)
                self.gate.research_jobs = research.describe() if research is not None and research.jobs else ""
                context = {"last": self._last_action(), "running": [n for n, _ in macos.running_apps()]}
                d = self.gate.decide(request, gaze, candidates, app, allow_clarify=allow_clarify, cloud=self._cloud,
                                     assume_command=assume_command, **context)
                last = _last_sentence(request)
                if d.route in BLOCKED and last and allow_clarify:
                    # Talk before the command ("...matches at a time. Can you drag this here?") can sink the
                    # whole thing: judge the last sentence on its own, and act on it if that's clearly a command.
                    retry = self.gate.decide(last, gaze, candidates, app, allow_clarify=False, cloud=self._cloud, **context)
                    ep["last_sentence"] = {"text": last, "route": retry.route, "gate": retry.answers}
                    if retry.route not in BLOCKED:
                        d, request = retry, last
        except Exception as e:
            self._status(f"Jev unavailable ({e}); asking the agent...")
            ep["decision"] = "agent"
            self._start_agent(request, gaze, target, candidates, app)
            return
        self._lap("decided")
        ep["gate"], ep["decision"] = d.answers, d.route
        if d.answers["nouls"].get("corrects_last", 0) >= YES:  # "no, not that": the last action was wrong
            self.episodes.corrected(ep)
        self._agent_route = (d.tier, d.budget)  # which model and how long, if this goes to the agent
        self._agent_research = d.research  # and whether it needs the web research tools
        self._candidates = candidates  # for _close: the close button that belongs to the target
        self._status(d.summary())

        if d.route == "ignore":
            self._status(f"Ignored: {d.summary()}")
            return
        if d.route == "maybe":  # not sure it was meant for the computer: hold it quietly; "do it" runs it
            self.maybe = (request, time.time() + MAYBE_WINDOW_S, gaze, target, candidates, ep)
            self._status(f'Heard "{request[:60]}" (not acting; "do it" runs it)')
            return
        if d.route == "choose":
            self._ask_which(d.options, d.action, {"clicks": 1, "button": "left"})
            return
        if d.route == "clarify":
            prompt = CLARIFY_PROMPTS.get(d.missing, "Can you say more?")
            self.clarifying = (request, d.missing, time.time() + CLARIFY_WINDOW_S, ep)
            self.gaze.report("pending", prompt)
            self._status(prompt)
            self._say(prompt)
            return
        if d.route == "agent" and d.research and self.research.available():  # in the background, the notch shows it
            self._start_research(request)
            return
        if d.route == "compound":
            self._compound(request, gaze, target, candidates, app)
            return
        if self._apply(d, gaze, request, target):
            return
        # Needs writing, reading the screen, or nothing direct matched: Claude takes it.
        if not self.agent.available:
            self._status(f"Skipped: needs the agent ({d.summary()}). Add ANTHROPIC_API_KEY to .env to enable it.")
            return
        self._status(f"Asking the agent ({d.why})...")
        self._start_agent(request, gaze, d.target or target, candidates, app)

    def _apply(self, d, gaze, request="", target=None) -> bool:
        """Carry out a direct decision (or hold it for "confirm" if risky). False if it isn't a direct one, or its
        text couldn't be pulled out (the agent takes it)."""
        risky = d.risky >= config.GATE_RISKY_MIN
        args = d.args or {}
        if d.route == "cancel":
            self._cancel()
            return True
        if d.route == "scroll":
            self._do("scroll", args, gaze, d.target or target)
            return True
        if d.route == "reference":
            self._reference(args["mode"], gaze, self._cloud, self._candidates)
            return True
        if d.route == "relative":
            self._relative(args["direction"])
            return True
        if d.route == "quit_app":
            self._quit_app(args.get("app"), args.get("force", False), target)
            return True
        if d.route == "navigate":
            with self._working("thinking", "reading"):
                where = extract.web(request)
            if where is None:
                return False
            self._navigate(**where)
            return True
        if d.route == "type":
            with self._working("thinking", "reading"):
                text = extract.text_to_type(request)
            if text is None:
                return False
            self._do("type", {"text": text}, gaze, d.target or target)
            return True
        if d.route in ("click", "double_click", "right_click", "close") and d.target is not None \
                and getattr(self, "_confidence", 1.0) < config.TRACKER_MIN_CONFIDENCE:
            # The tracker wasn't sure (head turned past what calibration covered): ask instead of acting.
            kind, kargs = _ROUTE_ACTIONS.get(d.route, ("close", {}))
            self._set_pending((kind, kargs, gaze, d.target), f"{kind} {d.target.describe()}")
            self._status(f'Not sure where you\'re looking ({kind} {d.target.describe()}?). Say "confirm", or "re-centre".')
            return True
        if d.route == "open_app" and d.app_path:
            self._status(f"Opening {d.app}: {d.summary()}")
            self._acted("open_app", {"app": d.app})
            self._status(apps.open_path(d.app, d.app_path))  # adapters open their app scriptable
            return True
        if d.route == "shortcut" and d.shortcut:
            desc, combo = commands.SHORTCUTS[d.shortcut]
            if risky or d.shortcut in commands.ALWAYS_CONFIRM:
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
            if self._close(d.target):
                return True
            # Nothing under the gaze to close, but they named the action ("close tab"): the app's own Close Tab,
            # not a 20 s agent run for Cmd+W.
            a = d.answers or {}
            if a.get("choices", {}).get("shortcut") == "close_tab" and a.get("nouls", {}).get("wants_shortcut", 0) >= YES:
                self._shortcut("close_tab")
                return True
            return False
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

    def _quit_app(self, name, force, target=None):
        """Quit (or force quit) the app the user names (Jev picks it from the running ones), or with name None the
        one they're looking at: its Dock icon, else the app in front. Always asks first: quitting loses whatever it
        had open."""
        if name is None:
            name = target.label if target is not None and target.role == "AXDockItem" and target.label \
                else macos.frontmost_app()
        running = macos.running_apps()
        wanted = name.lower().strip()
        match = next(((n, pid) for n, pid in running if n.lower() == wanted), None) or next(
            ((n, pid) for n, pid in running if wanted in n.lower()), None)
        if match is None:
            self._status(f"No running app called {name!r}.")
            self._say(f"No {name} running.")
            return
        label, pid = match
        verb = "Force quit" if force else "Quit"
        self._set_pending(("quit", {"pid": pid, "force": force, "name": label}, None, None), f"{verb} {label}")
        self._status(f'{verb} {label}? Say "confirm".')

    def _really_quit(self, pid, force, name):
        from AppKit import NSRunningApplication

        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app is None:
            self._status(f"{name} isn't running any more.")
            return
        (app.forceTerminate if force else app.terminate)()
        self._acted("open_app", {"quit": name, "force": force})
        self._status(f"{'Force quit' if force else 'Quit'} {name}")

    def _navigate(self, url=None, query=None):
        """A website or a web search. In a browser: its own "Open Location" (the address bar, in any browser),
        type, Return, so it lands in the window you're in (an incognito one included). Elsewhere: the default
        browser."""
        text = url or query
        command = self._menu_for("address_bar")
        via = uitree.press_menu_command(command) if command else None
        if via:
            time.sleep(0.15)  # let the address bar take focus
            actions.type_text(text)
            actions.press("return")
            self._acted("type", {"text": text, "via": via})
            self._status(f"Going to {text}")
            return
        if url:
            target = url if "://" in url else f"https://{url}"
        else:
            target = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
        subprocess.Popen(["open", target])
        self._acted("open_app", {"url": target})
        self._status(f"Opened {target}")

    def _start_research(self, request):
        """Research `request` in the background (research.py): you carry on, the notch shows how it's going."""
        ep = self._ep
        page = macos.front_page()
        _, window = uitree.front_context()
        context = "\n".join(line for line in (
            f"Page they're on: {page[0]} ({page[1]})" if page else (f"Window they're in: {window}" if window else ""),
            self._conversation_context(request) and f"Conversation before:\n{self._conversation_context(request)}",
        ) if line)
        job = self.research.start(request, context)
        if job is None:
            self._status("Already researching two things: say \"stop\" to drop them, or wait.")
            self._say("Still on two other questions.")
            return
        ep.update(route="research", acted=True, research_job=job.id)
        self._research_eps[job.id] = ep
        self._flash = (Status("done", "✓ researching"), time.time() + FLASH_S)
        self._note_activity("Researching…", ep, agent=True)
        self._status(f"Researching in the background: {request[:60]}")

    def _research_done(self, job):
        """A background research job ended (main thread): its answer joins the conversation and the notch; it's
        spoken only if nothing else is going on."""
        ep = self._research_eps.pop(job.id, None)
        if job.phase == "done":
            self._converse("assistant", f"Research on {job.question}: {job.answer}")
            if ep is not None:
                self._note_activity(job.answer, ep, agent=True)
                self.episodes.outcome(ep, research_answer=job.answer, research_sources=len(job.cards),
                                      research_s=round((job.ended or time.time()) - job.started, 1),
                                      research_charge_usd=job.charge_usd)
            quiet = (self.agent.busy or self.speaker.speaking or self.pending or self.dictating
                     or self.hud.mic in ("listening", "transcribing"))
            if not quiet:
                self._say(job.answer)
        elif ep is not None:
            self._note_activity("Research stopped" if job.phase == "cancelled" else "Research failed", ep, agent=True)
            self.episodes.outcome(ep, research_error=job.error or job.phase)

    def _last_action(self):
        """What the last action did, for Jev to read "again" / "no, the left one" against; None if none recent."""
        last = self._last_target
        if last is None or time.time() - last.t > RELATIVE_WINDOW_S:
            return None
        return f"{last.action.replace('_', ' ')} on {last.target.describe()}, {time.time() - last.t:.0f} s ago"

    def _remember_command(self, text):
        """What was just said, as context for Jev's next decision (follow-ups, answers, corrections) and the agent's.
        Everything heard counts, even what Jev ignored: "the report" said a moment ago is what "open that" means."""
        now = time.time()
        said = [s for t, who, s in self._converse_log() if who == "user" and now - t < RECENT_CONTEXT_S]
        self.gate.recent = said[-4:]  # before this one
        replies = [s for t, who, s in self._converse_log() if who == "assistant" and now - t < RECENT_CONTEXT_S / 2]
        self.gate.last_reply = replies[-1] if replies else None
        remembered = getattr(self, "memory", None)
        self.gate.remembered = remembered.latest_item() if remembered is not None else None
        self._converse("user", text)

    def _converse_log(self):
        return list(self.__dict__.setdefault("conversation", deque(maxlen=24)))  # (tests build controllers bare)

    def _converse(self, who, text):
        if text:
            self.__dict__.setdefault("conversation", deque(maxlen=24)).append((time.time(), who, text))
            memory = getattr(self, "memory", None)
            if memory is not None and who in ("user", "assistant"):
                app = window = ""
                if who == "user":
                    app, window = uitree.front_context()
                memory.remember(who, text, app=app, window=window, page=window)

    def _conversation_context(self, request):
        """The last few minutes, oldest first, for the agent: what was said, done and answered before `request`."""
        now = time.time()
        lines = []
        for t, who, text in self._converse_log():
            if now - t > CONVERSATION_S:
                continue
            ago = "just now" if now - t < 60 else f"{int((now - t) // 60)} min ago"
            lines.append({"user": f'- user ({ago}): "{text}"', "done": f"- done: {text}",
                          "assistant": f'- you said: "{text}"'}[who])
        if lines and lines[-1].endswith(f'"{request}"'):
            lines.pop()  # that's this request
        body = "\n".join(lines[-12:])
        memory = getattr(self, "memory", None)
        recalled = memory.recall(request) if memory is not None else ""
        if not recalled:
            return body
        note = ("Remembered from earlier. \"The one we talked about\" means these facts, not the gaze ring. "
                "Match a remembered title to a control on this page:\n" + recalled)
        return f"{body}\n\n{note}" if body else note

    def _on_research(self, state):
        """Research cards reach the dashboard and, once finished, the durable memory."""
        self.gaze.report("research", research=state)
        if not isinstance(state, dict) or state.get("phase") not in ("complete", "partial"):
            return
        cards = state.get("cards") or []
        if not cards:
            return
        memory = getattr(self, "memory", None)
        if memory is None:
            return
        memory.remember("research", state.get("query") or "", cards=cards)
        self.gate.remembered = memory.latest_item()

    def _menu_for(self, shortcut):
        """The frontmost app's own menu command for a standard action, chosen by Jev from its real menus once per
        app (they don't change), or None: press the keys."""
        if not self.gate.available:
            return None
        try:
            app, titles = uitree.menu_commands()
        except Exception:
            return None
        if not titles:
            return None
        key = (app, shortcut)
        if key not in self._menu_choice:
            desc, combo = commands.SHORTCUTS[shortcut]
            try:
                with self._working("thinking", "thinking"):
                    self._menu_choice[key] = self.gate.menu_command(desc, combo, titles)
            except Exception:
                return None  # not cached: ask again next time
        return self._menu_choice[key]

    def _shortcut(self, shortcut, times=1):
        """A standard action, the way the frontmost app does it: its own menu command if it has one (shortcuts
        differ between apps: "previous tab" is Ctrl+Shift+Tab in Chrome, Cmd+Shift+[ in Finder), else the keys."""
        desc, combo = commands.SHORTCUTS[shortcut]
        command = self._menu_for(shortcut)
        for i in range(max(1, times)):
            via = uitree.press_menu_command(command) if command else None
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
        self._say(f"Which one? {names.capitalize()}?")

    def _resolve_choice(self, text, cloud):
        choosing, self.choosing = self.choosing, None
        self.hud.badges = []
        options = choosing["options"]
        pick = None
        if self.gate.available:
            try:
                with self._working("thinking", "thinking"):
                    pick = self.gate.pick(text, options)
            except Exception:
                pick = "looked_at"
        if pick == "none":
            self._status("Didn't catch which one: nothing done.")
            return
        if (pick is None or isinstance(pick, str)) and cloud:  # "this one" / "that": where they're looking now
            shares = heat.scores(options, cloud)
            best = max(range(len(options)), key=shares.__getitem__)
            pick = options[best] if shares[best] >= 0.5 else None
        if pick is None or isinstance(pick, str):
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
        """Several actions in a row (split by the extraction model). Each step is decided by Jev just before it runs,
        against the screen the earlier ones left; the first one that isn't a direct action hands the rest of the
        request to the agent."""
        ep = self._ep
        with self._working("thinking", "reading"):
            clauses = extract.steps(request) or []
        ep["clauses"] = steps = []
        if not 2 <= len(clauses) <= MAX_CLAUSES:
            self._status(f"Asking the agent ({len(clauses)} parts)...")
            self._start_agent(request, gaze, target, candidates, app)
            return
        done = []
        for i, clause in enumerate(clauses):
            if i:  # the screen has changed: look again where the user is looking now
                time.sleep(SETTLE_S.get(steps[-1].get("route"), DEFAULT_SETTLE_S))
                self._frame = None  # what was on screen while they spoke is gone too
                gaze = self.gaze.point() or gaze
                target, candidates = snapping.snap(*gaze, self.screen.w_pt, self.screen.h_pt) if gaze else (None, [])
                app = macos.frontmost_app()
            step = {"clause": clause}
            steps.append(step)
            d = None
            if self.gate.available:
                try:
                    with self._working("thinking", "thinking"):
                        d = self.gate.decide(clause, gaze, candidates, app, allow_clarify=False, allow_compound=False,
                                             assume_command=True)  # the whole request was one
                    self._candidates = candidates
                except Exception as e:
                    step["error"] = str(e)
            if d is not None:
                step.update(route=d.route, gate=d.answers)
                if d.route == "ignore":
                    continue
                before = self.pending
                if self._apply(d, gaze, clause, target):
                    if self.pending is not before:  # risky: wait for "confirm" and drop the rest
                        rest = ", ".join(clauses[i + 1:])
                        self._status(f'Say "confirm" to {clause}' + (f" (skipping: {rest})" if rest else ""))
                        return
                    done.append(clause)
                    continue
            rest = " and then ".join(clauses[i:])
            if not self.agent.available:
                self._status(f'Did {len(done)} of {len(clauses)}; "{rest}" needs the agent (ANTHROPIC_API_KEY).')
                return
            note = f' (Already done: {"; ".join(done)}. Do the rest: {rest}.)' if done else ""
            self._status(f"Asking the agent for: {rest}")
            self._start_agent(request + note, gaze, target, candidates, app)
            return
        self._status(f"Done: {'; '.join(done)}")

    def _say(self, text):
        """Speak `text` and run it as a faint caption under the gaze pill (shown for a reading time if speech is off)."""
        self._converse("assistant", text)
        self.indicator.say(text)
        if self.speaker.available:
            self.speaker.say(text)
        else:
            self.indicator._caption_end = time.time() + 1.0 + len(text) / 15

    @contextlib.contextmanager
    def _working(self, state, label):
        """Show the worker waiting on a model (Jev deciding, dictation being formatted) on the gaze indicator."""
        self._busy_with = Status(state, label)
        try:
            yield
        finally:
            self._busy_with = None

    def _indicator_status(self) -> Status | None:
        """What the gaze indicator shows, most important first; None hides it."""
        now = time.time()
        if not self.enabled:
            return None
        if self._flash and now < self._flash[1]:
            return self._flash[0]
        if self.speaker.speaking:
            return Status("speaking", "speaking")
        if self.agent.busy:
            if self.agent.phase == "acting":
                return Status("acting", f"Agent: {self.agent.doing}")
            tier = "" if self.agent.tier == "standard" else f" · {self.agent.tier}"
            return Status("agent", f"Agent · step {max(self.agent._steps, 1)}{tier}")
        if self.agent.paused or self.pending:
            return Status("waiting", 'say "confirm"')
        if self.choosing is not None:
            return Status("waiting", "which one?")
        if self.clarifying is not None and now < self.clarifying[2]:
            return Status("waiting", "answer?")
        if self._busy_with is not None:
            return self._busy_with
        if self.hud.mic == "listening":
            return Status("listening", "listening")
        if self.hud.mic == "transcribing":
            return Status("hearing", "hearing…")
        if self.dictating:
            return Status("dictating", "dictating")
        return None

    def _acted(self, kind, args, where=None):
        what = {"click": "clicked", "scroll": "scrolled", "key": "pressed", "type": "typed",
                "open_app": "opened"}.get(kind, "done")
        self._flash = (Status("done", f"✓ {what}"), time.time() + FLASH_S)
        self._note_activity(_describe_action(kind, args, where))
        if self._ep is not None:
            self._ep["acted"] = True
            self._ep.setdefault("actions", []).append({"kind": kind, **args, **({"where": where} if where else {})})

    def _note_activity(self, did, ep=None, agent=False):
        """What a command did, under what was said; several actions of one command share a line."""
        ep = ep or self._ep
        said = (ep or {}).get("text", "")
        key = (ep or {}).get("id")
        if not agent:  # (the agent's reply is logged by _say)
            self._converse("done", did)
        activity = self.__dict__.setdefault("activity", deque(maxlen=4))  # (tests build controllers bare)
        same = next((item for item in activity if key is not None and item.get("id") == key), None)
        if same is not None:  # a later action of the same command (or research finishing after other commands)
            same.update(did=did, t=time.time(), agent=agent)
        else:  # the episode rides along (by reference: later outcomes show up) for the notch's expanded row
            activity.append({"id": key, "said": said, "did": did, "t": time.time(), "agent": agent, "ep": ep})

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
            with self._working("writing", "writing…"):
                text = dictation.format(args["text"])
            actions.type_text(text)
            if self._ep is not None:
                self._ep["typed"] = text
            self.events.put(("flash", (x, y)))
            self._status(f"Typed into {where}")
