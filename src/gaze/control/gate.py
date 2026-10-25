"""Fast tier: one Jev request of small judgments; code turns them into a decision.

Jev never decides what to do. It answers yes/no and pick-one questions about the command and the elements near
the pointer ("does the command ask to open an app?", "is the command about the 'Search for repositories'
button?"), and `_decide` maps those answers to an action with a truth table. All questions are in one request
and run in parallel; most are speculative, and code only reads the ones the table needs.
"""

import json
import os
import re
import time
from dataclasses import dataclass

from . import commands, config
from .router import DEICTIC
from .snapping import Element

try:
    from typesafe_sdk import Choice, Noul, NoulCriteria, RetryPolicy, TypeSafeClient
except ImportError:  # gate is optional; without it everything unmatched goes to the agent
    TypeSafeClient = None

YES, NO = 0.7, 0.3  # Noul thresholds: above YES reads as yes, below NO as no, between is unsettled
# Ask a question only below this: cut-off or missing-recipient commands score 0.03-0.05, terse but guessable
# ones ("next", "make this bigger") 0.23-0.25 (tests/agent_cases.jsonl). A needless question costs more than a guess.
CLARIFY_BELOW = 0.15
MAX_CANDIDATES = 25  # nearest elements, one question each

# --- Question wording (everything Jev reads lives here) ---

IS_COMMAND = Noul(
    instructions="Is `command` an instruction for the computer?",
    criteria=NoulCriteria(
        true="It asks the computer to do something, even in a word or two, like \"search\", \"copy that\" or \"next\"",
        false="Talking to someone else, thinking aloud, filler, or words that don't make sense as an instruction",
    ),
)
WANTS_ELEMENT = Noul(
    instructions="Does `command` ask to press, click, open or select a specific control or item on screen?",
    criteria=NoulCriteria(
        true="It is about one particular button, link, field, tab or item, named or pointed at (\"close this\", "
             "\"open racecard\", \"search repositories\")",
        false="A general command that doesn't depend on any particular control, such as copy, paste, undo, "
              "scrolling or switching apps",
    ),
)
WANTS_APP = Noul(instructions="Does `command` ask to open, launch or switch to an application?")
WANTS_SHORTCUT = Noul(instructions="Does `command` ask for a standard editing, navigation, tab or window command, "
                                   "such as copy, paste, undo, save, find, go back, go forward, reload, new tab, "
                                   "close tab, zoom or quit?")
WANTS_SCROLL = Noul(instructions="Does `command` ask to scroll or move up or down the page?")
NEEDS_AGENT = Noul(
    instructions="Does `command` need writing new text, reading or summarizing the screen, searching for "
                 "something, or steps that depend on what appears on screen?",
    criteria=NoulCriteria(
        true="e.g. \"summarize this page\", \"reply saying I'm running late\", \"send this to Sam\", "
             "\"find a cheap flight to Boston\", or acting on several things or with a condition: \"close these "
             "tabs but keep one\", \"delete all the empty ones\", or anything that takes more than one click or "
             "keypress, like renaming or moving something or setting a field to a value",
        false="One click or keypress on something it names or points at (\"press send\") (\"search repositories\" means press the "
              "search button), scrolling, opening an app, a keyboard command, or a few of these in a row",
    ),
)
ENOUGH_CONTEXT = Noul(
    instructions="Does `command` say enough to carry it out? The user is looking at the elements near the pointer, "
                 "so this, that, it and here are clear: they mean what the user is looking at.",
    criteria=NoulCriteria(
        true="Clear what to do and to what, e.g. \"click this\", \"open Safari\", \"close that tab\", "
             "\"reply saying I'm late\", \"scroll down\"",
        false="Something essential is missing: the sentence stops midway (\"open the\"), or it needs a person, "
              "text or place it never gives (\"send it to\", \"write a message\" with nothing to write)",
    ),
)
MISSING = Choice(
    instructions="What does `command` leave out that it needs?",
    criteria={"what": "which thing or item to act on", "who": "which person to send or share with",
              "content": "what text to write or say", "where": "where to go, or which app or place",
              "cut_off": "the sentence stops before it is finished"},
)
IS_COMPOUND = Noul(
    instructions="Does `command` ask for two or three separate actions, one after another?",
    criteria=NoulCriteria(
        true="e.g. \"open Safari and make a new tab\", \"copy this then switch to Notes\", "
             "\"select all and copy\"",
        false="One action, even if it mentions several things, e.g. \"search for salt and pepper\", "
              "\"click black and white\"; or one action with a condition or exception, e.g. \"close these "
              "but keep one\"",
    ),
)
SCROLL_DIRECTION = Choice(instructions="Which way does `command` want to scroll?",
                          criteria={"down": "down, further, next, more", "up": "up, back, toward the top"})
ELEMENT_ACTION = Choice(
    instructions="What does `command` want done to the on-screen element it is about?",
    criteria={"activate": "click it: open, press, select, go to, search with, play or check it",
              "close": "close or dismiss it: a tab, window, dialog, popup or panel (not deleting an item, file or message)",
              "double_click": "double-click it",
              "context_menu": "only if it explicitly asks to right-click it or to show its options menu"},
)
REFERS_TO_ELEMENT = ("Is `command` about this element, or is this the control that does what the command asks "
                     "(e.g. the close button of the popup or panel it's `inside`, to close or dismiss it)? Words "
                     "like this, that, it and here mean what's at or nearest the pointer, unless the command names "
                     "a different one.")
RISKY = Noul(instructions="Could doing `command` lose data or do something hard to undo, such as sending a message, "
                          "submitting a form, paying, or deleting something? Opening, closing, resizing, scrolling "
                          "or switching windows, tabs and pages is not risky.")


def _app_question(apps):
    return Choice(instructions="Which application does `command` want to open or switch to?",
                  criteria={name: None for name in apps})


def _shortcut_question():
    return Choice(instructions="Which keyboard command does `command` ask for?",
                  criteria={k: desc for k, (desc, _) in commands.SHORTCUTS.items()})


def _element_info(el, gaze):
    """What Jev sees of an element: what kind of control, its label, what it sits inside, how far from the gaze."""
    d = el.distance(*gaze) if gaze else None
    info = {"kind": el.kind or el.role.removeprefix("AX"), "label": el.label or "(no label)",
            "distance_from_pointer": "under the pointer" if d == 0 else
            (f"{round(d)} pt away" if d is not None else "unknown")}
    if el.parent:
        info["inside"] = el.parent
    return info


def _element_questions(gaze, elements):
    """One yes/no per element, each carrying the element itself (not an index into a shared list)."""
    return {f"el_{i}": Noul(instructions={"element": _element_info(el, gaze), "question": REFERS_TO_ELEMENT})
            for i, el in enumerate(elements)}


CLARIFY_PROMPTS = {"what": "Which one? Look at it and say what to do.", "who": "Who to?", "content": "Saying what?",
                   "where": "Where to?", "cut_off": "I missed the end. Say it again?"}

# --- Decision ---

@dataclass
class Decision:
    route: str  # click | close | double_click | right_click | scroll_down | scroll_up | open_app | shortcut | agent | ignore
    #             | clarify | compound
    why: str  # the judgment(s) that decided it, for the status line and logs
    ms: float
    target: Element | None = None
    app: str | None = None
    app_path: str | None = None
    shortcut: str | None = None
    risky: float = 0.0
    missing: str | None = None  # clarify: what the command leaves out (a MISSING choice)
    answers: dict | None = None  # every judgment, for the episode log and offline replay

    def summary(self):
        what = {"open_app": self.app, "shortcut": self.shortcut}.get(self.route) or \
            (self.target.describe() if self.target else "")
        return f"Jev: {self.route}{' -> ' + what if what else ''} ({self.why}), {self.ms:.0f} ms"


def _element_route(element_p, gaze, choices, why):
    best_p = element_p[0][0]
    # Several near-equal matches (e.g. two "Search" buttons): take the one nearest the pointer.
    close = [el for q, el in element_p if q >= best_p - 0.05]
    target = min(close, key=lambda el: el.distance(*gaze)) if gaze else close[0]
    route = {"activate": "click", "close": "close", "double_click": "double_click", "context_menu": "right_click"}[
        choices["element_action"].choice]
    return route, f"{why + ', ' if why else ''}element {best_p:.2f}", {"target": target}


def _decide(a, elements, gaze, apps, allow_clarify=True, allow_compound=True, command=""):
    """The truth table. `a` = Jev's answers; returns (route, why, extras).
    A clarified request never asks again, and a clause of a compound one is never split again."""
    nouls, choices = a.nouls, a.choices
    p = lambda k: nouls[k].noul  # noqa: E731
    element_p = sorted(((p(f"el_{i}"), el) for i, el in enumerate(elements)), key=lambda x: -x[0])

    if p("is_command") < NO:
        return "ignore", f"not a command {p('is_command'):.2f}", {}
    # Ask only when it's clearly a command and clearly missing something: guessing wrong is worse than one question.
    if allow_clarify and p("is_command") >= YES and p("enough_context") < CLARIFY_BELOW:
        return "clarify", f"not enough to act on {p('enough_context'):.2f}", {"missing": choices["missing"].choice}
    if allow_compound and p("is_compound") >= YES:
        return "compound", f"several actions {p('is_compound'):.2f}", {}
    if p("needs_agent") >= YES:
        return "agent", f"needs steps/writing {p('needs_agent'):.2f}", {}
    # A specific on-screen control wins over apps/shortcuts ("close this" on a window's close button), and since
    # the command is clearly about some element, the best match only needs to be more likely than not. "This" /
    # "that" point at what's under the gaze too: Cmd+W for "close this" on a popup would close the window behind.
    deictic = bool(DEICTIC & set(re.findall(r"[a-z]+", command.lower())))
    if (p("wants_element") >= YES or deictic and p("is_command") >= YES) and element_p and element_p[0][0] >= 0.5:
        return _element_route(element_p, gaze, choices, f"control {p('wants_element'):.2f}")
    # Clearly "close" something, but no one element clearly meant: close what's under the gaze (its own or its
    # dialog's close control, its tab, or its window), rather than a shortcut that may hit something else.
    if p("wants_element") >= YES and elements and choices["element_action"].choice == "close":
        return "close", f"close what's under the gaze {p('wants_element'):.2f}", {"target": elements[0]}
    if p("wants_app") >= YES:
        app = choices["app"].choice
        return "open_app", f"open app {p('wants_app'):.2f}", {"app": app, "app_path": apps.get(app)}
    if p("wants_scroll") >= YES:  # before shortcuts: "scroll down" also reads as navigation
        return f"scroll_{choices['scroll_direction'].choice}", f"scroll {p('wants_scroll'):.2f}", {}
    if p("wants_shortcut") >= YES:
        return "shortcut", f"shortcut {p('wants_shortcut'):.2f}", {"shortcut": choices["shortcut"].choice}
    if element_p and element_p[0][0] >= YES:
        return _element_route(element_p, gaze, choices, "")
    if p("is_command") < YES:
        return "ignore", f"probably not a command {p('is_command'):.2f}", {}  # doing nothing is the safe default
    return "agent", "a command, but no single direct action matched", {}


class Gate:
    def __init__(self):
        self.error = None
        self._client = None
        if TypeSafeClient is None:
            self.error = "typesafe-sdk not installed"
            return
        if not os.environ.get("TYPESAFE_API_KEY"):
            self.error = "TYPESAFE_API_KEY not set"
            return
        try:
            self._client = TypeSafeClient(
                model=config.GATE_MODEL,
                **({"base_url": config.GATE_BASE_URL} if config.GATE_BASE_URL else {}),
                retry=RetryPolicy(max_retries=1, timeout=config.GATE_TIMEOUT_S),
            )
        except Exception as e:
            self.error = f"TypeSafe client error: {e}"

    @property
    def available(self):
        return self._client is not None

    def _ask(self, kind, state, questions):
        t0 = time.perf_counter()
        result = self._client.system_one(state, questions)
        ms = (time.perf_counter() - t0) * 1000
        _log({"tier": "gate", "kind": kind, "ms": round(ms), "request_id": getattr(result, "request_id", None)})
        return result, ms

    @staticmethod
    def _state(utterance, app_name, gaze, elements):
        return {
            "command": utterance,
            "frontmost_app": app_name,
            "pointer": "The user is pointing at (looking at) a spot on screen; distances are measured from it.",
            "elements_near_pointer": [_element_info(el, gaze) for el in elements],
        }

    def decide(self, utterance, gaze, candidates, app_name, allow_clarify=True, allow_compound=True) -> Decision:
        """All judgments for a free-form command in one request; `_decide` turns them into an action."""
        elements = candidates[:MAX_CANDIDATES]
        apps = commands.available_apps()
        questions = {
            "is_command": IS_COMMAND, "enough_context": ENOUGH_CONTEXT, "missing": MISSING, "is_compound": IS_COMPOUND,
            "needs_agent": NEEDS_AGENT, "wants_element": WANTS_ELEMENT, "wants_app": WANTS_APP,
            "wants_shortcut": WANTS_SHORTCUT, "wants_scroll": WANTS_SCROLL, "risky": RISKY,
            "app": _app_question(apps), "shortcut": _shortcut_question(),
            "scroll_direction": SCROLL_DIRECTION, "element_action": ELEMENT_ACTION,
            **_element_questions(gaze, elements),
        }
        result, ms = self._ask("decide", self._state(utterance, app_name, gaze, elements), questions)
        flags = {"allow_clarify": allow_clarify, "allow_compound": allow_compound}
        route, why, extras = _decide(result, elements, gaze, apps, command=utterance, **flags)
        answers = {"route": route, "why": why, "ms": round(ms), "flags": flags,
                   "nouls": {k: round(v.noul, 4) for k, v in result.nouls.items()},
                   "choices": {k: v.choice for k, v in result.choices.items()}}
        decision = Decision(route, why, ms, risky=result.nouls["risky"].noul, answers=answers, **extras)
        _log({"tier": "gate", "kind": "decision", "utterance": utterance, "route": route, "why": why,
              "target": decision.target.describe() if decision.target else None, "app": decision.app,
              "shortcut": decision.shortcut, "risky": round(decision.risky, 3),
              "judgments": {k: round(v.noul, 3) for k, v in result.nouls.items() if not k.startswith("el_")},
              "elements": {el.describe(): round(result.nouls[f"el_{i}"].noul, 3) for i, el in enumerate(elements)}})
        return decision

    def pick_target(self, utterance, gaze, candidates):
        """Which of several nearby elements a bare command ("click this") means: (element, probability, ms)."""
        elements = candidates[:MAX_CANDIDATES]
        result, ms = self._ask("target", self._state(utterance, "", gaze, elements),
                               _element_questions(gaze, elements))
        scored = sorted(((result.nouls[f"el_{i}"].noul, el) for i, el in enumerate(elements)), key=lambda x: -x[0])
        return (scored[0][1], scored[0][0], ms) if scored else (None, 0.0, ms)

    def risk_of(self, request, planned_actions):
        """Probability that the agent's next actions do something irreversible the user didn't ask for."""
        state = {"user_request": request, "planned_actions": planned_actions}
        result, ms = self._ask("risk", state, {"risky": Noul(
            instructions="Could `planned_actions` lose data or do something hard to undo (sending, submitting, "
                         "paying, deleting) that `user_request` did not explicitly ask for?")})
        return result.nouls["risky"].noul, ms


def _log(record):
    config.LOG_DIR.mkdir(exist_ok=True)
    with open(config.LOG_DIR / "latency.jsonl", "a") as f:
        f.write(json.dumps({"t": time.time(), **record}) + "\n")
