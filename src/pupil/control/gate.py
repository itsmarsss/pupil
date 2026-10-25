"""Fast tier: one Jev request of small judgments; code turns them into a decision.

Jev never decides what to do. It answers yes/no and pick-one questions about the command and the elements near
the pointer ("does the command ask to open an app?", "is the command about the 'Search for repositories'
button?"), and `_decide` maps those answers to an action with a truth table. All questions are in one request
and run in parallel; most are speculative, and code only reads the ones the table needs.
"""

import json
import os
import time
from dataclasses import dataclass

from . import commands, config, heat, models
from .snapping import Element

try:
    from typesafe_sdk import Choice, Noul, NoulCriteria, RetryPolicy, TypeSafeClient
except ImportError:  # gate is optional; without it everything unmatched goes to the agent
    TypeSafeClient = None

YES, NO = 0.7, 0.3
NAMED_ELEMENT = 0.85  # an element the command clearly names  # Noul thresholds: above YES reads as yes, below NO as no, between is unsettled
# Ask a question only below this: cut-off or missing-recipient commands score 0.03-0.05, terse but guessable
# ones ("next", "make this bigger") 0.23-0.25 (tests/agent_cases.jsonl). A needless question costs more than a guess.
CLARIFY_BELOW = 0.15
ASK_ABOUT = {"cut_off", "who", "content"}  # what's missing that's worth a question (the screen can't supply it)
MAX_CANDIDATES = 25  # nearest elements, one question each
MAX_RUNNING = 60  # open apps offered for "quit ..."

# --- Question wording (everything Jev reads lives here) ---

IS_COMMAND = Noul(
    instructions="Is `command` an instruction for the computer? `previous_commands` (if any) are what the user "
                 "asked just before: a follow-up that continues or corrects them counts.",
    criteria=NoulCriteria(
        true="It asks the computer to do something, even in a word or two (\"search\", \"copy that\", \"next\") or "
             "phrased as a question or a wish (\"can you open my notes?\", \"I want to watch this\")",
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
WANTS_APP = Noul(
    instructions="Does `command` ask to open, launch or switch to an application?",
    criteria=NoulCriteria(
        true="e.g. \"open Notes\", \"launch Spotify\", \"switch to Finder\"",
        false="Opening something inside the current app: a window, tab, mode or page (\"open incognito\", \"open a "
              "new tab\", \"open settings\"), or a website",
    ),
)
WANTS_SHORTCUT = Noul(instructions="Does `command` ask for a standard editing, navigation, tab or window command, "
                                   "or a plain key, such as copy, paste, undo, save, find, go back, go forward, "
                                   "reload, new tab, close tab, zoom, quit, or a key it names or implies (an arrow key "
                                   "to move something, \"with space\", \"press enter\", page down)?")
WANTS_SCROLL = Noul(instructions="Does `command` ask to scroll the page or view, up, down, left or right (not to "
                                "move a selected item or game piece with the arrow keys)?")
NEEDS_AGENT = Noul(
    instructions="Does `command` need writing new text, reading or summarizing the screen, searching for "
                 "something, or steps that depend on what appears on screen?",
    criteria=NoulCriteria(
        true="e.g. \"summarize this page\", \"reply saying I'm running late\", \"send this to Sam\", "
             "\"find a cheap flight to Boston\", or acting on several things or with a condition: \"close these "
             "tabs but keep one\", \"delete all the empty ones\", or anything that takes several different steps, like "
             "renaming or moving a file or setting a field to a value (pressing one key a few times is one step)",
        false="One click or keypress on something it names or points at (\"press send\") (\"search repositories\" means press the "
              "search button), scrolling, opening an app or a website, one web search, a keyboard command, or a few "
              "of these in a row",
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
CUT_OFF = Noul(
    instructions="Does `command` stop before the sentence is finished, as if the speaker was cut off?",
    criteria=NoulCriteria(
        true="e.g. \"open the\", \"move it to\", \"put this in the\", \"send it to\"",
        false="A complete sentence, however short: \"open\", \"zoom in\", \"turn it on\", \"click this\"",
    ),
)
POINTS_AT_GAZE = Noul(
    instructions="Does `command` pick out what it acts on by pointing (it, this, that, here, these, the one I'm "
                 "looking at) rather than by naming it or not needing anything on screen?",
    criteria=NoulCriteria(
        true="e.g. \"close this\", \"open that\", \"click here\", \"delete these\", \"what's this?\"",
        false="e.g. \"open Safari\", \"close the Gemini tab\", \"scroll down\", \"copy\", \"that was quick\"",
    ),
)
FOLLOWS_UP = Noul(
    instructions="Does `command` carry on from `previous_commands`: the next move or step of the same thing, an answer, "
                 "or a correction, rather than talk about something else?",
    criteria=NoulCriteria(
        true="e.g. after \"play the best move\": \"knight g2\", \"bishop here\"; after \"open my notes\": \"the "
             "work one\"; after \"scroll down\": \"more\"",
        false="A remark, a comment on what's happening, or a new topic (\"that bot is good\", \"the credits ran out\")",
    ),
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
NEEDS_RESEARCH = Noul(
    instructions="Does `command` need current information from the web that isn't on the screen: comparing "
                 "products or options, prices, reviews, the latest news, checking a claim against sources, or "
                 "gathering listings?",
    criteria=NoulCriteria(
        true="e.g. \"compare this with three alternatives\", \"is this the cheapest?\", \"what are people saying "
             "about this\", \"find current prices for these\"",
        false="Working with what's on screen or in an app (\"summarize this page\", \"reply to her\", \"open my "
              "notes\"), opening a website or one search, or asking about research already going or done "
              "(`background_research`): how it's going, what it found, or what to do with it",
    ),
)
ABOUT_RESEARCH = Noul(
    instructions="Is `command` about the research in `background_research`: how it's going, what it found, its "
                 "sources, or doing something with its results (open them, put them in a note, compare them)?",
    criteria=NoulCriteria(
        true="e.g. \"is the research done?\", \"what did it find?\", \"open the sources\", \"put that in a note\"",
        false="A new question to look into, including about what's on screen now (\"compare this with three "
              "alternatives\"), or anything else",
    ),
)
AGENT_TIER = Choice(
    instructions="If `command` goes to the agent, which kind of model should do it?",
    criteria={"fast": "quick: a question about the screen, one or two steps, or real-time play where speed matters "
                      "(a game)",
              "standard": "most multi-step tasks in one app",
              "deep": "long or careful work: writing something substantial, building or designing, several apps, or "
                      "anything where a mistake is costly"},
)
AGENT_BUDGET = Choice(
    instructions="How long should the agent keep going on `command`?",
    criteria={"quick": "a few steps and done",
              "normal": "a task that takes a while but ends",
              "ongoing": "an activity that continues until told to stop: playing a game, keep doing something, "
                         "watching for something"},
)
TIMES = Choice(instructions="How many times does `command` want it done?",
               criteria={"once": "once, or it doesn't say", "few": "a few times or a little: \"a bit\", \"two or "
                                                                      "three\", \"a couple\"",
                         "many": "many times or as far as it goes: \"all the way\", \"to the end\", \"lots\""})
TIMES_COUNT = {"once": 1, "few": 3, "many": 10}
SCROLL_DIRECTION = Choice(instructions="Which way does `command` want to scroll?",
                          criteria={"down": "down, further, next, more", "up": "up, back, toward the top",
                                    "left": "left, sideways back", "right": "right, sideways on"})
SCROLL_AMOUNT = Choice(instructions="How far does `command` want to scroll?",
                       criteria={"little": "a little, a bit, a touch", "normal": "it doesn't say",
                                 "lot": "a lot, much further, all the way"})
SCROLL_STEPS = {"little": 1, "normal": 4, "lot": 10}
WANTS_CANCEL = Noul(instructions="Does `command` ask the computer itself to stop, cancel or drop what it's doing or "
                                 "about to do (\"stop\", \"never mind\", \"forget it\")? Not closing or dismissing "
                                 "something on screen (a popup, dialog or tab), and not undoing a change already made.")
WANTS_TYPE = Noul(
    instructions="Does `command` explicitly ask for some words to be typed into a text field, and give those exact "
                 "words?",
    criteria=NoulCriteria(
        true="\"type hello world\", \"write see you at five\", \"enter sam at example dot com\"",
        false="Anything else: text still to be composed (\"reply saying I'm late\", \"write back that I'll join\"), "
              "renaming or editing something, a move in a game, a name or a single word on its own (\"castle\"), "
              "or a command",
    ),
)
WANTS_WEB = Noul(instructions="Is `command` one step: open a website by its address, or run one web search "
                              "(\"go to github.com\", \"google eye trackers\", \"look up the weather\")? Not a task "
                              "that needs browsing several pages (\"find the cheapest flight\"), and not searching "
                              "inside the current app or page.")
WANTS_QUIT = Noul(instructions="Does `command` ask to quit or force quit an application (not close a tab, window or "
                               "popup)?")
FORCE_QUIT = Noul(instructions="Does `command` ask to force quit (kill) the app rather than quit it normally?")
LOOKED_AT = "looked_at"  # quit_which: the app the user is looking at (its window or Dock icon), or the frontmost
ABOUT_LAST = Noul(
    instructions="Does `command` refer back to `last_action` (do it again, do the same somewhere else, undo or "
                 "correct it, or act on what it acted on)?",
    criteria=NoulCriteria(
        true="e.g. \"again\", \"do that again\", \"same here\", \"close the one I just clicked\", \"no, the left "
             "one\", \"the next one\", \"not that one\"",
        false="A new command that doesn't depend on what was just done",
    ),
)
LAST_MODE = Choice(
    instructions="How does `command` use `last_action`?",
    criteria={"again": "do the same action to the same thing again",
              "same_here": "do the same action to what the user is looking at now",
              "close_it": "close the thing the last action was on",
              "neighbour": "the last action hit the wrong thing: do it to the one beside it (left, right, above, "
                           "below, next, previous)"},
)
NEIGHBOUR = Choice(instructions="Which neighbour of the thing `last_action` acted on does `command` mean?",
                   criteria={"left": "to its left, or the previous one", "right": "to its right, or the next one",
                             "up": "above it", "down": "below it"})
CORRECTS_LAST = Noul(instructions="Is `command` saying that `last_action` was wrong or unwanted (\"no\", \"wrong "
                                  "one\", \"not that\", \"undo that\")?")
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
# One reading of the whole command, for when the yes/no questions above are split (each just short of YES): the
# most likely of these decides instead of every unsure command going to the agent. is_command is its null case.
INTENT = Choice(
    instructions="What does `command` want the computer to do?",
    criteria={
        "click_on_screen": "click, open, select or close one particular thing on screen: a button, link, tab, item "
                           "or popup, named or looked at",
        "key_command": "a standard command that needs no particular control: copy, paste, undo, save, find, new "
                       "tab, close tab, go back, reload, or a plain key",
        "scroll": "scroll the page or list",
        "open_app": "open, launch or switch to an application",
        "website_or_search": "open a website or run one web search",
        "type_words": "type words it gives into a text field",
        "quit_app": "quit an application",
        "several_steps": "several steps, writing something new, or reading the screen to answer or decide",
    },
)
# Each reading's own yes/no: the reading is taken only if that leans yes too (between NO and YES is fine)
INTENT_NOUL = {"click_on_screen": ("wants_element", "points_at_gaze"), "key_command": ("wants_shortcut",),
               "scroll": ("wants_scroll",), "open_app": ("wants_app",), "website_or_search": ("wants_web",),
               "type_words": ("wants_type",), "quit_app": ("wants_quit",), "several_steps": ("needs_agent",)}
LEAN = 0.5  # the intent's top reading, and its own yes/no, at least this
SURE_COMMAND = 0.9  # acting on that weaker reading needs a surer command: chatter that got past is_command scores
#                     0.7-0.9 there, commands 0.95+ (tests/agent_cases_fallthrough.jsonl, real misroutes)

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
    route: str  # choose | click | close | double_click | right_click | scroll | open_app | shortcut | agent | ignore
    #             | maybe | clarify | compound | cancel | type | navigate | quit_app | reference | relative
    why: str  # the judgment(s) that decided it, for the status line and logs
    ms: float
    target: Element | None = None
    app: str | None = None
    app_path: str | None = None
    shortcut: str | None = None
    times: int = 1  # repeatable shortcuts and keys: "all the way right"
    risky: float = 0.0
    missing: str | None = None  # clarify: what the command leaves out (a MISSING choice)
    answers: dict | None = None  # every judgment, for the episode log and offline replay
    tier: str = "standard"  # agent routes: which model tier (config.AGENT_TIERS)
    budget: str = "normal"  # agent routes: how many steps (config.AGENT_BUDGETS)
    research: bool = False  # agent routes: needs current web information (offers the Apify research tools)
    options: list | None = None  # "choose": [(share, element)] to ask between
    action: str | None = None  # "choose": what to do to the one picked
    args: dict | None = None  # scroll {direction, amount}; quit_app {app, force}; reference {mode}; relative {direction}

    def summary(self):
        what = {"open_app": self.app, "shortcut": self.shortcut}.get(self.route) or \
            (self.target.describe() if self.target else "")
        return f"Jev: {self.route}{' -> ' + what if what else ''} ({self.why}), {self.ms:.0f} ms"


CHOICES_SHOWN = 3


def _element_route(element_p, gaze, choices, why, cloud=None):
    """The element and what to do to it. With a gaze cloud, the target is the element with the most gaze heat x
    fit with the words (heat.scores); if the top two are too close to call, route "choose" with them instead."""
    best_p = element_p[0][0]
    route = {"activate": "click", "close": "close", "double_click": "double_click", "context_menu": "right_click"}[
        choices["element_action"].choice]
    why = f"{why + ', ' if why else ''}element {best_p:.2f}"
    if cloud:
        els = [el for _, el in element_p]
        shares = heat.scores(els, cloud, fit=[p for p, _ in element_p])
        ranked = sorted(zip(shares, els), key=lambda x: -x[0])
        if route == "close":  # closing a field or the close button of the same popup closes the same popup
            groups: dict = {}
            for share, el in ranked:
                groups[el.parent or el.label] = groups.get(el.parent or el.label, 0.0) + share
            shares = list(groups.values())
        if heat.ambiguous(shares):
            options = [(round(s, 3), el) for s, el in ranked[:CHOICES_SHOWN] if s >= 0.1]
            return "choose", f"{why}, too close to call", {"target": ranked[0][1], "options": options, "action": route}
        return route, f"{why}, heat {ranked[0][0]:.2f}", {"target": ranked[0][1]}
    # Several near-equal matches (e.g. two "Search" buttons): take the one nearest the pointer.
    close = [el for q, el in element_p if q >= best_p - 0.05]
    target = min(close, key=lambda el: el.distance(*gaze)) if gaze else close[0]
    return route, why, {"target": target}


def _decide(a, elements, gaze, apps, allow_clarify=True, allow_compound=True, command="", cloud=None,
            assume_command=False):
    """The truth table. `a` = Jev's answers; returns (route, why, extras).
    A clarified request never asks again, and a clause of a compound one is never split again. assume_command:
    the user already said it's for the computer ("do it" after a "maybe"), or it's a clause of a command."""
    nouls, choices = a.nouls, a.choices
    p = lambda k: nouls[k].noul if k in nouls else 0.0  # noqa: E731  (not asked, or a log from before it was)
    element_p = sorted(((p(f"el_{i}"), el) for i, el in enumerate(elements)), key=lambda x: -x[0])

    # "is the research done?", "open its sources": for us even when it doesn't sound like a command; the agent has
    # the research tool. A new question that happens to sound related ("compare this") is new research instead.
    if p("about_research") >= YES and p("about_research") >= p("needs_research"):
        return "agent", f"about the research {p('about_research'):.2f}", {}
    if p("is_command") < NO and not assume_command:
        return "ignore", f"not a command {p('is_command'):.2f}", {}
    follow_up = p("follows_up") >= YES  # "knight g2" after "play the best move"
    sure = p("is_command") >= YES or follow_up or assume_command
    # Ask only when it's clearly missing something the screen can't supply (a cut-off sentence, a person, text to
    # write): which thing and where, the gaze and the screen settle, so those go on to be guessed or worked out.
    missing = "cut_off" if p("cut_off") >= YES else choices["missing"].choice
    if allow_clarify and sure and p("enough_context") < CLARIFY_BELOW and missing in ASK_ABOUT:
        return "clarify", f"not enough to act on {p('enough_context'):.2f}", {"missing": missing}
    if allow_compound and p("is_compound") >= YES:
        return "compound", f"several actions {p('is_compound'):.2f}", {}
    if p("about_last") >= YES:  # "again", "same here", "no, the left one"
        mode = choices["last_mode"].choice
        if mode == "neighbour":
            return "relative", f"next to the last one {p('about_last'):.2f}", {"args": {"direction": choices["neighbour"].choice}}
        return "reference", f"the last action {p('about_last'):.2f}", {"args": {"mode": {"close_it": "close_last"}.get(mode, mode)}}
    if p("needs_agent") >= YES:
        return "agent", f"needs steps/writing {p('needs_agent'):.2f}", {}
    if p("needs_research") >= YES and config.APIFY_TOKEN:  # weighing up sources, not one search: the agent researches
        return "agent", f"needs web research {p('needs_research'):.2f}", {}
    # A specific on-screen control wins over apps/shortcuts ("close this" on a window's close button), and since
    # the command is clearly about some element, the best match only needs to be more likely than not. "This" /
    # "that" point at what's under the gaze too: Cmd+W for "close this" on a popup would close the window behind.
    pointing = p("points_at_gaze") >= YES
    if (p("wants_element") >= YES or pointing and sure) and element_p and element_p[0][0] >= 0.5:
        return _element_route(element_p, gaze, choices, f"control {p('wants_element'):.2f}", cloud)
    # Clearly "close" something, but no one element clearly meant: close what's under the gaze (its own or its
    # dialog's close control, its tab, or its window), rather than a shortcut that may hit something else.
    if p("wants_element") >= YES and elements and choices["element_action"].choice == "close":
        return "close", f"close what's under the gaze {p('wants_element'):.2f}", {"target": elements[0]}
    # A control on screen that the command clearly names beats a generic app/shortcut reading ("switch to github"
    # with a GitHub tab in view is that tab, not "next tab").
    if element_p and element_p[0][0] >= NAMED_ELEMENT and p("wants_element") >= 0.5:
        return _element_route(element_p, gaze, choices, f"names it {element_p[0][0]:.2f}", cloud)
    # Commands that need no particular control, after the on-screen ones ("switch to github" with a GitHub tab in
    # view is that tab, "don't save" in a dialog is its button)
    if sure and p("wants_cancel") >= YES:
        return "cancel", f"stop {p('wants_cancel'):.2f}", {}
    if p("wants_quit") >= YES:
        which = choices["quit_which"].choice
        return "quit_app", f"quit {p('wants_quit'):.2f}", {"args": {"app": None if which == LOOKED_AT else which,
                                                                  "force": p("force_quit") >= YES}}
    if p("wants_web") >= YES:
        return "navigate", f"web {p('wants_web'):.2f}", {}
    if sure and p("wants_type") >= YES:
        return "type", f"type {p('wants_type'):.2f}", {}
    if p("wants_app") >= YES:
        app = choices["app"].choice
        return "open_app", f"open app {p('wants_app'):.2f}", {"app": app, "app_path": apps.get(app)}
    if p("wants_scroll") >= YES:  # before shortcuts: "scroll down" also reads as navigation
        return "scroll", f"scroll {p('wants_scroll'):.2f}", {"args": {
            "direction": choices["scroll_direction"].choice,
            "amount": SCROLL_STEPS.get(choices["scroll_amount"].choice, 4) if "scroll_amount" in choices else 4}}
    if p("wants_shortcut") >= YES:
        shortcut = choices["shortcut"].choice
        times = TIMES_COUNT.get(choices["times"].choice, 1) if shortcut in commands.REPEATABLE and "times" in choices else 1
        return "shortcut", f"shortcut {p('wants_shortcut'):.2f}", {"shortcut": shortcut, "times": times}
    if element_p and element_p[0][0] >= YES:
        return _element_route(element_p, gaze, choices, "", cloud)
    # Nothing above was sure: the one reading of the whole command decides ("look up the sofa again" with web
    # just short of YES is still a web search), if it's clearly the most likely and its own yes/no leans yes.
    # Context the agent can use: "do it" (assume_command), the answer to our question (allow_clarify off), or a
    # follow-up ("knight g2" after "play the best move").
    has_context = assume_command or not allow_clarify or follow_up
    clearly = p("is_command") >= SURE_COMMAND or has_context
    if "intent" in choices and clearly:
        intent = choices["intent"]
        top = intent.choice
        top_p = (getattr(intent, "probabilities", None) or {}).get(top, 1.0)
        if top_p >= LEAN:
            if top != "several_steps" and max(p(k) for k in INTENT_NOUL[top]) >= LEAN:
                routed = _intent_route(top, p, choices, elements, gaze, apps, cloud)
                if routed is not None:
                    return routed[0], f"reads as {top} {top_p:.2f}, {routed[1]}", routed[2]
            # a clear command with no direct way to do it ("clear the screen": no such shortcut)
            return "agent", f"reads as {top} {top_p:.2f}, needs the agent", {}
    # Unsure it was meant for the computer and nothing direct fits: offer it ("do it" runs it) rather than drop it.
    if not sure:
        return "maybe", f"maybe a command {p('is_command'):.2f}", {}
    # Probably a command, but no clear reading of it: hold it too, rather than a slow agent run that says
    # "unclear" or guesses (most of these are talk the lip gate let through).
    if not clearly:
        return "maybe", f"no clear reading of it (command {p('is_command'):.2f})", {}
    return "agent", "a command, but no single direct action matched", {}


def _intent_route(intent, p, choices, elements, gaze, apps, cloud):
    """The route for the intent's reading, or None if it can't be carried out directly (the agent takes it)."""
    element_p = sorted(((p(f"el_{i}"), el) for i, el in enumerate(elements)), key=lambda x: -x[0])
    if intent == "click_on_screen":
        if element_p and element_p[0][0] >= LEAN:
            return _element_route(element_p, gaze, choices, "", cloud)
        if elements and p("points_at_gaze") >= LEAN:  # "click", "open it": what's under the gaze
            action = choices["element_action"].choice
            route = {"activate": "click", "close": "close", "double_click": "double_click",
                     "context_menu": "right_click"}[action]
            return route, "what's under the gaze", {"target": elements[0]}
        return None
    if intent == "key_command":
        shortcut = choices["shortcut"].choice
        times = TIMES_COUNT.get(choices["times"].choice, 1) if shortcut in commands.REPEATABLE and "times" in choices else 1
        return "shortcut", shortcut, {"shortcut": shortcut, "times": times}
    if intent == "scroll":
        return "scroll", choices["scroll_direction"].choice, {"args": {
            "direction": choices["scroll_direction"].choice,
            "amount": SCROLL_STEPS.get(choices["scroll_amount"].choice, 4) if "scroll_amount" in choices else 4}}
    if intent == "open_app":
        app = choices["app"].choice
        return "open_app", app, {"app": app, "app_path": apps.get(app)}
    if intent == "website_or_search":
        return "navigate", "web", {}
    if intent == "type_words":
        return "type", "type", {}
    if intent == "quit_app":
        which = choices["quit_which"].choice
        return "quit_app", "quit", {"args": {"app": None if which == LOOKED_AT else which,
                                             "force": p("force_quit") >= YES}}
    return None


class Gate:
    def __init__(self):
        self.recent: list[str] = []  # what the user said just before (the controller keeps it): for follow-ups
        self.last_reply: str | None = None  # what the assistant last said aloud, if just now (a question it asked)
        self.research_jobs: str = ""  # background research going on or done (research.Research.describe)
        self.remembered: str | None = None  # one line from durable memory: the item a later "select that" means
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
        with models.calling("jev", config.GATE_MODEL, kind):
            result = self._client.system_one(state, questions)
        ms = (time.perf_counter() - t0) * 1000
        _log({"tier": "gate", "kind": kind, "ms": round(ms), "request_id": getattr(result, "request_id", None)})
        return result, ms

    def _state(self, utterance, app_name, gaze, elements, last=None):
        return {
            "command": utterance,
            **({"previous_commands": self.recent[-4:]} if self.recent else {}),
            **({"assistant_just_said": self.last_reply} if self.last_reply else {}),
            **({"background_research": self.research_jobs} if getattr(self, "research_jobs", "") else {}),
            **({"remembered_item": self.remembered} if getattr(self, "remembered", None) else {}),
            **({"last_action": last} if last else {}),
            "frontmost_app": app_name,
            "pointer": "The user is pointing at (looking at) a spot on screen; distances are measured from it.",
            "elements_near_pointer": [_element_info(el, gaze) for el in elements],
        }

    def decide(self, utterance, gaze, candidates, app_name, allow_clarify=True, allow_compound=True,
               cloud=None, assume_command=False, last=None, running=()) -> Decision:
        """All judgments for a free-form command in one request; `_decide` turns them into an action.
        last: what the previous action did, if recent ("clicked the Run button 4 s ago"); running: open apps."""
        elements = candidates[:MAX_CANDIDATES]
        apps = commands.available_apps()
        quit_options = {LOOKED_AT: "the app the user is looking at (its window or Dock icon), or the one in front",
                        **{name: None for name in list(running)[:MAX_RUNNING]}}
        questions = {
            "is_command": IS_COMMAND, "enough_context": ENOUGH_CONTEXT, "cut_off": CUT_OFF,
            "points_at_gaze": POINTS_AT_GAZE,
            **({"follows_up": FOLLOWS_UP} if self.recent else {}), "missing": MISSING, "is_compound": IS_COMPOUND,
            "needs_agent": NEEDS_AGENT, "wants_element": WANTS_ELEMENT, "wants_app": WANTS_APP,
            "wants_shortcut": WANTS_SHORTCUT, "wants_scroll": WANTS_SCROLL, "risky": RISKY,
            "app": _app_question(apps), "shortcut": _shortcut_question(), "times": TIMES,
            "agent_tier": AGENT_TIER, "agent_budget": AGENT_BUDGET, "needs_research": NEEDS_RESEARCH,
            **({"about_research": ABOUT_RESEARCH} if self.research_jobs else {}),
            "scroll_direction": SCROLL_DIRECTION, "scroll_amount": SCROLL_AMOUNT, "element_action": ELEMENT_ACTION,
            "wants_cancel": WANTS_CANCEL, "wants_type": WANTS_TYPE, "wants_web": WANTS_WEB,
            "wants_quit": WANTS_QUIT, "force_quit": FORCE_QUIT, "intent": INTENT,
            "quit_which": Choice(instructions="Which application does `command` want to quit?", criteria=quit_options),
            **({"about_last": ABOUT_LAST, "last_mode": LAST_MODE, "neighbour": NEIGHBOUR,
                "corrects_last": CORRECTS_LAST} if last else {}),
            **_element_questions(gaze, elements),
        }
        result, ms = self._ask("decide", self._state(utterance, app_name, gaze, elements, last), questions)
        flags = {"allow_clarify": allow_clarify, "allow_compound": allow_compound, "assume_command": assume_command}
        route, why, extras = _decide(result, elements, gaze, apps, command=utterance, cloud=cloud, **flags)
        answers = {"route": route, "why": why, "ms": round(ms), "flags": flags,
                   "nouls": {k: round(v.noul, 4) for k, v in result.nouls.items()},
                   "choices": {k: v.choice for k, v in result.choices.items()},
                   "intent": {k: round(v, 3) for k, v in (getattr(result.choices["intent"], "probabilities", None)
                                                         or {}).items()}}
        decision = Decision(route, why, ms, risky=result.nouls["risky"].noul, answers=answers,
                            tier=result.choices["agent_tier"].choice, budget=result.choices["agent_budget"].choice,
                            research=result.nouls["needs_research"].noul >= YES
                            and not why.startswith("about the research"),  # asking about it isn't new research
                            **extras)
        _log({"tier": "gate", "kind": "decision", "utterance": utterance, "route": route, "why": why,
              "target": decision.target.describe() if decision.target else None, "app": decision.app,
              "shortcut": decision.shortcut, "risky": round(decision.risky, 3),
              "judgments": {k: round(v.noul, 3) for k, v in result.nouls.items() if not k.startswith("el_")},
              "elements": {el.describe(): round(result.nouls[f"el_{i}"].noul, 3) for i, el in enumerate(elements)}})
        return decision

    def is_for_the_task(self, utterance, task):
        """While the agent works on `task`, is `utterance` an instruction for it (change, add to, narrow or stop
        the task), rather than the user talking to someone or thinking aloud? (probability, ms)"""
        result, ms = self._ask("steer", {"assistant_is_working_on": task, "user_just_said": utterance}, {
            "steer": Noul(instructions="Is `user_just_said` meant for the assistant, about the task it's working on "
                                       "(a correction, an addition, a preference, a follow-up, or stop)?",
                          criteria=NoulCriteria(
                              true="e.g. \"actually make it blue\", \"view it from above\", \"delete this wheel\", "
                                   "\"no, the other one\", \"also add a spoiler\"",
                              false="talking to someone else, thinking aloud, filler, or words unrelated to the task"))})
        return result.nouls["steer"].noul, ms

    def element_fit(self, utterance, gaze, candidates):
        """How well each nearby element fits a bare command ("click this", "click the blue one"): one probability
        per element of candidates[:MAX_CANDIDATES], to multiply with gaze heat (heat.scores)."""
        elements = candidates[:MAX_CANDIDATES]
        result, ms = self._ask("target", self._state(utterance, "", gaze, elements),
                               _element_questions(gaze, elements))
        return [result.nouls[f"el_{i}"].noul for i in range(len(elements))]

    def pick(self, answer, options, question="Which one?"):
        """Which of `options` (elements, as shown with numbered badges) the answer to "which one?" means, or None
        ("this one" while looking: the caller uses the gaze). Positions are given relative to each other."""
        def place(el):
            cx, cy = el.center
            xs, ys = sorted(o.center[0] for o in options), sorted(o.center[1] for o in options)
            h = "leftmost" if cx == xs[0] else "rightmost" if cx == xs[-1] else "in the middle"
            v = "top" if cy == ys[0] and ys[0] != ys[-1] else "bottom" if cy == ys[-1] and ys[0] != ys[-1] else ""
            return f"{h}{', ' + v if v else ''}"

        criteria = {str(i + 1): f"option {i + 1} ({['first', 'second', 'third', 'fourth'][i]}): "
                                f"{el.describe()}, {place(el)}" for i, el in enumerate(options)}
        criteria["looked_at"] = "the one the user is looking at (\"this one\", \"that\"), not named or numbered"
        criteria["none"] = "none of them, or the answer isn't about the choice"
        result, _ = self._ask("pick", {"asked": question, "answer": answer}, {
            "pick": Choice(instructions="Which option does `answer` choose?", criteria=criteria)})
        choice = result.choices["pick"].choice
        return options[int(choice) - 1] if choice.isdigit() else choice

    def menu_command(self, action, combo, titles):
        """The frontmost app's menu command for a standard action ("Go to the previous tab", Ctrl+Shift+Tab), from
        its menu `titles` ("title (shortcut)"), or None if it has none."""
        criteria = {t: None for t in titles[:250]}
        criteria["none"] = "the app has no menu command for this"
        result, _ = self._ask("menu", {"action": action, "usual_shortcut": combo}, {
            "menu": Choice(instructions="Which of this app's menu commands does `action` (usually `usual_shortcut`)? "
                                        "Only one that does that same action to what's open; a command with a "
                                        "similar name that does something else (reloading the app's settings or "
                                        "configuration instead of the page) is none.", criteria=criteria)})
        choice = result.choices["menu"].choice
        return None if choice == "none" else choice

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
