"""Tier 0: exact command phrases, matched with rules in microseconds.

Anything that isn't an exact phrase comes back as kind "unknown" and goes to the Jev gate (gate.py),
which decides between a direct action, the Claude agent, or ignoring it. `correct` first snaps near-misses
("closed time" -> "close tab") to a known phrase or an on-screen label.
"""

import difflib
import re
from dataclasses import dataclass, field

from . import config


@dataclass
class Command:
    kind: str  # click | scroll | key | type | cancel | confirm | calibrate | recentre | check | unknown
    args: dict = field(default_factory=dict)
    text: str = ""


_DEICTIC = r"(?: (?:it|this|that|here|there|this one|that one))?"
_PATTERNS = [
    (rf"^(?:click|tap|press|select|open){_DEICTIC}$", lambda m: Command("click", {"clicks": 1, "button": "left"})),
    (rf"^double[ -]?click{_DEICTIC}$", lambda m: Command("click", {"clicks": 2, "button": "left"})),
    (rf"^right[ -]?click{_DEICTIC}$", lambda m: Command("click", {"clicks": 1, "button": "right"})),
    (r"^scroll (up|down|left|right)(?: (a lot|a little))?$",
     lambda m: Command("scroll", {"direction": m[1], "amount": {"a lot": 10, "a little": 1}.get(m[2], 4)})),
    (r"^(?:go )?back$", lambda m: Command("key", {"combo": "command+["})),
    (r"^(?:go )?forward$", lambda m: Command("key", {"combo": "command+]"})),
    (r"^new tab$", lambda m: Command("key", {"combo": "command+t"})),
    (r"^(?:stop|cancel|never ?mind|abort)$", lambda m: Command("cancel")),
    (r"^(?:confirm|yes|yes please|do it|go ahead)$", lambda m: Command("confirm")),
    (r"^(?:re)?calibrate$", lambda m: Command("calibrate")),
    (r"^re[ -]?cent(?:re|er)(?: (?:gaze|tracking))?$", lambda m: Command("recentre")),
    (r"^(?:check|test) (?:the )?(?:accuracy|tracking|gaze)$", lambda m: Command("check")),
]


# What people say for tier-0 commands and the shortcuts in commands.SHORTCUTS, for `correct`.
PHRASES = [
    "click", "double click", "right click", "scroll up", "scroll down", "scroll left", "scroll right",
    "scroll up a lot", "scroll down a lot", "scroll up a little", "scroll down a little", "go back", "go forward",
    "new tab", "close tab", "reopen tab", "next tab", "previous tab", "new window", "private window", "full screen",
    "zoom in", "zoom out", "actual size", "select all", "undo", "redo", "copy", "paste", "cut", "save", "find",
    "refresh", "address bar", "minimize", "hide app", "quit app", "switch app", "spotlight", "mission control",
    "escape", "enter", "stop", "cancel", "never mind", "confirm", "calibrate",
]
MATCH_MIN = 0.7  # measured: misheard commands 0.70-0.96 ("closed time" -> "close tab" 0.70), chatter <= 0.50
MATCH_MARGIN = 0.08  # and closer than the runner-up ("scroll dawn a lot": down a lot 0.94, up a lot 0.84)
MAX_WORDS = 5  # longer utterances are requests, not misheard commands
VERBS = {"click", "tap", "press", "select", "open"}
DEICTIC = {"this", "that", "these", "those", "it", "here", "there", "them"}  # meaning comes from the gaze  # "click search repose" -> "click Search repositories"


def correct(utterance: str, labels=()) -> str | None:
    """The known phrase or on-screen label (`labels`, e.g. "Search repositories") that `utterance` is a near-miss
    of, or None if it is exact, too long, or not clearly close to one thing."""
    norm = _normalize(utterance)
    if not norm or len(norm.split()) > MAX_WORDS:
        return None
    words = norm.split()
    if DEICTIC & set(words):  # "close this" means what you're looking at, not "close tab"
        return None
    if any(p.startswith(norm + " ") for p in PHRASES):  # "close" is a whole command, not a misheard "close tab"
        return None
    options = {p: p for p in PHRASES}
    verb = norm.split()[0] if norm.split()[0] in VERBS else ""
    for label in labels:
        options.setdefault(_normalize(label), label)
        if verb:
            options.setdefault(f"{verb} {_normalize(label)}", f"{verb} {label}")
    if norm in options:
        return None
    scored = sorted(((difflib.SequenceMatcher(None, norm, o).ratio(), o) for o in options if o), reverse=True)
    if not scored:
        return None
    best, second = scored[0], scored[1] if len(scored) > 1 else (0.0, "")
    if best[0] >= MATCH_MIN and best[0] - second[0] >= MATCH_MARGIN:
        return options[best[1]]
    return None


def _normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text.lower())).strip()


def route(utterance: str, wake: bool = True) -> Command | None:
    """None = not addressed to us (wake word missing). wake=False: already checked (re-routing a correction)."""
    text = utterance.strip()
    norm = _normalize(text)
    if config.WAKE_WORD and wake:
        if not norm.startswith(config.WAKE_WORD):
            return None
        norm = norm[len(config.WAKE_WORD):].strip()
        text = text[text.lower().find(config.WAKE_WORD) + len(config.WAKE_WORD):].lstrip(" ,.:")

    for pattern, build in _PATTERNS:
        m = re.match(pattern, norm)
        if m:
            cmd = build(m)
            cmd.text = text
            return cmd

    # "type hello world" -> click the snapped field, then type. Keep the original casing/punctuation.
    m = re.match(r"^\s*type\s+(.+)$", text, flags=re.IGNORECASE)
    if m:
        return Command("type", {"text": m[1]}, text)

    return Command("unknown", {"request": text}, text)


_CLAUSE_BREAK = re.compile(r"\s*,?\s*\b(?:and then|after that|then|and)\b\s*,?\s*", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    """ "open Safari and then make a new tab" -> ["open Safari", "make a new tab"]. Only used once the gate has
    judged the command to be several actions, so "salt and pepper" inside one action isn't split."""
    return [c.strip(" ,.") for c in _CLAUSE_BREAK.split(text) if c.strip(" ,.")]
