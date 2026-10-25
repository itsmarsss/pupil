"""Tier 0: exact command phrases, matched with rules in microseconds.

Anything that isn't an exact phrase comes back as kind "unknown" and goes to the Jev gate (gate.py),
which decides between a direct action, the Claude agent, or ignoring it.
"""

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


def _normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", text.lower())).strip()


def route(utterance: str) -> Command | None:
    """None = not addressed to us (wake word missing)."""
    text = utterance.strip()
    norm = _normalize(text)
    if config.WAKE_WORD:
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
