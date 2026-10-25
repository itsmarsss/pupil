"""Tier 0: the few phrases that must work instantly and offline, matched with rules in microseconds.

Anything that isn't an exact phrase comes back as kind "unknown" and goes to the Jev gate (gate.py),
which decides between a direct action, the agent, or ignoring it. Misheard words ("closed time") are Jev's to
read too: fuzzy-matching them to known phrases here turned chatter into commands ("please" -> paste).
"""

import re
from dataclasses import dataclass, field

from . import config


@dataclass
class Command:
    kind: str  # cancel | confirm | calibrate | recentre | check | dictate_on | dictate_off | unknown (for Jev)
    args: dict = field(default_factory=dict)
    text: str = ""


# Only what must work instantly, offline and mid-anything: stopping, confirming, dictation on/off (while dictating
# everything else is typed) and the tracker's own controls. Everything else is Jev's to read (gate.py).
_PATTERNS = [
    (r"^(?:stop|cancel|never ?mind|abort)$", lambda m: Command("cancel")),
    (r"^(?:confirm|yes|yes please|do it|go ahead)$", lambda m: Command("confirm")),
    (r"^(?:re)?calibrate$", lambda m: Command("calibrate")),
    (r"^re[ -]?cent(?:re|er)(?: (?:gaze|tracking))?$", lambda m: Command("recentre")),
    (r"^(?:check|test) (?:the )?(?:accuracy|tracking|gaze)$", lambda m: Command("check")),
    (r"^(?:start )?(?:dictation|dictating)(?: mode)?$|^start dictating$|^dictate$", lambda m: Command("dictate_on")),
    (r"^(?:stop|end|finish|exit) (?:dictation|dictating)(?: mode)?$", lambda m: Command("dictate_off")),
    (r"^forget that$|^clear (?:your )?memory$", lambda m: Command("forget")),
]


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

    return Command("unknown", {"request": text}, text)
