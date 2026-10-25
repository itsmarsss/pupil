"""Which model is being called right now, and the last call of each, for the dashboard's Voice & agent panel.

Every model call goes through `calling(role, model, what)`; streams that stay open (speech-to-text, speaking)
use `streaming(role, model, on)`. Safe from any thread.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

ROLES = {  # role -> what it does, in the order the panel shows them
    "stt": "hears you",
    "jev": "decides what you meant",
    "extract": "pulls out text to type, URLs, steps",
    "dictation": "formats dictation",
    "agent": "does multi-step tasks",
    "tts": "speaks replies",
}

_lock = threading.Lock()
_state: dict[str, dict] = {}  # role -> {model, active, what, t, ms}
_version = 0


def _set(role, **fields):
    global _version
    with _lock:
        _state[role] = {**_state.get(role, {}), **fields}
        _version += 1


@contextmanager
def calling(role: str, model: str, what: str = ""):
    t0 = time.time()
    _set(role, model=model, active=True, what=what, t=t0)
    try:
        yield
    finally:
        _set(role, active=False, t=time.time(), ms=round((time.time() - t0) * 1000))


def streaming(role: str, model: str, on: bool, what: str = "") -> None:
    """A stream that's open while something happens (listening, speaking), not a single request."""
    with _lock:
        same = _state.get(role, {}).get("active") == on and _state.get(role, {}).get("model") == model
    if not same:
        _set(role, model=model, active=on, what=what, t=time.time(), **({} if on else {"ms": None}))


def configure(role: str, model: str) -> None:
    """The model a role will use, shown before its first call."""
    with _lock:
        known = role in _state
    if not known:
        _set(role, model=model, active=False)


def used(role: str, model: str) -> None:
    """The model that actually answered, when it isn't the one asked (a fallback)."""
    _set(role, model=model)


def version() -> int:
    return _version


def snapshot() -> list[dict]:
    with _lock:
        return [{"role": role, "does": ROLES[role], **_state[role]} for role in ROLES if role in _state]
