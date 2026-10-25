"""Settings the agent can read and change ("use the AirPods mic", "switch to the other camera", "stop talking back").

Whoever owns a setting registers it (the app: camera, gaze display, smoothing; voice control: microphone, voice
isolation, spoken replies, ...) with a getter, a setter and its options; the agent's `settings` tool lists and
changes them by name. Setters must be safe to call from the agent's thread (marshal to the main thread if needed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Setting:
    name: str
    describe: str
    get: Callable[[], str]
    set: Callable[[str], str | None]  # returns a status line, or None
    options: Callable[[], list[str]] | None = None  # None: free text


_REGISTRY: dict[str, Setting] = {}


def register(name: str, describe: str, get, set, options=None) -> None:
    _REGISTRY[name] = Setting(name, describe, get, set, options)


def listing() -> str:
    if not _REGISTRY:
        return "No settings available."
    lines = []
    for s in _REGISTRY.values():
        try:
            current = s.get()
        except Exception as e:
            current = f"(unknown: {e})"
        opts = f" options: {', '.join(s.options())}" if s.options else ""
        lines.append(f"{s.name} = {current!r} — {s.describe}.{opts}")
    return "\n".join(lines)


def change(name: str, value: str) -> tuple[str, bool]:
    """Set `name` to `value` (case-insensitive; an option may be named by a unique part, "airpods" for
    "😨😨😨 AirPods Pro"). Returns (status, is_error)."""
    setting = _REGISTRY.get(name) or next((s for s in _REGISTRY.values() if name.lower() in s.name.lower()), None)
    if setting is None:
        return f"No setting {name!r}. There is: {', '.join(_REGISTRY)}", True
    if setting.options:
        options = setting.options()
        wanted = value.lower().strip()
        exact = [o for o in options if o.lower() == wanted]
        partial = [o for o in options if wanted in o.lower()]
        if not exact and len(partial) != 1:
            return f"{setting.name}: {value!r} matches {len(partial)} options; choose one of: {', '.join(options)}", True
        value = (exact or partial)[0]
    try:
        status = setting.set(value)
    except Exception as e:
        return f"Couldn't set {setting.name}: {e}", True
    return status or f"{setting.name} set to {value!r}", False
