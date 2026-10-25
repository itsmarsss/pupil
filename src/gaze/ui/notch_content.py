"""What the notch says, as plain data: no AppKit here, so it's testable and the drawing code stays dumb.

The app builds a NotchModel every tick from the tracker, the voice controller and the health tracker; these
functions turn it into the few words and numbers each part of the notch shows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Health signal strength (0 fine .. 1 strong sign of tiredness) -> tone, as on the dashboard
OK, WARN, BAD = "ok", "warn", "bad"


def tone(signal: float | None) -> str | None:
    if signal is None:
        return None
    return OK if signal < 0.35 else WARN if signal < 0.65 else BAD


@dataclass
class NotchModel:
    tracking: str = "starting"  # tracking | no face | looking away | uncalibrated | calibrating | starting
    accuracy_deg: float | None = None  # last measured gaze accuracy
    distance_cm: float | None = None
    fps: float = 0.0
    voice_on: bool | None = None  # None: voice control isn't running
    state: str | None = None  # the gaze pill's state (indicator.STYLES key), None when idle
    label: str = ""  # the gaze pill's label ("Agent · step 3")
    heard: str = ""  # what was last heard
    caption: tuple[str, float, float | None] | None = None  # (what the agent is saying, started, ended)
    pending: str | None = None  # waiting for "confirm": what
    reminder: tuple[str, float] | None = None  # (health reminder, until)
    mic: str = ""
    model: str | None = None  # the model being called right now, if any ("Jev", "claude-sonnet-5-5")
    health: dict = field(default_factory=dict)
    fatigue_series: list = field(default_factory=list)  # [(t, 0..100)], last 30 min
    reminders_on: bool = True


TRACKING_COLOURS = {"tracking": "green", "uncalibrated": "amber", "looking away": "grey", "no face": "grey",
                    "calibrating": "blue", "starting": "grey"}

# Short labels for the collapsed notch (the pill's labels are longer)
SHORT = {"listening": "Listening", "hearing": "Hearing", "thinking": "Thinking", "writing": "Writing",
         "agent": "Agent", "acting": "Doing", "done": "Done", "waiting": "Your turn", "speaking": "Speaking",
         "dictating": "Dictating", "error": "Error"}


def ear_label(m: NotchModel) -> str:
    """The word or two right of the notch while something's happening."""
    if m.state is None:
        return ""
    step = re.search(r"step (\d+)", m.label) if m.state == "agent" else None
    if step:
        return f"Agent · {step[1]}"  # "Agent · step 3 · deep" -> keep it short
    if m.state == "done":
        return m.label.lstrip("✓ ").capitalize() or "Done"
    return SHORT.get(m.state, m.label)


def peek_line(m: NotchModel, now: float) -> tuple[str, str] | None:
    """(kind, text) for the line the notch grows to show without being asked, or None.
    Kinds: confirm (needs an answer), speech (what the agent says), reminder (eye health)."""
    if m.pending:
        return "confirm", f"{m.pending[0].upper()}{m.pending[1:]}?"  # the "say confirm" chip sits beside it
    if m.caption is not None:
        text, _, ended = m.caption
        if ended is None or now - ended < 1.5:
            return "speech", text
    if m.reminder is not None and now < m.reminder[1]:
        return "reminder", m.reminder[0]
    return None


def headline(m: NotchModel) -> tuple[str, str]:
    """(title, subtitle) for the expanded notch's status row."""
    if m.voice_on is None:
        return "Gaze", _gaze_line(m) or "Starting…"
    if not m.voice_on:
        return "Voice control off", "Turn it on to talk to your Mac"
    if m.pending:
        return "Waiting for you", f"Say “confirm” to {m.pending[0].lower()}{m.pending[1:]}"
    if m.state is not None:
        title = {"listening": "Listening…", "hearing": "Hearing you…", "thinking": "Working out what you meant",
                 "writing": "Writing it out", "speaking": "Answering", "dictating": "Dictating",
                 "done": "Done"}.get(m.state, m.label)
        if m.state in ("agent", "acting"):
            title = m.label.replace("Agent: ", "Agent · ")
        sub = m.caption[0] if m.caption else (f"“{m.heard}”" if m.heard else "")
        return title, sub
    if m.tracking in ("no face", "looking away"):
        return "Can’t see you", "Face the screen to talk"
    return "Ready", (f"Last: “{m.heard}”" if m.heard else "Look at something and say what to do")


def _gaze_line(m: NotchModel) -> str:
    parts = []
    if m.accuracy_deg is not None:
        parts.append(f"±{m.accuracy_deg:.1f}°")
    if m.distance_cm:
        parts.append(f"{m.distance_cm:.0f} cm")
    if m.fps:
        parts.append(f"{m.fps:.0f} fps")
    return " · ".join(parts)


def footer(m: NotchModel) -> tuple[str, str]:
    """(left, right): gaze quality, and the model being called (else the mic)."""
    left = _gaze_line(m)
    left = f"{m.tracking.capitalize()}{' · ' + left if left else ''}"
    right = m.model or m.mic
    return left, right


@dataclass(frozen=True)
class Tile:
    title: str
    value: str
    detail: str
    tone: str | None
    progress: float | None = None  # 0..1 for a ring
    unit: str = ""  # small, after the value
    icon: str = ""  # SF Symbol beside the title


def tiles(m: NotchModel) -> list[Tile]:
    """Fatigue, blinking, breaks: the expanded notch's health row."""
    h, s = m.health or {}, (m.health or {}).get("signals") or {}
    fatigue = h.get("fatigue")
    level = h.get("level")
    level_tone = {"low": OK, "moderate": WARN, "high": BAD}.get(level)
    signals = {k: v for k, v in s.items() if v is not None and v >= 0.35}
    why = max(signals, key=signals.get) if signals else None
    why_text = {"blink_rate": "few blinks", "perclos": "heavy lids", "blink_duration": "slow blinks",
                "droop": "lids drooping", "screen_time": "long stretch", "gaze_drift": "unsteady gaze",
                "saccade_speed": "slow eyes", "eye_drag": "eyes dragging", "saccade_rate": "staring"}.get(why, "")
    out = [Tile("Fatigue", "–" if fatigue is None else f"{fatigue}",
                "measuring…" if fatigue is None else (why_text or "rested"), level_tone,
                None if fatigue is None else fatigue / 100, icon="bolt.heart.fill")]  # level: its colour
    rate = h.get("blink_rate")
    out.append(Tile("Blinks", "–" if rate is None else f"{rate:.0f}", "aim for 15+",
                    tone(s.get("blink_rate")), unit="" if rate is None else "/min", icon="eye.fill"))
    every = h.get("break_every_min") or 20
    minutes = h.get("minutes_on_screen") or 0.0
    left = every - minutes
    out.append(Tile("Break", "Now" if left <= 0 else f"{max(1, round(left))}",
                    "look 6 m away" if left <= 0 else f"{h.get('breaks') or 0} taken today",
                    WARN if left <= 0 else OK, min(minutes / every, 1.0), unit="" if left <= 0 else "min",
                    icon="cup.and.saucer.fill"))
    return out


def sparkline(series, now: float, window_s: float = 1800.0) -> list[tuple[float, float]]:
    """[(0..1 across the window, 0..1 value)] for the fatigue graph, oldest first."""
    pts = [((t - (now - window_s)) / window_s, v / 100) for t, v in series if now - t <= window_s and v is not None]
    return [(max(0.0, x), min(max(y, 0.0), 1.0)) for x, y in pts]
