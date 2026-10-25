"""What the notch says, as plain data: no AppKit here, so it's testable and the drawing code stays dumb.

The notch is for the happy path: look at something, say what to do, see it heard and done. The app builds a
NotchModel every tick from the tracker and the voice controller; these functions turn it into the few words each
part of the notch shows. (Eye health has its own reminders; the notch stays about commands.)
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

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
    mic: str = ""
    model: str | None = None  # the model being called right now, if any ("Jev", "claude-sonnet-5-5")
    activity: list = field(default_factory=list)  # recent [{said, did, t}], newest last
    research: list = field(default_factory=list)  # background research jobs, newest first (research.Job.summary)


TRACKING_COLOURS = {"tracking": "green", "uncalibrated": "amber", "looking away": "grey", "no face": "grey",
                    "calibrating": "blue", "starting": "grey"}

# Short labels for the collapsed notch (the pill's labels are longer)
SHORT = {"listening": "Listening", "hearing": "Hearing", "thinking": "Thinking", "writing": "Writing",
         "agent": "Agent", "acting": "Doing", "done": "Done", "waiting": "Your turn", "speaking": "Speaking",
         "dictating": "Dictating", "error": "Error"}


RESEARCH_PEEK_S = 10.0  # a finished research shows its answer under the notch this long


def researching(m: NotchModel) -> list[dict]:
    return [j for j in m.research if j.get("phase") in ("starting", "running")]


def ear_state(m: NotchModel) -> str | None:
    """What the ears show: what voice control is doing, else background research, else nothing (idle)."""
    return m.state or ("researching" if researching(m) else None)


def ear_label(m: NotchModel) -> str:
    """The word or two right of the notch while something's happening."""
    if m.state is None:
        running = researching(m)
        if running:
            return "Researching" if len(running) == 1 else f"Researching · {len(running)}"
        return ""
    step = re.search(r"step (\d+)", m.label) if m.state == "agent" else None
    if step:
        return f"Agent · {step[1]}"  # "Agent · step 3 · deep" -> keep it short
    if m.state == "done":
        return m.label.lstrip("✓ ").capitalize() or "Done"
    return SHORT.get(m.state, m.label)


def peek_line(m: NotchModel, now: float) -> tuple[str, str] | None:
    """(kind, text) for the line the notch grows to show without being asked, or None.
    Kinds: confirm (needs an answer), speech (what the agent says)."""
    if m.pending:
        return "confirm", f"{m.pending[0].upper()}{m.pending[1:]}?"  # the "say confirm" chip sits beside it
    if m.caption is not None:
        text, _, ended = m.caption
        if ended is None or now - ended < 1.5:
            return "speech", text
    for job in m.research:  # newest first
        if job.get("ended") and now - job["ended"] < RESEARCH_PEEK_S and job.get("phase") in ("done", "failed"):
            return "research", job.get("answer") if job["phase"] == "done" else "Research failed: try again later"
    return None


def headline(m: NotchModel) -> tuple[str, str]:
    """(title, subtitle) for the expanded notch's status row."""
    if m.voice_on is None:
        return "Pupil", _gaze_line(m) or "Starting…"
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
    if m.tracking == "no face":
        return "Can’t see you", "Face the screen to talk"
    if m.tracking == "looking away":
        return "Not listening", "You’re looking away: look at the screen to talk"
    return "Ready", "Look at something and say what to do"


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


def research_rows(m: NotchModel, now: float) -> list[dict]:
    """Background research for the open notch: running first, then the last finished ones (two at most in all)."""
    rows = []
    for job in m.research:
        phase = job.get("phase")
        if phase == "cancelled":
            continue
        if phase in ("starting", "running"):
            status = f"{int(job.get('elapsed_s') or 0)}s"
        elif phase == "done":
            n = len(job.get("sources") or [])
            status = f"{n} source{'s' if n != 1 else ''}"
        else:
            status = "failed"
        sources = job.get("sources") or []
        rows.append({"question": job.get("question") or job.get("said", ""), "phase": phase, "status": status,
                     "answer": job.get("answer") or "", "url": sources[0][1] if sources else ""})
    rows.sort(key=lambda r: r["phase"] not in ("starting", "running"))
    return rows[:2]


HINTS = ["“click this”", "“scroll down”", "“open Notes”", "“close that tab”"]


def recent(m: NotchModel, now: float, keep_s: float = 600.0, n: int = 3) -> list[dict]:
    """The last few commands, newest first, the last ten minutes only: {id, said, did, when, agent, details}."""
    out = []
    for item in reversed(m.activity):
        age = now - item.get("t", now)
        if age > keep_s:
            break
        when = "now" if age < 60 else f"{int(age // 60)}m"
        out.append({"id": item.get("id"), "said": item.get("said", ""), "did": item.get("did", ""), "when": when,
                    "agent": bool(item.get("agent")), "details": details(item)})
        if len(out) == n:
            break
    return out


def details(item: dict) -> list[str]:
    """What an expanded recent row says: the whole result, how it was handled, and when."""
    ep = item.get("ep") or {}
    route = "research" if ep.get("route") == "research" else ep.get("decision") or ep.get("route") or ""
    lines = [item.get("did", "")]
    if route == "research":
        how = "Researched in the background"
        if ep.get("research_sources") is not None:
            how += f" · {ep['research_sources']} sources · {ep.get('research_s')} s"
    elif route == "agent":
        how = " · ".join(str(p) for p in ("Agent", ep.get("agent_tier"), ep.get("agent_steps") and
                                          f"{ep['agent_steps']} steps", ep.get("agent_s") and f"{ep['agent_s']} s") if p)
    else:
        gate = ep.get("gate") or {}
        how = " · ".join(p for p in (f"Jev → {route}" if route else "", gate.get("why", ""),
                                     f"{ep['ms'] / 1000:.1f} s" if ep.get("ms") else "") if p)
    if how:
        lines.append(how)
    if item.get("t"):
        lines.append(time.strftime("at %H:%M:%S", time.localtime(item["t"])))
    return [line for line in lines if line]
