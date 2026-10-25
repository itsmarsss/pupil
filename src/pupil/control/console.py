"""What the terminal shows while voice control runs: one short block per command, nothing else by default.

    02:32:37  "Can you tell the Claude code session on the left terminal"
              ? asked: I missed the end. Say it again?                      Jev 109 ms · 0.3 s
    02:32:39  "summarize this page"
              ✦ agent (fast · quick)                                        Jev 104 ms
              ✦ agent done in 3.1 s, 2 steps: Three points: pricing, roadmap, team.
    02:32:43  · not you (lips still): "Shut up, just shut up"

Library chatter (HTTP requests, websocket connections, MediaPipe's C++ logs) is silenced; PUPIL_VERBOSE=1 brings
back every status line for debugging. The full record of each command is in .logs/episodes.jsonl either way.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import time

from ..core.env import env

VERBOSE = env("VERBOSE", "0") == "1"
_TTY = sys.stdout.isatty()
_INDENT = " " * 10

# route -> (mark, colour) for the outcome line
_MARKS = {"ignore": ("·", "dim"), "maybe": ("·", "dim"), "clarify": ("?", "yellow"), "choose": ("?", "yellow"),
          "agent": ("✦", "magenta"), "research": ("⌕", "cyan"), "cancel": ("■", "dim")}
_COLOURS = {"dim": "2", "yellow": "33", "magenta": "35", "cyan": "36", "green": "32", "red": "31", "bold": "1"}


def _c(text: str, colour: str) -> str:
    return f"\033[{_COLOURS[colour]}m{text}\033[0m" if _TTY else text


def _width() -> int:
    return max(60, min(shutil.get_terminal_size((110, 20)).columns, 140))


def _one_line(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _print(line: str) -> None:
    print(line, flush=True)


def quiet_libraries() -> None:
    """Keep the terminal for what Pupil did: no per-request HTTP or connection logs, no MediaPipe C++ logs."""
    os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe / TFLite (read when they load)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if VERBOSE:
        return
    for name in ("httpx", "httpx2", "httpcore", "typesafe_sdk", "websockets", "websockets.server", "mcp",
                 "anthropic", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def status(text: str) -> None:
    """A status line: only with PUPIL_VERBOSE=1 (the command's block says what mattered)."""
    if VERBOSE:
        _print(f"{_c(time.strftime('%H:%M:%S'), 'dim')}  {_c(_one_line(text, _width() - 10), 'dim')}")


def notice(text: str) -> None:
    """Something about the app itself (microphone switched, a service unavailable): always shown."""
    _print(f"{_c(time.strftime('%H:%M:%S'), 'dim')}  {text}")


def dropped(text: str, why: str) -> None:
    line = f"· not you ({why}): “{_one_line(text, 70)}”"
    _print(f"{_c(time.strftime('%H:%M:%S'), 'dim')}  {_c(line, 'dim')}")


def command(ep: dict) -> None:
    """A command finished (or was handed off): what was said, and what happened."""
    when = time.strftime("%H:%M:%S", time.localtime(ep.get("t", time.time())))
    route = "research" if ep.get("route") == "research" else ep.get("decision") or ep.get("route") or "?"
    said = _one_line(ep.get("text", ""), _width() - 14)
    mark, colour = _MARKS.get(route, ("✓", "green") if ep.get("acted") else ("·", "dim"))
    what = _outcome_text(ep, route)
    gate = (ep.get("gate") or {}).get("ms")
    timing = " · ".join(t for t in (f"Jev {gate} ms" if gate else "", f"{ep['ms'] / 1000:.1f} s" if "ms" in ep
                                    and route not in ("agent", "research") else "") if t)
    if route in ("ignore", "maybe"):
        quiet = f"· {what}: “{_one_line(ep.get('text', ''), 60)}”"
        _print(f"{_c(when, 'dim')}  {_c(quiet, 'dim')}")
        return
    _print(f"{_c(when, 'dim')}  {_c(f'“{said}”', 'bold')}")
    line = f"{mark} {_one_line(what, _width() - 14 - len(timing) - 4)}"
    pad = max(2, _width() - 10 - len(line) - len(timing))
    _print(f"{_INDENT}{_c(line, colour)}{' ' * pad}{_c(timing, 'dim')}")


def result(ep: dict, **fields) -> None:
    """Something that finished after its command's block: the agent's answer, a research answer."""
    if "agent_result" in fields:
        steps, secs = fields.get("agent_steps"), fields.get("agent_s")
        head = f"✦ agent done in {secs} s, {steps} step{'s' if steps != 1 else ''}: "
        _print(f"{_INDENT}{_c(head, 'magenta')}{_one_line(fields['agent_result'], _width() - 14 - len(head))}")
    elif "research_answer" in fields:
        head = f"⌕ research done in {fields.get('research_s')} s, {fields.get('research_sources')} sources: "
        _print(f"{_INDENT}{_c(head, 'cyan')}{_one_line(fields['research_answer'], _width() - 14 - len(head))}")
    elif "research_error" in fields:
        _print(f"{_INDENT}{_c('⌕ research ' + str(fields['research_error']), 'red')}")


def _outcome_text(ep: dict, route: str) -> str:
    status = ep.get("status") or ""
    if route == "ignore":
        return "ignored (not a command)"
    if route == "maybe":
        return "held (say “do it” to run)"
    if route == "agent":
        tier, budget = ep.get("agent_tier"), ep.get("agent_budget")
        return f"agent ({tier} · {budget}) working…" if tier else "agent working…"
    if route == "research":
        return "researching in the background"
    if route in ("clarify", "choose"):
        return f"asked: {status}"
    for prefix in ("Ignored: ", "Jev: "):
        status = status.removeprefix(prefix)
    return status or route
