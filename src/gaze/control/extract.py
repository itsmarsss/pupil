"""The text inside a command, for what Jev's yes/no and pick-one answers can't carry.

Jev decides *what kind* of command it is ("type something", "go to a website", "several steps"); this pulls out
the words that go with it: the text to type, the URL or search, the steps in order. One fast model call with a
forced tool, so the answer is structured; None when it's unavailable or fails (the agent takes the command whole).
"""
from __future__ import annotations

import os

from . import config, models

_TOOLS = {
    "type": ({"text": {"type": "string", "description": "exactly what to type, as the user wants it written"}},
             "The user asked for text to be typed. Give the text to type, without the instruction around it: "
             "\"type hello world\" -> \"hello world\"; \"write that I'll be late\" -> \"I'll be late\"."),
    "web": ({"url": {"type": "string", "description": "a web address the user named, if they named one"},
             "query": {"type": "string", "description": "otherwise, what to search the web for"}},
            "The user wants a website or a web search. Give the address if they said one (\"go to github dot com\" "
            "-> github.com), else the search (\"look up flights to Boston\" -> flights to Boston)."),
    "steps": ({"steps": {"type": "array", "items": {"type": "string"}, "minItems": 2,
                         "description": "each action as its own short command, in order"}},
              "The user asked for several actions one after another. Split them into separate commands, in order, "
              "each understandable alone (\"open Safari and make a new tab\" -> [\"open Safari\", \"make a new "
              "tab\"]). Keep one action whole even if it mentions several things (\"search for salt and pepper\")."),
}

_client = None


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(timeout=config.EXTRACT_TIMEOUT_S, max_retries=0)
    return _client


def warm() -> None:
    """Open the API connection ahead of the first extraction (TLS setup ~1 s); no model call."""
    if available():
        try:
            _get_client().models.list(limit=1)
        except Exception:
            pass


def extract(kind: str, command: str, context: str = "") -> dict | None:
    """kind: type | web | steps. Returns the tool's fields, or None."""
    if not available():
        return None
    properties, instructions = _TOOLS[kind]
    tool = {"name": kind, "description": instructions,
            "input_schema": {"type": "object", "properties": properties,
                             "required": [] if kind == "web" else list(properties)}}
    content = f"Command: {command}" + (f"\nContext: {context}" if context else "")
    try:
        with models.calling("extract", config.EXTRACT_MODEL, kind):
            response = _get_client().messages.create(
                model=config.EXTRACT_MODEL, max_tokens=512, tools=[tool],
                tool_choice={"type": "tool", "name": kind}, messages=[{"role": "user", "content": content}])
    except Exception as e:
        print(f"[extract] {kind} failed: {e}", flush=True)
        return None
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    return None


def text_to_type(command: str) -> str | None:
    found = extract("type", command)
    return (found or {}).get("text") or None


def web(command: str) -> dict | None:
    """{"url": ...} or {"query": ...}."""
    found = extract("web", command) or {}
    if found.get("url"):
        return {"url": found["url"]}
    return {"query": found["query"]} if found.get("query") else None


def steps(command: str) -> list[str] | None:
    found = extract("steps", command) or {}
    parts = [s.strip() for s in found.get("steps", []) if s.strip()]
    return parts if len(parts) >= 2 else None
