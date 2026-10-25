"""App adapters: apps that are far easier to drive through their own scripting than by clicking.

The agent's tools are app-agnostic (screen, mouse, keyboard, menus, accessibility, AppleScript); an adapter adds
one more tool for one app, offered only when that app is running or named in the request. An adapter module has
APP (the app's name), TOOL (the tool's schema), call(input) -> (text, is_error), describe(input) for the risk
check, and launch(app_path) -> status (opening the app in its scriptable form). Add one by listing it in ADAPTERS.
"""
from __future__ import annotations

import subprocess

from .. import commands
from . import blender

ADAPTERS = [blender]


def _running(app: str) -> bool:
    return subprocess.run(["pgrep", "-xi", app], capture_output=True).returncode == 0 or \
        subprocess.run(["pgrep", "-f", f"/{app}.app/"], capture_output=True).returncode == 0


def adapter_for_app(name: str):
    return next((a for a in ADAPTERS if name.lower().startswith(a.APP.lower())), None)


def tools_for(request: str) -> list[dict]:
    """Adapter tools worth offering for this request: their app is running or named in it."""
    text = request.lower()
    return [a.TOOL for a in ADAPTERS if a.APP.lower() in text or _running(a.APP)]


def call(name: str, inp: dict):
    """(text, is_error) if `name` is an adapter tool, else None."""
    adapter = next((a for a in ADAPTERS if a.TOOL["name"] == name), None)
    return adapter.call(inp) if adapter else None


def describe(name: str, inp: dict) -> str | None:
    adapter = next((a for a in ADAPTERS if a.TOOL["name"] == name), None)
    return adapter.describe(inp) if adapter else None


def open_path(name: str, path: str) -> str:
    """Open an app, in its scriptable form if an adapter has one; returns a status line."""
    adapter = adapter_for_app(name)
    if adapter is not None:
        return adapter.launch(path)
    commands.open_app(path)
    return f"Opened {name}."


def open_app(name: str) -> tuple[str, bool]:
    """Open an app by (approximate) name: (status, is_error)."""
    apps = commands.available_apps()
    wanted = name.lower().removesuffix(".app").strip()
    match = next((n for n in apps if n.lower() == wanted), None) or next(
        (n for n in apps if wanted in n.lower()), None)
    if match is None:
        return f"No app called {name!r} is installed.", True
    return open_path(match, apps[match]), False
