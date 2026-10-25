"""App-specific helpers: apps that are far easier to drive through their own scripting than by clicking."""
from __future__ import annotations

from .. import commands
from . import blender


def open_path(name: str, path: str) -> str:
    """Open an app (Blender with its scripting bridge); returns a status line."""
    if blender.is_blender(name):
        return blender.launch(path)
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
