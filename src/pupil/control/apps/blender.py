"""Blender, scripted: launch it with the bridge (blender_bridge.py) and run Python in it.

Only a Blender started here has the bridge; one the user opened themselves doesn't, and `run` says so.
"""
from __future__ import annotations

import json
import secrets
import socket
import subprocess
import time
from pathlib import Path

from ...core.paths import APP_SUPPORT
from ...core.env import env

BRIDGE = Path(__file__).with_name("blender_bridge.py")
TOKEN_FILE = APP_SUPPORT / "blender_bridge.token"  # the bridge reads the same path
LOG = APP_SUPPORT / "blender.log"
PORT = int(env("BLENDER_PORT", "47810"))
START_TIMEOUT_S = 30


def is_blender(name: str) -> bool:
    return name.lower().startswith("blender")


def _binary(app_path: str) -> str:
    return str(Path(app_path) / "Contents" / "MacOS" / "Blender")


def _running() -> bool:
    return subprocess.run(["pgrep", "-f", "Blender.app/Contents/MacOS/Blender"], capture_output=True).returncode == 0


def run(code: str, timeout: float = 60) -> dict:
    """Run Python in Blender: {"ok", "output", "result"} or {"ok": False, "error"}."""
    try:
        token = TOKEN_FILE.read_text().strip()
        with socket.create_connection(("127.0.0.1", PORT), timeout=2) as s:
            s.settimeout(timeout + 5)
            s.sendall(json.dumps({"token": token, "code": code, "timeout": timeout}).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
        return json.loads(data)
    except (OSError, ValueError):
        if _running():
            return {"ok": False, "error": "Blender is open but wasn't started by voice control, so it can't be "
                                          "scripted. Ask the user to save and quit Blender, then say "
                                          "\"open Blender\" (or call open_app); until then use the mouse and keys."}
        return {"ok": False, "error": "Blender isn't running: call open_app with \"Blender\" first."}


def bridge_up() -> bool:
    return run("1", timeout=2).get("ok", False)


def launch(app_path: str) -> str:
    """Open Blender with the bridge (or bring it forward); returns a status line."""
    if bridge_up():
        subprocess.Popen(["open", "-a", app_path])
        return "Blender is open and scriptable."
    if _running():
        subprocess.Popen(["open", "-a", app_path])
        return "Blender was already open without the bridge: it can't be scripted until it's reopened from here."
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(secrets.token_hex(16))
    TOKEN_FILE.chmod(0o600)
    with open(LOG, "ab") as log:
        subprocess.Popen([_binary(app_path), "--python", str(BRIDGE)], stdout=log, stderr=log,
                         start_new_session=True)  # outlives voice control
    deadline = time.time() + START_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(0.5)
        if bridge_up():
            subprocess.Popen(["open", "-a", app_path])  # to the front
            return "Opened Blender (scriptable)."
    return "Opened Blender, but the scripting bridge didn't start (see Pupil/blender.log in Application Support)."


# ---- as an agent adapter (apps/__init__.py) ----------------------------------------------------

APP = "Blender"
TOOL = {
    "name": "blender_python",
    "description": "Run Python inside Blender (bpy is imported) and get back what it printed and the value of its "
                   "last expression, like Blender's Python console. Far faster and more precise than clicking for "
                   "building or editing scenes: meshes, modifiers, materials, cameras, lights, rendering. Each call "
                   "is one undo step for the user. Only works on a Blender opened by open_app; the error says what "
                   "to do otherwise. Take a screenshot after bigger changes to check the result, keep each call to "
                   "one logical step, and build alongside what's already in the scene: don't delete or replace "
                   "existing objects (the default cube included) unless asked.",
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


def call(inp: dict) -> tuple[str, bool]:
    res = run(inp["code"], timeout=120)
    if not res.get("ok"):
        return "\n".join(filter(None, [res.get("output"), res.get("error")])), True
    parts = [res.get("output", "").rstrip(), f"=> {res['result']}" if res.get("result") else ""]
    return "\n".join(p for p in parts if p) or "OK (no output)", False


def describe(inp: dict) -> str:
    return f"run Python in Blender:\n{inp.get('code', '')[:600]}"
