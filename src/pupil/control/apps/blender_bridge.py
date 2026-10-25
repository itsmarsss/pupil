"""Runs inside Blender (`Blender --python blender_bridge.py`): lets voice control's Claude agent run Python here.

Blender draws its own UI (no Accessibility tree), so building anything by clicking takes hundreds of precise mouse
moves; its Python API (bpy) does it in a few calls. Requests come over a socket on 127.0.0.1 only and must carry
the token the launcher wrote to an owner-only file. Code runs on Blender's main thread (bpy isn't thread-safe),
one request at a time; stdout and the value of a final expression come back, like the Python console.
"""
import ast
import contextlib
import io
import json
import os
import queue
import socket
import threading
import traceback

import bpy

PORT = int(os.environ.get("PUPIL_BLENDER_PORT") or os.environ.get("GAZE_BLENDER_PORT", "47810"))
TOKEN_FILE = os.path.expanduser("~/Library/Application Support/Gaze/blender_bridge.token")
MAX_OUTPUT = 8000

_jobs: queue.Queue = queue.Queue()


def _run(code):
    out = io.StringIO()
    try:
        tree = ast.parse(code)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        scope = {"bpy": bpy, "__name__": "__gaze__"}
        result = None
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(tree, "<gaze>", "exec"), scope)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<gaze>", "eval"), scope)
                result = None if value is None else repr(value)[:MAX_OUTPUT]
        try:  # one undo step per request, so the user can Cmd+Z what the agent did
            bpy.ops.ed.undo_push(message="gaze agent")
        except Exception:
            pass
        return {"ok": True, "output": out.getvalue()[-MAX_OUTPUT:], "result": result}
    except Exception:
        return {"ok": False, "output": out.getvalue()[-MAX_OUTPUT // 2:],
                "error": traceback.format_exc(limit=6)[-MAX_OUTPUT // 2:]}


def _pump():
    """Blender timer: run queued requests on the main thread."""
    while True:
        try:
            code, reply = _jobs.get_nowait()
        except queue.Empty:
            return 0.05
        reply.put(_run(code))


def _handle(conn, token):
    with conn:
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                return
            data += chunk
        try:
            req = json.loads(data)
        except ValueError:
            return
        if req.get("token") != token:
            res = {"ok": False, "error": "bad token"}
        else:
            reply: queue.Queue = queue.Queue()
            _jobs.put((req.get("code", ""), reply))
            try:
                res = reply.get(timeout=float(req.get("timeout", 60)))
            except queue.Empty:
                res = {"ok": False, "error": "timed out waiting for Blender (busy, or a modal dialog is open)"}
        conn.sendall(json.dumps(res).encode() + b"\n")


def _serve():
    with open(TOKEN_FILE) as f:
        token = f.read().strip()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(4)
    print(f"gaze bridge listening on 127.0.0.1:{PORT}")
    while True:
        conn, _ = srv.accept()
        threading.Thread(target=_handle, args=(conn, token), daemon=True).start()


bpy.app.timers.register(_pump, persistent=True)
threading.Thread(target=_serve, name="gaze-bridge", daemon=True).start()
