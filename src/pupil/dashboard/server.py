"""Local dashboard server: streams a feed (live tracker or replay) to the web dashboard, takes commands back.

One port on 127.0.0.1 serves both the built web app (HTTP) and a WebSocket at /ws:
  - text messages: JSON `scene` and `calibration` (on connect and when they change) and `frame` states (~30 Hz)
  - binary messages: the camera image as JPEG, prefixed by b"JPG0"
  - from the client: JSON {"cmd": name, "value": optional}; the feed handles its own commands (replay
    playback), the rest map to callables supplied by the app
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import socket
import threading
from pathlib import Path
from typing import Callable

from .feeds import Feed

log = logging.getLogger(__name__)

STATE_HZ = 30
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


PORT_TRIES = 10


class DashboardServer:
    def __init__(self, feed: Feed, commands: dict[str, Callable[[], None]] | None = None,
                 host: str = "127.0.0.1", port: int = 8766, static_dir: Path = WEB_DIST):
        self.feed = feed
        self.commands = commands or {}
        self.host, self.port, self.static_dir = host, port, static_dir
        self.port = self._free_port()  # now, so `url` is right before the server starts
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ---- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="dashboard", daemon=True)
        self._thread.start()
        ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=3)

    def serve_forever(self) -> None:
        """Run in the calling thread until interrupted (for replays, which have no app loop)."""
        try:
            self._run(threading.Event())
        except KeyboardInterrupt:
            pass

    def _free_port(self) -> int:
        """`port`, or the next one nothing answers on. Binding alone doesn't tell: another server listening on all
        addresses (e.g. `python -m http.server`, IPv6 dual-stack) lets us bind 127.0.0.1 yet takes the requests."""
        for port in range(self.port, self.port + PORT_TRIES):
            try:
                socket.create_connection((self.host, port), timeout=0.2).close()
            except OSError:
                return port
            log.warning("port %d is taken by another server; trying %d", port, port + 1)
        return self.port

    def _run(self, ready: threading.Event) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main(ready))
        except OSError as e:
            log.error("dashboard could not start on %s: %s", self.url, e)
            ready.set()

    async def _main(self, ready: threading.Event) -> None:
        from websockets.asyncio.server import serve

        self._stop = asyncio.Event()
        async with serve(self._client, self.host, self.port, process_request=self._http, max_size=2**16):
            log.info("dashboard at %s", self.url)
            ready.set()
            await asyncio.gather(self._broadcast(), self._stop.wait())

    # ---- HTTP: the built web app -------------------------------------------------------
    def _http(self, connection, request):
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        path = request.path.split("?")[0]
        if path == "/ws":
            return None  # proceed with the WebSocket handshake
        if not self.static_dir.is_dir():
            body = b"Dashboard not built. Run: cd web && pnpm install && pnpm build\n"
            return Response(503, "Service Unavailable", Headers({"Content-Type": "text/plain"}), body)
        rel = path.lstrip("/") or "index.html"
        target = (self.static_dir / rel).resolve()
        if self.static_dir.resolve() not in target.parents or not target.is_file():
            target = self.static_dir / "index.html"  # single-page app
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return Response(200, "OK", Headers({"Content-Type": ctype, "Cache-Control": "no-cache"}), target.read_bytes())

    # ---- WebSocket ----------------------------------------------------------------------
    async def _client(self, ws) -> None:
        self._clients.add(ws)
        try:
            for m in self.feed.hello():
                await ws.send(m)
            async for message in ws:
                self._command(message)
        except Exception:  # client went away
            pass
        finally:
            self._clients.discard(ws)

    def _command(self, message) -> None:
        try:
            msg = json.loads(message)
            name, value = msg.get("cmd"), msg.get("value")
        except (ValueError, AttributeError):
            return
        if self.feed.command(name, value):
            return
        fn = self.commands.get(name)
        if fn is None:
            log.warning("unknown dashboard command %r", name)
            return
        fn()

    async def _broadcast(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(1 / STATE_HZ)
            if not self._clients:
                continue
            messages = self.feed.poll()
            for ws in list(self._clients):
                for m in messages:
                    try:
                        await ws.send(m)
                    except Exception:
                        self._clients.discard(ws)
                        break
