"""Local dashboard server: streams the engine's state to the web dashboard and takes commands back.

One port on 127.0.0.1 serves both the built web app (HTTP) and a WebSocket at /ws:
  - text messages: JSON `scene` and `calibration` (on connect and when they change) and `frame` states (~30 Hz)
  - binary messages: the camera image as JPEG (~12 Hz, 640 px wide), prefixed by b"JPG0"
  - from the client: JSON {"cmd": name}; names map to callables supplied by the app

States are built on the server's own thread from the latest tracker sample, so a slow or absent client
never slows tracking down.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import threading
import time
from pathlib import Path
from typing import Callable

import cv2

from ..core.device import load_device
from ..core.types import Frame, Sample
from ..telemetry import calibration_info, frame_state, scene_info

log = logging.getLogger(__name__)

STATE_HZ = 30
IMAGE_HZ = 12
IMAGE_WIDTH = 640
IMAGE_PREFIX = b"JPG0"
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class DashboardServer:
    def __init__(self, tracker, context: Callable[[], dict] | None = None,
                 commands: dict[str, Callable[[], None]] | None = None,
                 lid: Callable[[], float | None] | None = None,
                 host: str = "127.0.0.1", port: int = 8766, static_dir: Path = WEB_DIST):
        self.tracker = tracker
        self.context = context or (lambda: {})
        self.commands = commands or {}
        self.lid = lid or (lambda: None)
        self.host, self.port, self.static_dir = host, port, static_dir
        self._latest: tuple[Frame, Sample] | None = None
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop: asyncio.Event | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    # ---- tracker side (inference thread) ----------------------------------------------
    def on_frame(self, frame: Frame, sample: Sample) -> None:
        self._latest = (frame, sample)  # a single reference swap; the server reads the newest

    # ---- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        self.tracker.add_frame_listener(self.on_frame)
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), name="dashboard", daemon=True)
        self._thread.start()
        ready.wait(timeout=5)

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=3)

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
            await ws.send(json.dumps(self._scene()))
            await ws.send(json.dumps(calibration_info(self.tracker.profile)))
            async for message in ws:
                self._command(message)
        except Exception:  # client went away
            pass
        finally:
            self._clients.discard(ws)

    def _command(self, message) -> None:
        try:
            name = json.loads(message).get("cmd")
        except (ValueError, AttributeError):
            return
        fn = self.commands.get(name)
        if fn is None:
            log.warning("unknown dashboard command %r", name)
            return
        fn()

    def _scene(self) -> dict:
        return scene_info(self.tracker.screen, self.tracker.plane, load_device(), self.lid())

    def _calib_key(self):
        profile = self.tracker.profile
        return id(profile), getattr(getattr(profile, "meta", None), "updated", None)

    async def _broadcast(self) -> None:
        last_t, last_image, last_calib = 0, 0.0, self._calib_key()  # new clients get it on connect
        while not self._stop.is_set():
            await asyncio.sleep(1 / STATE_HZ)
            latest = self._latest
            if not self._clients or latest is None:
                continue
            frame, sample = latest
            messages: list = []
            profile = self.tracker.profile
            calib_key = self._calib_key()
            if calib_key != last_calib:
                last_calib = calib_key
                messages.append(json.dumps(calibration_info(profile)))
            if sample.t_ns != last_t:
                last_t = sample.t_ns
                state = frame_state(sample, profile, self.tracker.plane, self.context(), self.tracker.fps)
                messages.append(json.dumps(state))
            now = time.monotonic()
            if now - last_image >= 1 / IMAGE_HZ:
                last_image = now
                messages.append(IMAGE_PREFIX + self._jpeg(frame))
            for ws in list(self._clients):
                for m in messages:
                    try:
                        await ws.send(m)
                    except Exception:
                        self._clients.discard(ws)
                        break

    @staticmethod
    def _jpeg(frame: Frame) -> bytes:
        img = frame.image
        if img.shape[1] > IMAGE_WIDTH:
            img = cv2.resize(img, (IMAGE_WIDTH, round(img.shape[0] * IMAGE_WIDTH / img.shape[1])), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes() if ok else b""
