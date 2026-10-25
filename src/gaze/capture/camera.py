"""Frame sources: the built-in camera (AVFoundation via OpenCV) and recorded files."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..core.types import Frame

log = logging.getLogger(__name__)


class FrameSource(Protocol):
    def start(self) -> None: ...
    def latest(self, after_t_ns: int = 0, timeout: float = 1.0) -> Frame | None: ...
    def stop(self) -> None: ...


@dataclass(frozen=True)
class CaptureMode:
    """`request` is what we ask the camera for; frames are centre-cropped to `aspect`."""

    name: str
    request: tuple[int, int]
    output: tuple[int, int] | None = None  # resize after cropping
    aspect: float = 4 / 3


# The gaze model was trained on 640x480 (4:3) frames. Both modes capture 1920x1080 and
# centre-crop to 1440x1080, so they share one field of view; "vga" then shrinks to
# 640x480 to match training resolution, "hd" keeps ~2.25x the pixels per eye.
# (Asking the camera for 640x480 directly gives a different framing.)
MODES = {
    "vga": CaptureMode("vga", (1920, 1080), output=(640, 480)),
    "hd": CaptureMode("hd", (1920, 1080)),
}


def list_cameras() -> list[str]:
    """Camera names in the order OpenCV's AVFoundation backend indexes them: video and muxed devices sorted
    by unique id. (AVFoundation's own order moves recently used cameras first, so it can't map indices.)"""
    import AVFoundation as AV

    devices = [*AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo),
               *AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed)]
    return [str(d.localizedName()) for d in sorted(devices, key=lambda d: str(d.uniqueID()))]


def find_camera(preferred: str = "FaceTime") -> int:
    for i, name in enumerate(list_cameras()):
        if preferred.lower() in name.lower():
            return i
    return 0


def crop_to_aspect(image: np.ndarray, aspect: float) -> np.ndarray:
    h, w = image.shape[:2]
    if abs(w / h - aspect) < 1e-3:
        return image
    if w / h > aspect:
        new_w = int(round(h * aspect))
        x0 = (w - new_w) // 2
        return image[:, x0 : x0 + new_w]
    new_h = int(round(w / aspect))
    y0 = (h - new_h) // 2
    return image[y0 : y0 + new_h]


class _LatestFrameSource:
    """Runs `_read()` on a thread and keeps only the newest frame (no backlog)."""

    source_id = ""

    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._running = False

    def _open(self) -> None: ...
    def _read(self) -> np.ndarray | None: ...
    def _close(self) -> None: ...

    def start(self) -> None:
        if self._running:
            return
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name=f"capture-{self.source_id}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            image = self._read()
            if image is None:
                time.sleep(0.005)
                continue
            frame = Frame(time.monotonic_ns(), image, self.source_id)
            with self._cond:
                self._frame = frame
                self._cond.notify_all()

    def latest(self, after_t_ns: int = 0, timeout: float = 1.0) -> Frame | None:
        """Newest frame captured after `after_t_ns`, waiting up to `timeout` seconds."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame is None or self._frame.t_ns <= after_t_ns:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._running:
                    return None
                self._cond.wait(remaining)
            return self._frame

    def stop(self) -> None:
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self._close()


class CameraSource(_LatestFrameSource):
    def __init__(self, mode: str = "hd", camera: str | int = "FaceTime", fps: int = 30) -> None:
        super().__init__()
        self.mode = MODES[mode]
        self.index = camera if isinstance(camera, int) else find_camera(camera)
        self.fps = fps
        names = list_cameras()
        self.source_id = names[self.index] if self.index < len(names) else f"camera{self.index}"
        self._cap: cv2.VideoCapture | None = None
        self.delivered: tuple[int, int] | None = None  # what the camera actually sends

    def _open(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.source_id!r}. Allow camera access for your terminal in "
                "System Settings → Privacy & Security → Camera."
            )
        w, h = self.mode.request
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap = cap
        log.info(
            "camera %r opened at %dx%d (%s mode)",
            self.source_id,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            self.mode.name,
        )

    @property
    def downgraded(self) -> bool:
        """True when the camera sends fewer pixels than requested, e.g. because another
        app opened it first at a lower resolution."""
        return self.delivered is not None and self.delivered[1] < self.mode.request[1]

    def _read(self) -> np.ndarray | None:
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None
        size = (bgr.shape[1], bgr.shape[0])
        if size != self.delivered:
            self.delivered = size
            if self.downgraded:
                log.warning("camera is delivering %dx%d, not %dx%d; another app may be using it",
                            *size, *self.mode.request)
        rgb = cv2.cvtColor(crop_to_aspect(bgr, self.mode.aspect), cv2.COLOR_BGR2RGB)
        if self.mode.output and (rgb.shape[1], rgb.shape[0]) != self.mode.output:
            rgb = cv2.resize(rgb, self.mode.output, interpolation=cv2.INTER_AREA)
        return rgb

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class FileSource(_LatestFrameSource):
    """Replays a video file or a single image at `fps`, looping, for tests and evaluation."""

    def __init__(self, path: str | Path, fps: float = 30.0, loop: bool = True) -> None:
        super().__init__()
        self.path = Path(path)
        self.fps = fps
        self.loop = loop
        self.source_id = self.path.name
        self._cap: cv2.VideoCapture | None = None
        self._image: np.ndarray | None = None
        self._next_t = 0.0

    def _open(self) -> None:
        if self.path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            bgr = cv2.imread(str(self.path))
            if bgr is None:
                raise FileNotFoundError(self.path)
            self._image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            self._cap = cv2.VideoCapture(str(self.path))
            if not self._cap.isOpened():
                raise FileNotFoundError(self.path)
        self._next_t = time.monotonic()

    def _read(self) -> np.ndarray | None:
        delay = self._next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next_t = max(self._next_t + 1 / self.fps, time.monotonic() - 1 / self.fps)
        if self._image is not None:
            return self._image
        ok, bgr = self._cap.read()
        if not ok and self.loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._cap.read()
        if not ok:
            self._running = False
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
