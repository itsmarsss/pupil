"""Screenshots for Claude: downscaled to the recommended size, with gaze annotations; coordinate conversion."""

import base64
import io
import time
from collections import deque
from dataclasses import dataclass

import threading

import Quartz
from PIL import Image, ImageDraw

from . import config

CAPTURE_TIMEOUT_S = 3.0
FRAME_PERIOD_S = 0.5  # the screen while you speak, this often
FRAMES_KEPT = 12  # ~6 s of speech
FRAME_CROP_PT = 360  # full-resolution close-up kept around the gaze, this many points square


class ScreenshotUnavailable(RuntimeError):
    pass


def _main_display_points():
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return bounds.size.width, bounds.size.height


class Screen:
    """Three coordinate spaces: physical pixels (Retina), screen points (what macOS input uses),
    and screenshot pixels (what Claude sees and answers in)."""

    def __init__(self):
        self._capturing: threading.Thread | None = None  # a capture that may still be stuck
        self.w_pt, self.h_pt = _main_display_points()
        max_w, max_h = config.SCREENSHOT_MAX
        self.scale = min(max_w / self.w_pt, max_h / self.h_pt, 1.0)  # points -> screenshot pixels
        self.size = (round(self.w_pt * self.scale), round(self.h_pt * self.scale))

    def to_points(self, x, y):
        return x / self.scale, y / self.scale

    def to_shot(self, x, y):
        return x * self.scale, y * self.scale

    def _grab(self):
        """The main display in physical pixels, or ScreenshotUnavailable within CAPTURE_TIMEOUT_S. macOS screen
        capture can stall system-wide (seen: 30 s per grab while a Screenshot-app session held the display), and an
        agent step shouldn't wait on it: the agent carries on with its accessibility and menu tools instead."""
        if self._capturing is not None and self._capturing.is_alive():
            raise ScreenshotUnavailable("screen capture is still stuck on the previous screenshot")
        result = {}

        def work():
            try:
                result["image"] = self._capture()
            except Exception as e:
                result["error"] = e

        self._capturing = threading.Thread(target=work, name="capture", daemon=True)
        self._capturing.start()
        self._capturing.join(CAPTURE_TIMEOUT_S)
        if self._capturing.is_alive():
            raise ScreenshotUnavailable(f"screen capture didn't respond within {CAPTURE_TIMEOUT_S:.0f} s")
        if "error" in result:
            raise ScreenshotUnavailable(str(result["error"]))
        return result["image"]

    def _capture(self):
        """Quartz directly (~70 ms when healthy)."""
        bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
        image = Quartz.CGWindowListCreateImage(bounds, Quartz.kCGWindowListOptionOnScreenOnly,
                                               Quartz.kCGNullWindowID, Quartz.kCGWindowImageDefault)
        if image is None:
            raise RuntimeError("screen capture failed: allow Screen Recording for your terminal")
        w, h = Quartz.CGImageGetWidth(image), Quartz.CGImageGetHeight(image)
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
        return Image.frombuffer("RGB", (w, h), bytes(data), "raw", "BGRX", Quartz.CGImageGetBytesPerRow(image), 1)

    def screenshot(self, gaze=None, target=None):
        """Screenshot at Claude's size. `gaze` / `target` (screen points) are drawn on it if given."""
        return self.mark(self._grab().resize(self.size, Image.LANCZOS), gaze, target)

    def mark(self, img, gaze=None, target=None):
        """Draw the target (blue box) and gaze (red ring), given in screen points, on a Claude-size image."""
        if target is not None:
            draw = ImageDraw.Draw(img)
            x0, y0 = self.to_shot(target.x, target.y)
            x1, y1 = self.to_shot(target.x + target.w, target.y + target.h)
            draw.rectangle([x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=(0, 200, 255), width=3)
        if gaze is not None:
            draw = ImageDraw.Draw(img)
            gx, gy = self.to_shot(*gaze)
            for r, width in ((18, 4), (6, 3)):
                draw.ellipse([gx - r, gy - r, gx + r, gy + r], outline=(255, 30, 30), width=width)
        return img

    def zoom(self, region):
        """Crop [x0, y0, x1, y1] (screenshot pixels) at full resolution, fitted to Claude's size."""
        full = self._grab()
        dpr = full.width / self.w_pt
        x0, y0, x1, y1 = region
        box = tuple(round(v / self.scale * dpr) for v in (x0, y0, x1, y1))
        crop = full.crop(box)
        max_w, max_h = config.SCREENSHOT_MAX
        fit = min(max_w / crop.width, max_h / crop.height)
        return crop.resize((max(1, round(crop.width * fit)), max(1, round(crop.height * fit))), Image.LANCZOS)


@dataclass
class Frame:
    """The screen at one moment: Claude's view, and a full-resolution close-up of where you were looking."""
    t: float  # time.time()
    gaze: tuple[float, float]  # screen points, when it was taken
    shot: Image.Image  # Claude's size, unmarked
    crop: Image.Image  # full resolution
    crop_box: tuple[float, float, float, float]  # what `crop` covers, in screen points

    def close_up(self, gaze):
        """The close-up with `gaze` (screen points) ringed, fitted to Claude's size."""
        x0, y0, x1, y1 = self.crop_box
        img = self.crop.copy()
        sx, sy = img.width / (x1 - x0), img.height / (y1 - y0)
        if x0 <= gaze[0] <= x1 and y0 <= gaze[1] <= y1:
            gx, gy = (gaze[0] - x0) * sx, (gaze[1] - y0) * sy
            r, width = 14 * sx, max(2, round(2 * sx))
            ImageDraw.Draw(img).ellipse([gx - r, gy - r, gx + r, gy + r], outline=(255, 30, 30), width=width)
        max_w, max_h = config.SCREENSHOT_MAX
        fit = min(max_w / img.width, max_h / img.height, 1.0)
        return img.resize((max(1, round(img.width * fit)), max(1, round(img.height * fit))), Image.LANCZOS)


class Frames(threading.Thread):
    """The screen a couple of times a second while you speak, so "this" can mean what was on screen then: a video
    (or anything else that moves) has changed by the time the agent takes its first screenshot."""

    def __init__(self, screen: Screen, gaze_now, wanted):
        super().__init__(name="frames", daemon=True)
        self.screen, self.gaze_now, self.wanted = screen, gaze_now, wanted
        self._frames: deque[Frame] = deque(maxlen=FRAMES_KEPT)
        self._lock = threading.Lock()

    def run(self):
        while True:
            t0 = time.time()
            gaze = self.gaze_now()
            if gaze is not None and self.wanted():
                try:
                    frame = self._take(gaze)
                except Exception:  # capture failing (permission, a stalled display) only loses this frame
                    frame = None
                if frame is not None:
                    with self._lock:
                        self._frames.append(frame)
            time.sleep(max(0.05, FRAME_PERIOD_S - (time.time() - t0)))

    def _take(self, gaze) -> Frame:
        t = time.time()
        full = self.screen._capture()  # its own thread: a stuck capture only stalls this loop, not the agent
        dpr = full.width / self.screen.w_pt
        half = FRAME_CROP_PT / 2
        x0 = min(max(0.0, gaze[0] - half), max(0.0, self.screen.w_pt - FRAME_CROP_PT))
        y0 = min(max(0.0, gaze[1] - half), max(0.0, self.screen.h_pt - FRAME_CROP_PT))
        box = (x0, y0, min(self.screen.w_pt, x0 + FRAME_CROP_PT), min(self.screen.h_pt, y0 + FRAME_CROP_PT))
        crop = full.crop(tuple(round(v * dpr) for v in box))
        half = full.reduce(2) if full.width >= 2 * self.screen.size[0] else full  # box filter: cheap and smooth
        return Frame(t, tuple(gaze), half.resize(self.screen.size, Image.BILINEAR), crop, box)  # ~3x faster than LANCZOS

    def during(self, t0, t1, gaze) -> Frame | None:
        """The frame taken while you said it whose gaze was nearest `gaze` (where "this" resolved to); latest wins ties."""
        with self._lock:
            frames = [f for f in self._frames if t0 - FRAME_PERIOD_S <= f.t <= t1 + FRAME_PERIOD_S]
        if not frames or gaze is None:
            return frames[-1] if frames else None
        return min(reversed(frames), key=lambda f: (f.gaze[0] - gaze[0]) ** 2 + (f.gaze[1] - gaze[1]) ** 2)


def image_block(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()},
    }
