"""Screenshots for Claude: downscaled to the recommended size, with gaze annotations; coordinate conversion."""

import base64
import io

import threading

import Quartz
from PIL import Image, ImageDraw

from . import config

CAPTURE_TIMEOUT_S = 3.0


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
        img = self._grab().resize(self.size, Image.LANCZOS)
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


def image_block(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()},
    }
