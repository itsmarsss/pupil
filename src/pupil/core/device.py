"""Per-machine camera geometry: the camera's focal length and where it sits relative to the screen.

The FaceTime HD camera reports a 0° field of view to AVFoundation and Apple doesn't publish it, so the focal
length is measured once with `pupil fov` (iris size at a known distance). It's stored in pixels at 1080p scale:
our frames are centre crops (same focal length) or downscales of the 1920x1080 capture, so the focal length
for any frame is this value scaled by frame height / 1080.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from .paths import APP_SUPPORT

DEVICE_FILE = APP_SUPPORT / "device.json"
NATIVE_HEIGHT = 1080
# Until measured: the value this project assumed so far (54° across the 1440x1080 crop, ~68° across 1920).
UNMEASURED_FOCAL_1080 = 720 / math.tan(math.radians(27))


@dataclass
class Device:
    focal_px_1080: float = UNMEASURED_FOCAL_1080
    measured: bool = False
    method: str = ""  # how focal_px_1080 was obtained
    # Camera centre relative to the screen, for the geometric model. Default: centred in the notch, measured by
    # the geo prototype on a 14" MacBook Pro as 0.32 cm below the panel's top edge.
    camera_below_top_cm: float = 0.32
    camera_x_offset_cm: float = 0.0

    def focal_px(self, frame_height: int) -> float:
        return self.focal_px_1080 * frame_height / NATIVE_HEIGHT

    def hfov_deg(self, frame_width: int, frame_height: int) -> float:
        return math.degrees(2 * math.atan(frame_width / 2 / self.focal_px(frame_height)))

    def save(self) -> None:
        DEVICE_FILE.parent.mkdir(parents=True, exist_ok=True)
        DEVICE_FILE.write_text(json.dumps(asdict(self), indent=2))


def load_device() -> Device:
    try:
        raw = json.loads(DEVICE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return Device()
    return Device(**{k: v for k, v in raw.items() if k in Device.__dataclass_fields__})
