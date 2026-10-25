"""Dashboard feed for `pupil control --gaze mouse`: mouse position and voice state, no camera."""
from __future__ import annotations

import json
import time
from typing import Callable

from ..core.device import load_device
from ..core.screen import main_screen
from ..geometry.eyeball import ScreenPlane
from ..telemetry import SCHEMA, scene_info


class MouseControlFeed:
    def __init__(self, gaze, context: Callable[[], dict] | None = None):
        self.gaze = gaze
        self.context = context or (lambda: {})
        self._started = time.time()
        self._last_json = ""
        screen = main_screen()
        device = load_device()
        self._plane = ScreenPlane(screen, device.camera_below_top_cm, device.camera_x_offset_cm)

    def hello(self) -> list[str]:
        screen = main_screen()
        device = load_device()
        calib = {"type": "calibration", "schema": SCHEMA, "calibrated": False}
        out = [json.dumps(scene_info(screen, self._plane, device, None)), json.dumps(calib)]
        frame = self._frame_payload()
        if frame is not None:
            out.append(json.dumps(frame))
        return out

    def _frame_payload(self) -> dict | None:
        point = self.gaze.point()
        if point is None:
            return None
        x, y = round(point[0], 1), round(point[1], 1)
        return {
            "type": "frame",
            "schema": SCHEMA,
            "t": round(time.time() - self._started, 4),
            "latency_ms": 0,
            "fps": 30,
            "state": "tracking",
            "confidence": 1.0,
            "gaze": {"fused": [x, y], "smoothed": [x, y], "fixation": None},
            "gaze_sigma_pt": 4,
            "phase": "mouse control",
            "camera_note": "Mouse mode uses the cursor for gaze; there is no webcam feed.",
            **self.context(),
        }

    def poll(self) -> list[str]:
        state = self._frame_payload()
        if state is None:
            return []
        payload = json.dumps(state)
        if payload == self._last_json:
            return []
        self._last_json = payload
        return [payload]

    def command(self, name: str, value) -> bool:
        return False
