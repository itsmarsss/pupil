"""Dashboard feed for mouse gaze + live camera: tracker images and scene, cursor on the Screen view."""
from __future__ import annotations

import json
from typing import Callable

from .feeds import LiveFeed, Message


class MouseHybridFeed(LiveFeed):
    """Live camera feed; gaze on the screen diagram follows the mouse, not the eye model."""

    def __init__(self, tracker, mouse_gaze, context: Callable[[], dict] | None = None,
                 lid: Callable[[], float | None] | None = None):
        self.mouse = mouse_gaze
        super().__init__(tracker, context, lid)

    def poll(self) -> list[Message]:
        out = super().poll()
        patched: list[Message] = []
        for m in out:
            if not isinstance(m, str):
                patched.append(m)
                continue
            msg = json.loads(m)
            if msg.get("type") != "frame":
                patched.append(m)
                continue
            point = self.mouse.point()
            if point is not None:
                x, y = round(point[0], 1), round(point[1], 1)
                msg.setdefault("gaze", {})
                msg["gaze"]["fused"] = [x, y]
                msg["gaze"]["smoothed"] = [x, y]
                msg["gaze"]["fixation"] = None
                msg["gaze_sigma_pt"] = 4
            msg["phase"] = "mouse control"
            patched.append(json.dumps(msg))
        return patched
