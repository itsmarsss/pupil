"""Snap gaze to nearby clickable UI elements using the macOS Accessibility API.

Webcam gaze is ~1–2° off; buttons and links are usually further apart than that, so
during a fixation we pick the clearly-closest actionable element. Raw gaze is never
changed; the snapped point is a separate output.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

log = logging.getLogger(__name__)

ACTIONABLE = {
    "AXButton", "AXLink", "AXCheckBox", "AXRadioButton", "AXMenuItem", "AXMenuBarItem",
    "AXPopUpButton", "AXMenuButton", "AXTextField", "AXTextArea", "AXComboBox", "AXTab",
    "AXDisclosureTriangle", "AXIncrementor", "AXSlider", "AXCell", "AXImage",
}
MAX_ELEMENT_AREA = 400 * 200  # ignore huge "actionable" containers


@dataclass(frozen=True)
class Element:
    role: str
    x: float
    y: float
    w: float
    h: float
    title: str = ""

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def distance(self, px: float, py: float) -> float:
        """0 inside the rect, else distance to its edge."""
        dx = max(self.x - px, 0, px - (self.x + self.w))
        dy = max(self.y - py, 0, py - (self.y + self.h))
        return math.hypot(dx, dy)


def choose(elements: list[Element], gaze: tuple[float, float], radius_pt: float, margin: float = 1.5) -> Element | None:
    """The element to snap to: within `radius_pt` of gaze, and either containing the gaze
    point or `margin` times closer than the runner-up. Otherwise None (ambiguous)."""
    near = sorted(
        (e for e in set(elements) if e.w * e.h <= MAX_ELEMENT_AREA and e.distance(*gaze) <= radius_pt),
        key=lambda e: (e.distance(*gaze), math.dist(e.centre, gaze)),
    )
    if not near:
        return None
    if len(near) == 1:
        return near[0]
    best, second = near[0], near[1]
    d1, d2 = best.distance(*gaze), second.distance(*gaze)
    if d1 == 0 and d2 > 0:
        return best
    if d1 == 0 and d2 == 0:  # nested or overlapping: prefer the smaller
        return min(near[:2], key=lambda e: e.w * e.h)
    return best if d2 >= margin * d1 else None


def probe_points(x: float, y: float, radius: float) -> list[tuple[float, float]]:
    """Centre plus two rings of 8 points, where we ask what element is under each."""
    pts = [(x, y)]
    for r in (radius * 0.5, radius):
        for k in range(8):
            a = k * math.pi / 4
            pts.append((x + r * math.cos(a), y + r * math.sin(a)))
    return pts


class Snapper:
    """Queries elements around a point. Needs Accessibility permission for the app
    running Python (e.g. Terminal); `available` is False until it's granted."""

    def __init__(self):
        import ApplicationServices as AS

        self._AS = AS
        self._system = AS.AXUIElementCreateSystemWide()

    @property
    def available(self) -> bool:
        return bool(self._AS.AXIsProcessTrusted())

    def request_permission(self) -> bool:
        """Shows the system prompt to grant Accessibility access."""
        AS = self._AS
        return bool(AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True}))

    def snap(self, gaze: tuple[float, float], radius_pt: float) -> Element | None:
        if not self.available:
            return None
        found = []
        for px, py in probe_points(*gaze, radius_pt):
            el = self._element_at(px, py)
            if el is not None:
                found.append(el)
        return choose(found, gaze, radius_pt)

    def _attr(self, el, name):
        err, value = self._AS.AXUIElementCopyAttributeValue(el, name, None)
        return value if err == 0 else None

    def _element_at(self, x: float, y: float) -> Element | None:
        AS = self._AS
        err, el = AS.AXUIElementCopyElementAtPosition(self._system, float(x), float(y), None)
        # Climb a few levels: the hit element is often a label inside the button.
        for _ in range(4):
            if err != 0 or el is None:
                return None
            role = self._attr(el, "AXRole")
            if role in ACTIONABLE:
                return self._to_element(el, str(role))
            el = self._attr(el, "AXParent")
            err = 0
        return None

    def _to_element(self, el, role: str) -> Element | None:
        AS = self._AS
        pos, size = self._attr(el, "AXPosition"), self._attr(el, "AXSize")
        if pos is None or size is None:
            return None
        ok_p, p = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        ok_s, s = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
        if not (ok_p and ok_s):
            return None
        title = self._attr(el, "AXTitle") or self._attr(el, "AXDescription") or ""
        return Element(role, float(p.x), float(p.y), float(s.width), float(s.height), str(title))
