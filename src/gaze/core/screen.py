"""Main-display geometry and conversions between points, millimetres and degrees."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Screen:
    width_pt: float
    height_pt: float
    width_mm: float
    height_mm: float

    @property
    def mm_per_pt(self) -> float:
        return self.width_mm / self.width_pt

    def pt_to_deg(self, dist_pt: float, eye_distance_mm: float) -> float:
        """Visual angle of an on-screen distance seen from `eye_distance_mm` away."""
        return math.degrees(2 * math.atan(dist_pt * self.mm_per_pt / (2 * eye_distance_mm)))

    def deg_to_pt(self, deg: float, eye_distance_mm: float) -> float:
        return 2 * eye_distance_mm * math.tan(math.radians(deg) / 2) / self.mm_per_pt

    def to_cocoa_y(self, y_top_left: float) -> float:
        """Top-left screen y → Cocoa's bottom-left y."""
        return self.height_pt - y_top_left

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return min(max(x, 0.0), self.width_pt), min(max(y, 0.0), self.height_pt)


def main_screen() -> Screen:
    """Query the main display. Falls back to a 14" MacBook Pro if Quartz reports nothing."""
    from AppKit import NSScreen
    from Quartz import CGDisplayScreenSize, CGMainDisplayID

    frame = NSScreen.mainScreen().frame()
    w_pt, h_pt = float(frame.size.width), float(frame.size.height)
    size_mm = CGDisplayScreenSize(CGMainDisplayID())
    w_mm, h_mm = float(size_mm.width), float(size_mm.height)
    if w_mm <= 0 or h_mm <= 0:
        w_mm, h_mm = 302.0, 196.0
    return Screen(w_pt, h_pt, w_mm, h_mm)
