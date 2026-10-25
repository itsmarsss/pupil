"""Click-through gaze ring (or heat map) that floats above every app, Space and fullscreen window.

The heat map shows gaze as what it is: a blur about as wide as the tracker's measured accuracy (wider when it's
less sure), the same Gaussian the voice controller uses to decide what "that" means (control/heat.py)."""
from __future__ import annotations

import math

from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSMakeRect,
    NSRectFill,
    NSScreenSaverWindowLevel,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingNone,
    NSWindowStyleMaskBorderless,
)

from ..core.screen import Screen

RING = 56  # diameter, points
LINE = 4.5
HEAT_LEVELS = 20  # stacked discs approximating the Gaussian (fewer bands visibly)
HEAT_ALPHA = 0.03  # per disc: ~0.45 at the centre
MAX_HEAT = 900  # window size cap, points


class _RingView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSRectFill(rect)
        if self.heat:
            # Disc i has the radius where the Gaussian falls to i/(N+1) of its peak, so the stacked alpha at any
            # radius follows exp(-r^2 / 2 sigma^2).
            size = self.bounds().size.width
            sigma = size / 4
            self.color.colorWithAlphaComponent_(HEAT_ALPHA).set()
            for i in range(1, HEAT_LEVELS + 1):
                r = min(sigma * math.sqrt(-2 * math.log(i / (HEAT_LEVELS + 1))), size / 2)
                NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(size / 2 - r, size / 2 - r, 2 * r, 2 * r)).fill()
            return
        inset = LINE / 2 + 1
        path = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(inset, inset, RING - 2 * inset, RING - 2 * inset))
        path.setLineWidth_(LINE)
        self.color.set()
        path.stroke()


class GazeRing:
    """Moves a small transparent window rather than redrawing a fullscreen one."""

    TRACKING = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.85, 0.35, 0.85)
    FIXATING = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.6, 1.0, 0.9)
    UNSURE = NSColor.colorWithCalibratedWhite_alpha_(0.6, 0.45)  # head outside its calibrated range

    def __init__(self, screen: Screen):
        self.screen = screen
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, RING, RING), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)
        win.setSharingType_(NSWindowSharingNone)  # not in screenshots: the voice agent marks gaze itself
        win.setLevel_(NSScreenSaverWindowLevel)
        win.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        view = _RingView.alloc().initWithFrame_(NSMakeRect(0, 0, RING, RING))
        view.color = self.TRACKING
        view.heat = False
        win.setContentView_(view)
        self._win, self._view = win, view
        self._style = "tracking"
        self._size = RING
        self.heat = False  # draw a heat map instead of a ring
        self.sigma = 60.0  # heat map width (points), for a fully confident point

    def show_at(self, x: float, y: float, fixating: bool = False, confidence: float = 1.0) -> None:
        x, y = self.screen.clamp(x, y)
        size = RING
        if self.heat:  # 4 sigma across, wider when less sure; resized in steps so it isn't redrawn every frame
            size = min(4 * self.sigma / math.sqrt(max(confidence, 0.25)), MAX_HEAT) // 8 * 8 + 8
        if size != self._size or self.heat != self._view.heat:
            self._size = size
            self._view.heat = self.heat
            self._win.setContentSize_((size, size))
            self._view.setFrame_(NSMakeRect(0, 0, size, size))
            self._view.setNeedsDisplay_(True)
        self._win.setFrameOrigin_((x - size / 2, self.screen.to_cocoa_y(y) - size / 2))
        style = "unsure" if confidence < 0.5 else "fixating" if fixating else "tracking"
        if style != self._style:
            self._style = style
            self._view.color = {"unsure": self.UNSURE, "fixating": self.FIXATING, "tracking": self.TRACKING}[style]
            self._view.setNeedsDisplay_(True)
        if not self._win.isVisible():
            self._win.orderFrontRegardless()

    def hide(self) -> None:
        if self._win.isVisible():
            self._win.orderOut_(None)
