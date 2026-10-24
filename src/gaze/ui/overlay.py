"""Click-through gaze ring that floats above every app, Space and fullscreen window."""
from __future__ import annotations

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
    NSWindowStyleMaskBorderless,
)

from ..core.screen import Screen

RING = 56  # diameter, points
LINE = 4.5


class _RingView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSRectFill(rect)
        inset = LINE / 2 + 1
        path = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(inset, inset, RING - 2 * inset, RING - 2 * inset))
        path.setLineWidth_(LINE)
        self.color.set()
        path.stroke()


class GazeRing:
    """Moves a small transparent window rather than redrawing a fullscreen one."""

    TRACKING = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.15, 0.85, 0.35, 0.85)
    FIXATING = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.2, 0.6, 1.0, 0.9)

    def __init__(self, screen: Screen):
        self.screen = screen
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, RING, RING), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)
        win.setLevel_(NSScreenSaverWindowLevel)
        win.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorIgnoresCycle
        )
        view = _RingView.alloc().initWithFrame_(NSMakeRect(0, 0, RING, RING))
        view.color = self.TRACKING
        win.setContentView_(view)
        self._win, self._view = win, view
        self._fixating = False

    def show_at(self, x: float, y: float, fixating: bool = False) -> None:
        x, y = self.screen.clamp(x, y)
        self._win.setFrameOrigin_((x - RING / 2, self.screen.to_cocoa_y(y) - RING / 2))
        if fixating != self._fixating:
            self._fixating = fixating
            self._view.color = self.FIXATING if fixating else self.TRACKING
            self._view.setNeedsDisplay_(True)
        if not self._win.isVisible():
            self._win.orderFrontRegardless()

    def hide(self) -> None:
        if self._win.isVisible():
            self._win.orderOut_(None)
