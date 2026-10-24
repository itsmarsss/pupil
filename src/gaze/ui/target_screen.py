"""Fullscreen window for instructions and calibration/validation dots."""
from __future__ import annotations

import math
import time
from typing import Callable

from AppKit import (
    NSApp,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMakeRect,
    NSRectFill,
    NSScreenSaverWindowLevel,
    NSString,
    NSTimer,
    NSView,
    NSWindow,
    NSWindowStyleMaskBorderless,
)

from ..calibration.targets import TargetTiming
from ..core.screen import Screen

BG = NSColor.colorWithCalibratedWhite_alpha_(0.16, 1.0)
DOT = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.36, 0.25, 1.0)
DOT_CENTRE = NSColor.colorWithCalibratedWhite_alpha_(0.05, 1.0)
TEXT = NSColor.colorWithCalibratedWhite_alpha_(0.92, 1.0)
DIM = NSColor.colorWithCalibratedWhite_alpha_(0.6, 1.0)

ESC, SPACE, RETURN = "\x1b", " ", "\r"


class _KeyWindow(NSWindow):
    def canBecomeKeyWindow(self):
        return True


class _TargetView(NSView):
    def isFlipped(self):  # top-left origin, same as our screen coordinates
        return True

    def acceptsFirstResponder(self):
        return True

    def keyDown_(self, event):
        handler = getattr(self, "on_key", None)
        if handler is not None:
            handler(str(event.characters()))

    def drawRect_(self, rect):
        BG.set()
        NSRectFill(rect)
        dot = getattr(self, "dot", None)
        if dot is not None:
            x, y, r = dot
            DOT.set()
            NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - r, y - r, 2 * r, 2 * r)).fill()
            DOT_CENTRE.set()
            NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x - 2, y - 2, 4, 4)).fill()
        lines = getattr(self, "lines", None) or []
        size = self.bounds().size
        y = size.height * 0.42 - 26 * len(lines) / 2
        for i, line in enumerate(lines):
            font = NSFont.systemFontOfSize_weight_(26 if i == 0 else 17, 0.3 if i == 0 else 0.0)
            attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: TEXT if i == 0 else DIM}
            s = NSString.stringWithString_(line)
            w = s.sizeWithAttributes_(attrs).width
            s.drawAtPoint_withAttributes_((size.width / 2 - w / 2, y), attrs)
            y += 44 if i == 0 else 28


class TargetScreen:
    def __init__(self, screen: Screen):
        self.screen = screen
        frame = NSMakeRect(0, 0, screen.width_pt, screen.height_pt)
        win = _KeyWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        win.setLevel_(NSScreenSaverWindowLevel - 1)
        win.setOpaque_(True)
        view = _TargetView.alloc().initWithFrame_(frame)
        view.dot, view.lines, view.on_key = None, [], None
        win.setContentView_(view)
        self._win, self._view = win, view

    @property
    def visible(self) -> bool:
        return bool(self._win.isVisible())

    def show(self, lines: list[str] | None = None, on_key: Callable[[str], None] | None = None) -> None:
        self._view.lines = lines or []
        self._view.dot = None
        self._view.on_key = on_key
        self._view.setNeedsDisplay_(True)
        if not self._win.isVisible():
            NSApp.activateIgnoringOtherApps_(True)
            self._win.makeKeyAndOrderFront_(None)
            self._win.makeFirstResponder_(self._view)

    def set_lines(self, lines: list[str]) -> None:
        self._view.lines = lines
        self._view.setNeedsDisplay_(True)

    def set_dot(self, x: float | None, y: float = 0, r: float = 0) -> None:
        self._view.dot = None if x is None else (x, y, r)
        self._view.setNeedsDisplay_(True)

    def set_on_key(self, on_key: Callable[[str], None] | None) -> None:
        self._view.on_key = on_key

    def hide(self) -> None:
        self._view.on_key = None
        self._win.orderOut_(None)


class TargetSequence:
    """Shows each target for `timing.dwell_s` with a shrinking dot, calling
    `on_target(target, onset_ns, end_ns)` as each finishes and `on_done(cancelled)` at the end.
    Esc cancels."""

    R_START, R_END = 22.0, 7.0

    def __init__(self, ui: TargetScreen, targets, timing: TargetTiming, on_target, on_done):
        self.ui, self.targets, self.timing = ui, list(targets), timing
        self.on_target, self.on_done = on_target, on_done
        self._i = -1
        self._onset_ns = 0
        self._timer = None
        self._finished = False

    def start(self) -> None:
        self.ui.show([], on_key=self._key)
        self._next()
        self._timer = NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 60, True, self._tick)

    def _next(self) -> None:
        self._i += 1
        if self._i >= len(self.targets):
            self._finish(cancelled=False)
            return
        x, y = self.targets[self._i]
        self.ui.set_dot(x, y, self.R_START)
        self._onset_ns = time.monotonic_ns()

    def _tick(self, _timer) -> None:
        if self._finished:
            return
        elapsed = (time.monotonic_ns() - self._onset_ns) / 1e9
        if elapsed >= self.timing.dwell_s:
            self.on_target(self.targets[self._i], self._onset_ns, time.monotonic_ns())
            self._next()
            return
        # ease the dot down to a point so the eye settles on its centre
        k = min(elapsed / (self.timing.dwell_s * 0.6), 1.0)
        r = self.R_END + (self.R_START - self.R_END) * (1 - math.sin(k * math.pi / 2))
        x, y = self.targets[self._i]
        self.ui.set_dot(x, y, r)

    def _key(self, chars: str) -> None:
        if chars == ESC:
            self._finish(cancelled=True)

    def _finish(self, cancelled: bool) -> None:
        if self._finished:
            return
        self._finished = True
        if self._timer is not None:
            self._timer.invalidate()
            self._timer = None
        self.ui.set_dot(None)
        self.on_done(cancelled)
