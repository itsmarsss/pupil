"""A small status pill that trails the user's gaze: what voice control is doing, colour-coded, at a glance.

It sits beside the gaze (never on it, so it doesn't cover what you're looking at), eases toward it so eye jitter
doesn't shake it, flips to the other side near screen edges, and hides when nothing is happening. Kept out of
screenshots, like the HUD, so the agent never sees it.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMakeRect,
    NSScreen,
    NSStatusWindowLevel,
    NSString,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingNone,
    NSWindowStyleMaskBorderless,
)

OFFSET = 34  # points from the gaze to the pill's near corner
EASE = 0.22  # per tick, toward the gaze (smooths eye jitter)
DEADZONE = 6  # points: smaller gaze movements don't move it
HEIGHT = 22
PAD = 9
DOT = 8


def _rgb(r, g, b, a=1.0):
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(r / 255, g / 255, b / 255, a)


# state -> (colour, pulses)
STYLES = {
    "listening": (_rgb(77, 171, 247), True),     # blue: you're talking
    "hearing": (_rgb(116, 192, 252), False),     # light blue: turning speech into text
    "thinking": (_rgb(255, 196, 61), True),      # amber: Jev deciding
    "writing": (_rgb(255, 196, 61), False),      # amber: formatting dictation
    "agent": (_rgb(177, 151, 252), True),        # purple: the agent thinking
    "acting": (_rgb(61, 220, 132), False),       # green: the agent or Jev doing it
    "done": (_rgb(61, 220, 132), False),         # green: done (brief)
    "waiting": (_rgb(255, 146, 43), True),       # orange: needs your answer
    "speaking": (_rgb(32, 201, 151), False),     # teal: reading a reply aloud
    "dictating": (_rgb(250, 82, 82), True),      # red: dictation mode
    "error": (_rgb(250, 82, 82), False),         # red: something went wrong (brief)
}
BACKGROUND = NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.82)
TEXT = NSColor.colorWithCalibratedWhite_alpha_(0.95, 1.0)
FONT = NSFont.systemFontOfSize_weight_(12, 0.3)


@dataclass(frozen=True)
class Status:
    state: str  # a STYLES key
    label: str


class _PillView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(rect)
        status = self.status
        if status is None:
            return
        color, pulses = STYLES.get(status.state, STYLES["thinking"])
        b = self.bounds()
        BACKGROUND.set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(b, HEIGHT / 2, HEIGHT / 2).fill()
        alpha = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.time() * 6)) if pulses else 1.0
        color.colorWithAlphaComponent_(alpha).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(PAD, (HEIGHT - DOT) / 2, DOT, DOT)).fill()
        attrs = {NSFontAttributeName: FONT, NSForegroundColorAttributeName: TEXT}
        text = NSString.stringWithString_(status.label)
        size = text.sizeWithAttributes_(attrs)
        text.drawAtPoint_withAttributes_((PAD + DOT + 6, (HEIGHT - size.height) / 2), attrs)


class Indicator:
    def __init__(self):
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 120, HEIGHT), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)
        win.setSharingType_(NSWindowSharingNone)
        win.setLevel_(NSStatusWindowLevel + 1)
        win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                                   | NSWindowCollectionBehaviorFullScreenAuxiliary
                                   | NSWindowCollectionBehaviorIgnoresCycle)
        view = _PillView.alloc().initWithFrame_(NSMakeRect(0, 0, 120, HEIGHT))
        view.status = None
        win.setContentView_(view)
        self._win, self._view = win, view
        frame = NSScreen.mainScreen().frame()
        self._screen_w, self._screen_h = frame.size.width, frame.size.height
        self._pos = None  # eased position of the gaze it follows (top-left points)
        self._shown: Status | None = None

    def update(self, status: Status | None, gaze) -> None:
        """Called every tick (~30 Hz) with what to show and where the user is looking (None: keep it put)."""
        if status is None:
            if self._win.isVisible():
                self._win.orderOut_(None)
            self._shown = None
            return
        if status != self._shown:
            self._shown = status
            self._view.status = status
            width = NSString.stringWithString_(status.label).sizeWithAttributes_({NSFontAttributeName: FONT}).width
            w = math.ceil(PAD + DOT + 6 + width + PAD)
            self._win.setContentSize_((w, HEIGHT))
            self._view.setFrame_(NSMakeRect(0, 0, w, HEIGHT))
            self._view.setNeedsDisplay_(True)
        elif STYLES.get(status.state, (None, False))[1]:
            self._view.setNeedsDisplay_(True)  # the pulse
        if gaze is not None:
            if self._pos is None:
                self._pos = gaze
            elif math.dist(self._pos, gaze) > DEADZONE:
                self._pos = (self._pos[0] + (gaze[0] - self._pos[0]) * EASE,
                             self._pos[1] + (gaze[1] - self._pos[1]) * EASE)
        if self._pos is None:
            return
        w = self._win.frame().size.width
        x, y = self._pos
        left = x + OFFSET if x + OFFSET + w < self._screen_w else x - OFFSET - w  # flip near the right edge
        top = y + OFFSET if y + OFFSET + HEIGHT < self._screen_h else y - OFFSET - HEIGHT  # and the bottom
        self._win.setFrameOrigin_((left, self._screen_h - top - HEIGHT))
        if not self._win.isVisible():
            self._win.orderFrontRegardless()
