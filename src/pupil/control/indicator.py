"""A small status pill that trails the user's gaze: what voice control is doing, colour-coded, at a glance, with
what the agent is saying running faintly underneath.

It sits centred just under the gaze (never on it, so it doesn't cover what you're looking at), eases toward it so eye jitter
doesn't shake it, moves above it near the bottom of the screen, and hides when nothing is happening. Kept out of
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
    NSGraphicsContext,
    NSMakeRect,
    NSScreen,
    NSShadow,
    NSShadowAttributeName,
    NSStatusWindowLevel,
    NSString,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)

from ..core.sharing import overlay_sharing

OFFSET = 30  # points below the gaze to the pill (centred under it)
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
CAPTION = NSColor.colorWithCalibratedWhite_alpha_(1.0, 1.0)
CAPTION_FONT = NSFont.systemFontOfSize_weight_(13, 0.2)
CAPTION_W = 200  # caption line width, points; longer speech scrolls
CAPTION_H = 18
CAPTION_GAP = 4
CAPTION_SPEED = 45  # points per second, once a long caption starts scrolling
CAPTION_HOLD_S = 0.8  # read the start before it scrolls
CAPTION_FADE_S = 0.3  # once it's said, it goes
_SHADOW = NSShadow.alloc().init()
_SHADOW.setShadowBlurRadius_(3)
_SHADOW.setShadowOffset_((0, -1))
_SHADOW.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.9))


@dataclass(frozen=True)
class Status:
    state: str  # a STYLES key
    label: str


class _PillView(NSView):
    def drawRect_(self, rect):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(rect)
        b = self.bounds()
        status = self.status
        if status is not None:
            color, pulses = STYLES.get(status.state, STYLES["thinking"])
            pill = NSMakeRect((b.size.width - self.pill_w) / 2, b.size.height - HEIGHT, self.pill_w, HEIGHT)
            BACKGROUND.set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(pill, HEIGHT / 2, HEIGHT / 2).fill()
            alpha = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.time() * 6)) if pulses else 1.0
            color.colorWithAlphaComponent_(alpha).set()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(pill.origin.x + PAD, pill.origin.y + (HEIGHT - DOT) / 2, DOT, DOT)).fill()
            attrs = {NSFontAttributeName: FONT, NSForegroundColorAttributeName: TEXT}
            text = NSString.stringWithString_(status.label)
            size = text.sizeWithAttributes_(attrs)
            text.drawAtPoint_withAttributes_((pill.origin.x + PAD + DOT + 6, pill.origin.y + (HEIGHT - size.height) / 2), attrs)
        caption = self.caption
        if caption:
            text, offset, alpha = caption
            attrs = {NSFontAttributeName: CAPTION_FONT,
                     NSForegroundColorAttributeName: CAPTION.colorWithAlphaComponent_(0.8 * alpha),
                     NSShadowAttributeName: _SHADOW}
            line = NSString.stringWithString_(text)
            width = line.sizeWithAttributes_(attrs).width
            NSGraphicsContext.saveGraphicsState()
            NSBezierPath.clipRect_(NSMakeRect(0, 0, b.size.width, CAPTION_H))
            x = (b.size.width - width) / 2 if width <= b.size.width else -offset  # long: scrolls left
            line.drawAtPoint_withAttributes_((x, 2), attrs)
            NSGraphicsContext.restoreGraphicsState()


class Indicator:
    """The pill sits centred just under the gaze; while the agent speaks, what it says runs underneath in faint
    text (scrolling when it's longer than fits), then fades."""

    def __init__(self):
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, CAPTION_W, HEIGHT), NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)
        win.setSharingType_(overlay_sharing())
        win.setLevel_(NSStatusWindowLevel + 1)
        win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                                   | NSWindowCollectionBehaviorFullScreenAuxiliary
                                   | NSWindowCollectionBehaviorIgnoresCycle)
        view = _PillView.alloc().initWithFrame_(NSMakeRect(0, 0, CAPTION_W, HEIGHT))
        view.status, view.caption, view.pill_w = None, None, 0
        win.setContentView_(view)
        self._win, self._view = win, view
        frame = NSScreen.mainScreen().frame()
        self._screen_w, self._screen_h = frame.size.width, frame.size.height
        self._pos = None  # eased position of the gaze it follows (top-left points)
        self._shown: Status | None = None
        self._caption: str | None = None
        self._caption_start = 0.0
        self._caption_end: float | None = None  # when speech stopped (then it fades)

    def say(self, text: str) -> None:
        """Show what's being said, under the pill."""
        self._caption, self._caption_start, self._caption_end = " ".join(text.split()), time.time(), None

    def clear(self) -> None:
        """Remove the caption now: it's no longer relevant (the user is talking, or something new started)."""
        self._caption, self._caption_end = None, None

    def said(self) -> None:
        """Speech ended (or was cut off): fade the caption."""
        if self._caption and self._caption_end is None:
            self._caption_end = time.time()

    def _caption_now(self):
        if not self._caption:
            return None
        now = time.time()
        alpha = 1.0
        if self._caption_end is not None:
            alpha = 1 - (now - self._caption_end) / CAPTION_FADE_S
            if alpha <= 0:
                self._caption = None
                return None
        width = NSString.stringWithString_(self._caption).sizeWithAttributes_({NSFontAttributeName: CAPTION_FONT}).width
        offset = max(0.0, min((now - self._caption_start - CAPTION_HOLD_S) * CAPTION_SPEED, width - CAPTION_W + 20))
        return self._caption, offset, alpha

    def update(self, status: Status | None, gaze) -> None:
        """Called every tick (~30 Hz) with what to show and where the user is looking (None: keep it put)."""
        caption = self._caption_now()
        if status is None and caption is None:
            if self._win.isVisible():
                self._win.orderOut_(None)
            self._shown = None
            return
        if status != self._shown:
            self._shown = status
            width = 0 if status is None else \
                NSString.stringWithString_(status.label).sizeWithAttributes_({NSFontAttributeName: FONT}).width
            self._view.pill_w = 0 if status is None else math.ceil(PAD + DOT + 6 + width + PAD)
        self._view.status = status
        self._view.caption = caption
        w = max(self._view.pill_w, CAPTION_W if caption else 0)
        h = HEIGHT + (CAPTION_GAP + CAPTION_H if caption else 0)
        if (w, h) != (self._win.frame().size.width, self._win.frame().size.height):
            self._win.setContentSize_((w, h))
            self._view.setFrame_(NSMakeRect(0, 0, w, h))
        self._view.setNeedsDisplay_(True)  # cheap: a small window (pulse, scrolling caption, fade)
        if gaze is not None:
            if self._pos is None:
                self._pos = gaze
            elif math.dist(self._pos, gaze) > DEADZONE:
                self._pos = (self._pos[0] + (gaze[0] - self._pos[0]) * EASE,
                             self._pos[1] + (gaze[1] - self._pos[1]) * EASE)
        if self._pos is None:
            return
        x, y = self._pos
        left = min(max(x - w / 2, 4), self._screen_w - w - 4)  # centred under the gaze, kept on screen
        top = y + OFFSET if y + OFFSET + h < self._screen_h else y - OFFSET - h  # above it near the bottom
        self._win.setFrameOrigin_((left, self._screen_h - top - h))
        if not self._win.isVisible():
            self._win.orderFrontRegardless()
