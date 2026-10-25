"""Voice-control overlay (AppKit): snapped target and nearby candidates, "which one?" badges, click flash.

A transparent, click-through window over everything, kept out of screenshots so the agent never sees it.
Redrawn only when something changes. The status bar it used to show at the bottom (mic, heard, status) is the
notch's job now (ui/notch.py); PUPIL_HUD_BAR=1 brings it back for debugging.
"""
from __future__ import annotations

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
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)

from ..core.env import env
from ..core.sharing import overlay_sharing

TARGET = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.78, 1.0, 0.86)
CANDIDATE = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.78, 1.0, 0.28)
FLASH = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.31, 1.0, 0.47, 0.9)
GAZE = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.24, 0.24, 0.63)
PILL = NSColor.colorWithCalibratedRed_green_blue_alpha_(0.08, 0.08, 0.09, 0.8)
TEXT = NSColor.colorWithCalibratedWhite_alpha_(0.94, 1.0)
BADGE = NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 0.62, 0.1, 0.95)
SHOW_BAR = env("HUD_BAR") == "1"  # the old bottom status bar, for debugging


class _HudView(NSView):
    def isFlipped(self):  # top-left origin: screen points, like the rest of the app
        return True

    def drawRect_(self, rect):
        hud = self.hud
        for c in hud.candidates:
            if c != hud.target:
                CANDIDATE.set()
                p = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(c.x, c.y, c.w, c.h), 4, 4)
                p.setLineWidth_(1)
                p.stroke()
        for i, b in enumerate(hud.badges):  # "which one?": outline + number on each option
            BADGE.set()
            box = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(b.x - 2, b.y - 2, b.w + 4, b.h + 4), 6, 6)
            box.setLineWidth_(3)
            box.stroke()
            dot = NSMakeRect(b.x - 12, b.y - 12, 24, 24)
            NSBezierPath.bezierPathWithOvalInRect_(dot).fill()
            label = NSString.stringWithString_(str(i + 1))
            attrs = {NSFontAttributeName: NSFont.boldSystemFontOfSize_(15), NSForegroundColorAttributeName: TEXT}
            size = label.sizeWithAttributes_(attrs)
            label.drawAtPoint_withAttributes_((dot.origin.x + (24 - size.width) / 2, dot.origin.y + (24 - size.height) / 2), attrs)
        if hud.target is not None:
            t = hud.target
            TARGET.set()
            p = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(t.x - 2, t.y - 2, t.w + 4, t.h + 4), 6, 6)
            p.setLineWidth_(3)
            p.stroke()
        for point, color, radius in ((hud.gaze, GAZE, 14), (hud.flash, FLASH, 22)):
            if point is not None and (point is not hud.gaze or hud.draw_gaze):
                color.set()
                p = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(point[0] - radius, point[1] - radius, 2 * radius, 2 * radius))
                p.setLineWidth_(3)
                p.stroke()
        text = hud.line() if SHOW_BAR else ""
        if text:
            attrs = {NSFontAttributeName: NSFont.systemFontOfSize_(14), NSForegroundColorAttributeName: TEXT}
            s = NSString.stringWithString_(text)
            size = s.sizeWithAttributes_(attrs)
            w, h = size.width + 32, 36
            bounds = self.bounds().size
            box = NSMakeRect((bounds.width - w) / 2, bounds.height - 72, w, h)
            PILL.set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(box, 18, 18).fill()
            s.drawAtPoint_withAttributes_((box.origin.x + 16, box.origin.y + (h - size.height) / 2), attrs)


class Hud:
    def __init__(self):
        frame = NSScreen.mainScreen().frame()
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setHasShadow_(False)
        win.setIgnoresMouseEvents_(True)
        win.setLevel_(NSStatusWindowLevel)
        win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                                   | NSWindowCollectionBehaviorFullScreenAuxiliary)
        win.setSharingType_(overlay_sharing())  # recordings see it; the agent's screenshots leave it out (screen.py)
        view = _HudView.alloc().initWithFrame_(NSMakeRect(0, 0, frame.size.width, frame.size.height))
        view.hud = self
        win.setContentView_(view)
        self._win, self._view = win, view
        self.gaze = None
        self.draw_gaze = False  # only in mouse mode; the tracker draws its own ring
        self.target = None
        self.candidates: list = []
        self.flash = None
        self.status = self.heard = self.warning = ""
        self.mic, self.mic_device = "off", ""
        self.gate = ""  # lip gate state, when it's on
        self.speaking = False  # replies being read aloud
        self.badges: list = []  # "which one?": numbered elements to choose between
        self._drawn = None

    def show(self) -> None:
        self._win.orderFrontRegardless()

    def hide(self) -> None:
        self._win.orderOut_(None)

    def line(self) -> str:
        mic = f"mic: {self.mic_device} · {self.mic}" if self.mic_device else f"mic: {self.mic}"
        if self.gate:
            mic += f" · {self.gate}"
        parts = [self.warning, "🔊 speaking (talk to interrupt)" if self.speaking else mic, f'heard: "{self.heard}"' if self.heard else "", self.status]
        return "   |   ".join(p for p in parts if p)

    def update(self) -> None:
        state = (self.line() if SHOW_BAR else "", self.target, tuple(self.candidates), self.flash, self.gaze if self.draw_gaze else None,
                 tuple(self.badges))
        if state != self._drawn:
            self._drawn = state
            self._view.setNeedsDisplay_(True)
