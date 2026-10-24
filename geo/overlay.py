"""Gaze dot drawn over everything on screen: a tiny borderless, click-through, always-on-top AppKit window
(just the dot) that moves to the gaze point. Tk's transparent backgrounds render black on macOS, so this
doesn't use Tk for drawing; it relies on the Tk event loop (already pumped by the app) to run AppKit.
"""
from __future__ import annotations

import AppKit
import Quartz


class Overlay:
    def __init__(self, display_info: dict, radius: int = 14, rgba=(0.0, 0.86, 0.86, 0.85)):
        self.r = radius
        self.safe_top = display_info["safe_area_top_pt"]
        self.screen_h = display_info["frame_pt"][1]
        d = 2 * radius
        self.win = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            AppKit.NSMakeRect(-100, -100, d, d), AppKit.NSWindowStyleMaskBorderless, AppKit.NSBackingStoreBuffered, False)
        self.win.setOpaque_(False)
        self.win.setBackgroundColor_(AppKit.NSColor.clearColor())
        self.win.setHasShadow_(False)
        self.win.setIgnoresMouseEvents_(True)
        self.win.setLevel_(AppKit.NSStatusWindowLevel)
        self.win.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                                        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
                                        | AppKit.NSWindowCollectionBehaviorStationary)
        self.win.setReleasedWhenClosed_(False)
        view = self.win.contentView()
        view.setWantsLayer_(True)
        layer = view.layer()
        layer.setCornerRadius_(radius)
        layer.setBackgroundColor_(Quartz.CGColorCreateGenericRGB(*rgba))
        self.visible = False

    def update(self, usable_px):
        """usable_px: gaze point in the usable (below-the-notch) screen area, in points, or None to hide the dot."""
        if usable_px is None:
            if self.visible:
                self.win.orderOut_(None)
                self.visible = False
            return
        # AppKit screen coordinates: origin bottom-left of the main screen; the usable area starts below the notch
        x = usable_px[0] - self.r
        y = self.screen_h - (usable_px[1] + self.safe_top) - self.r
        self.win.setFrameOrigin_(AppKit.NSMakePoint(x, y))
        if not self.visible:
            self.win.orderFrontRegardless()
            self.visible = True

    def hide(self):
        self.win.orderOut_(None)
        self.visible = False

    def show(self):
        pass  # reappears on the next update with a gaze point

    def close(self):
        self.win.orderOut_(None)
        self.win.close()
