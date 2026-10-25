"""The notch: what the app is doing, for the person using it (the web dashboard is for debugging).

A black shape fused to the MacBook's notch (or hanging from the top centre on other screens), in the spirit of
notchy.dev's Dynamic Island. Collapsed it has small "ears": the tracking state on the left, and while voice
control works, what it's doing on the right. It grows a line to say things you should see (the agent's reply,
"say confirm", an eye-health nudge), and opens fully when you hover it or look at it: status, health and gaze
quality at a glance, with voice and reminder toggles.

One borderless, non-activating panel above the menu bar; one view draws everything; sizes move on a spring.
Kept out of screenshots, so the agent never sees it; click-through unless open. Content: notch_content.py.
"""
from __future__ import annotations

import math
import time

import objc
from AppKit import (
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSCompositingOperationSourceOver,
    NSEvent,
    NSFont,
    NSFontDescriptorSystemDesignRounded,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSGradient,
    NSGraphicsContext,
    NSImage,
    NSImageSymbolConfiguration,
    NSMakeRect,
    NSPanel,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSShadow,
    NSString,
    NSView,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingNone,
    NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel,
)

from . import notch_content as content

# ---- geometry (points) ----
FALLBACK_W, FALLBACK_H = 180, 32  # screens without a notch: the shape hangs from the top centre
EAR_IDLE = 34  # each side, while idle
EAR_ACTIVE_MIN = 64
EAR_ACTIVE_MAX = 132
SHOULDER = 7  # the concave curve where the shape meets the top edge
RADIUS = 11  # bottom corners collapsed
RADIUS_OPEN = 22
PEEK_H = 30  # the line it grows for something to read
PEEK_W = 380
OPEN_W = 456
OPEN_BODY_H = 188  # below the notch band, open
SHADOW_ROOM = 40  # around the shape, for its shadow
MAX_W, MAX_H = OPEN_W + 2 * SHOULDER + 2 * SHADOW_ROOM, FALLBACK_H + OPEN_BODY_H + SHADOW_ROOM

# ---- feel ----
STIFFNESS, DAMPING = 340.0, 30.0  # spring: a little bounce, settles in ~0.35 s
HOVER_OPEN_S = 0.06  # hover this long to open (Notchy: 60 ms)
GAZE_OPEN_S = 0.6  # or look at it this long
CLOSE_AFTER_S = 0.35  # after the mouse leaves
GAZE_CLOSE_AFTER_S = 1.0  # after the gaze leaves (it jitters)
GAZE_ZONE = (140, 70)  # around the notch, generous: gaze is only good to a couple of cm


def _rgb(hex_: str, a: float = 1.0):
    r, g, b = (int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, a)


BLACK = NSColor.blackColor()
WHITE = _rgb("#ffffff")
DIM = _rgb("#ffffff", 0.55)
FAINT = _rgb("#ffffff", 0.32)
HAIR = _rgb("#ffffff", 0.08)
TILE = _rgb("#ffffff", 0.07)
TONES = {"ok": _rgb("#3ddc84"), "warn": _rgb("#ffc43d"), "bad": _rgb("#ff6b6b"), None: _rgb("#ffffff", 0.5)}
NAMED = {"green": _rgb("#3ddc84"), "amber": _rgb("#ffc43d"), "grey": _rgb("#8b93a1"), "blue": _rgb("#4dabf7")}
# voice states: the gaze pill's palette (control/indicator.py), so both speak the same colours
STATE = {"listening": "#4dabf7", "hearing": "#74c0fc", "thinking": "#ffc43d", "writing": "#ffc43d",
         "agent": "#b197fc", "acting": "#3ddc84", "done": "#3ddc84", "waiting": "#ff922b", "speaking": "#20c997",
         "dictating": "#fa5252", "error": "#fa5252"}
GLYPH = {"listening": "waveform", "hearing": "waveform", "thinking": "sparkle", "writing": "pencil",
         "agent": "sparkles", "acting": "cursorarrow.rays", "done": "checkmark", "waiting": "questionmark.bubble",
         "speaking": "speaker.wave.2.fill", "dictating": "mic.fill", "error": "exclamationmark.triangle.fill"}
ANIMATED = {"listening", "hearing", "thinking", "agent", "speaking", "dictating", "waiting"}


def _font(size, weight=0.0, rounded=False):
    font = NSFont.systemFontOfSize_weight_(size, weight)
    if rounded:  # SF Pro Rounded for numbers, like Apple's widgets
        descriptor = font.fontDescriptor().fontDescriptorWithDesign_(NSFontDescriptorSystemDesignRounded)
        font = NSFont.fontWithDescriptor_size_(descriptor, size) if descriptor is not None else font
    return font


F_EAR = _font(12, 0.35)
F_HEADER = _font(11.5, 0.4)
F_TITLE = _font(15, 0.45)
F_SUB = _font(12.5, 0.0)
F_SMALL = _font(10.5, 0.15)
F_TILE_TITLE = _font(11, 0.35)
F_TILE_VALUE = _font(24, 0.4, rounded=True)
F_UNIT = _font(11, 0.3)
F_PEEK = _font(13, 0.25)
F_CHIP = _font(11, 0.4)

_symbols: dict = {}


def _symbol(name, size, color_hex_or_color, weight=0.3):
    key = (name, size, str(color_hex_or_color), weight)
    if key not in _symbols:
        color = _rgb(color_hex_or_color) if isinstance(color_hex_or_color, str) else color_hex_or_color
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
        if img is not None:
            config = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, weight)
            config = config.configurationByApplyingConfiguration_(
                NSImageSymbolConfiguration.configurationWithPaletteColors_([color]))
            img = img.imageWithSymbolConfiguration_(config)
        _symbols[key] = img
    return _symbols[key]


def _draw_symbol(name, size, color, cx, cy, alpha=1.0):
    img = _symbol(name, size, color)
    if img is None:
        return
    s = img.size()
    img.drawInRect_fromRect_operation_fraction_respectFlipped_hints_(
        NSMakeRect(cx - s.width / 2, cy - s.height / 2, s.width, s.height), NSMakeRect(0, 0, 0, 0),
        NSCompositingOperationSourceOver, alpha, True, None)


def _text(s, font, color, x, y, alpha=1.0, max_w=None, align="left"):
    """Draw `s` with its top at y (flipped view); truncated with … to max_w. Returns its width."""
    attrs = {NSFontAttributeName: font, NSForegroundColorAttributeName: color.colorWithAlphaComponent_(
        color.alphaComponent() * alpha)}
    string = NSString.stringWithString_(s)
    w = string.sizeWithAttributes_(attrs).width
    if max_w is not None and w > max_w:
        while s and NSString.stringWithString_(s + "…").sizeWithAttributes_(attrs).width > max_w:
            s = s[:-1]
        string = NSString.stringWithString_(s.rstrip() + "…")
        w = string.sizeWithAttributes_(attrs).width
    if align == "right":
        x -= w
    elif align == "center":
        x -= w / 2
    string.drawAtPoint_withAttributes_((x, y), attrs)
    return w


def text_width(s, font) -> float:
    return NSString.stringWithString_(s).sizeWithAttributes_({NSFontAttributeName: font}).width


def notch_geometry(screen) -> tuple[float, float, bool]:
    """(width, height, real notch?) of the screen's notch, from the menu bar areas either side of it."""
    try:
        top = screen.safeAreaInsets().top
        left, right = screen.auxiliaryTopLeftArea(), screen.auxiliaryTopRightArea()
        if top > 0 and left.size.width > 0 and right.size.width > 0:
            width = screen.frame().size.width - left.size.width - right.size.width
            if width > 40:
                return float(width), float(top), True
    except (AttributeError, objc.error):
        pass
    return float(FALLBACK_W), float(FALLBACK_H), False


def shape_path(cx, w, h, radius, shoulder=SHOULDER):
    """The notch shape in a flipped view: hangs from y=0, concave shoulders into the top edge, rounded bottom."""
    left, right = cx - w / 2, cx + w / 2
    r = min(radius, h / 2, w / 2)
    p = NSBezierPath.bezierPath()
    p.moveToPoint_((left - shoulder, 0))
    p.curveToPoint_controlPoint1_controlPoint2_((left, shoulder), (left - shoulder / 3, 0), (left, shoulder / 3))
    p.lineToPoint_((left, h - r))
    p.appendBezierPathWithArcFromPoint_toPoint_radius_((left, h), (left + r, h), r)
    p.lineToPoint_((right - r, h))
    p.appendBezierPathWithArcFromPoint_toPoint_radius_((right, h), (right, h - r), r)
    p.lineToPoint_((right, shoulder))
    p.curveToPoint_controlPoint1_controlPoint2_((right + shoulder, 0), (right, shoulder / 3), (right + shoulder / 3, 0))
    p.closePath()
    return p


class _Spring:
    def __init__(self, x):
        self.x, self.v, self.target = x, 0.0, x

    def step(self, dt):
        a = STIFFNESS * (self.target - self.x) - DAMPING * self.v
        self.v += a * dt
        self.x += self.v * dt
        if abs(self.target - self.x) < 0.2 and abs(self.v) < 2:
            self.x, self.v = self.target, 0.0

    @property
    def settled(self):
        return self.x == self.target and self.v == 0.0


class _NotchPanel(NSPanel):
    """A panel macOS lets sit over the menu bar and the notch: AppKit pushes windows below the menu bar
    (constrainFrameRect:toScreen:) unless they opt out, and it must never take focus from the app you're in."""

    def constrainFrameRect_toScreen_(self, rect, screen):
        return rect

    def canBecomeKeyWindow(self):
        return False

    def canBecomeMainWindow(self):
        return False


class _NotchView(NSView):
    def isFlipped(self):
        return True

    def acceptsFirstMouse_(self, event):
        return True

    def mouseDown_(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        for rect, action in getattr(self, "buttons", []):
            x, y, w, h = rect
            if x <= p.x <= x + w and y <= p.y <= y + h:
                action()
                return

    def drawRect_(self, rect):
        self.owner._draw(self.bounds().size.width)


class Notch:
    def __init__(self, on_toggle_voice=None, on_toggle_reminders=None):
        self._on_voice, self._on_reminders = on_toggle_voice, on_toggle_reminders
        screen = NSScreen.mainScreen()
        self._notch_w, self._notch_h, self.real_notch = notch_geometry(screen)
        frame = screen.frame()
        self._screen = frame
        panel = _NotchPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(frame.origin.x + (frame.size.width - MAX_W) / 2, frame.origin.y + frame.size.height - MAX_H,
                       MAX_W, MAX_H),
            NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered, False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setHasShadow_(False)
        panel.setIgnoresMouseEvents_(True)
        panel.setSharingType_(NSWindowSharingNone)  # never in screenshots: the agent doesn't see it
        panel.setHidesOnDeactivate_(False)
        # over the menu bar and the gaze blob (overlay.py, screen-saver level), so the blob slides under it; set
        # after anything that resets the level
        panel.setLevel_(NSScreenSaverWindowLevel + 1)
        panel.setMovable_(False)
        panel.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                                     | NSWindowCollectionBehaviorFullScreenAuxiliary
                                     | NSWindowCollectionBehaviorIgnoresCycle)
        view = _NotchView.alloc().initWithFrame_(NSMakeRect(0, 0, MAX_W, MAX_H))
        view.owner, view.buttons = self, []
        panel.setContentView_(view)
        self._panel, self._view = panel, view
        self._place()
        self.model = content.NotchModel()
        self._w, self._h = _Spring(self._collapsed_w(0)), _Spring(self._notch_h)
        self._radius = _Spring(RADIUS)
        self._mode = "collapsed"  # collapsed | peek | open
        self._hover_since = self._away_since = self._gaze_since = None
        self._opened_by_gaze = False
        self._last_tick = time.time()
        self._drawn_key = None
        self.enabled = True

    def _place(self):
        """Pin the panel's top edge to the screen's top edge, centred on the notch."""
        f = self._screen
        self._panel.setFrameOrigin_((f.origin.x + (f.size.width - MAX_W) / 2, f.origin.y + f.size.height - MAX_H))

    # ---- per tick (main thread, ~30 Hz) ----

    def update(self, model: content.NotchModel, gaze=None, hidden=False) -> None:
        """gaze: where the user looks (top-left screen points) or None. hidden: e.g. while calibrating."""
        now = time.time()
        dt = min(now - self._last_tick, 0.05)
        self._last_tick = now
        if hidden or not self.enabled:
            if self._panel.isVisible():
                self._panel.orderOut_(None)
            return
        self.model = model
        self._mode = self._next_mode(now, gaze)
        w, h, r = self._targets(now)
        self._w.target, self._h.target, self._radius.target = w, h, r
        for s in (self._w, self._h, self._radius):
            s.step(dt)
        self._panel.setIgnoresMouseEvents_(self._mode != "open")
        moving = not (self._w.settled and self._h.settled and self._radius.settled)
        live = model.state in ANIMATED or self._mode == "peek"  # pulsing glyphs, scrolling text
        key = (self._mode, model.state, model.label, model.heard, model.pending, model.caption, model.tracking,
               model.voice_on, model.model, str(model.health.get("fatigue")), str(model.health.get("blink_rate")),
               round(model.health.get("minutes_on_screen") or 0), model.reminders_on)
        if moving or live or key != self._drawn_key:
            self._drawn_key = key
            self._view.setNeedsDisplay_(True)
        if not self._panel.isVisible():
            self._place()
            self._panel.orderFrontRegardless()

    def _collapsed_w(self, ear):
        return self._notch_w + 2 * ear

    def _ear(self) -> float:
        m = self.model
        label = content.ear_label(m)
        if not label:
            return EAR_IDLE
        return min(max(EAR_ACTIVE_MIN, text_width(label, F_EAR) + 30), EAR_ACTIVE_MAX)

    def _targets(self, now):
        if self._mode == "open":
            return OPEN_W, self._notch_h + OPEN_BODY_H, RADIUS_OPEN
        collapsed = self._collapsed_w(self._ear())
        if self._mode == "peek":
            return max(collapsed, PEEK_W), self._notch_h + PEEK_H, RADIUS + 6
        return collapsed, self._notch_h, RADIUS

    def _next_mode(self, now, gaze):
        mouse = NSEvent.mouseLocation()
        top = self._screen.origin.y + self._screen.size.height
        mx, my = mouse.x - self._screen.origin.x, top - mouse.y  # top-left, like gaze
        cx = self._screen.size.width / 2
        w, h = self._w.x, self._h.x
        margin = 10 if self._mode == "open" else 3
        over = abs(mx - cx) <= w / 2 + margin and my <= h + margin
        looking = gaze is not None and abs(gaze[0] - cx) <= self._notch_w / 2 + GAZE_ZONE[0] and gaze[1] <= GAZE_ZONE[1]

        if over:
            self._hover_since = self._hover_since or now
        else:
            self._hover_since = None
        if looking and not over:
            self._gaze_since = self._gaze_since or now
        elif not looking:
            self._gaze_since = None

        if self._mode == "open":
            keep = over or (self._opened_by_gaze and looking)
            if keep:
                self._away_since = None
                return "open"
            self._away_since = self._away_since or now
            wait = GAZE_CLOSE_AFTER_S if self._opened_by_gaze else CLOSE_AFTER_S
            if now - self._away_since < wait:
                return "open"
        self._away_since = None
        if self._hover_since and now - self._hover_since >= HOVER_OPEN_S:
            self._opened_by_gaze = False
            return "open"
        if self._gaze_since and now - self._gaze_since >= GAZE_OPEN_S:
            self._opened_by_gaze = True
            return "open"
        return "peek" if content.peek_line(self.model, now) else "collapsed"

    # ---- drawing ----

    def _draw(self, view_w):
        NSColor.clearColor().set()
        NSBezierPath.fillRect_(NSMakeRect(0, 0, view_w, MAX_H))
        cx, w, h = view_w / 2, self._w.x, self._h.x
        left, right = cx - w / 2, cx + w / 2
        nh = self._notch_h
        open_amount = max(0.0, min((h - nh - PEEK_H) / (OPEN_BODY_H - PEEK_H), 1.0))
        peek_amount = max(0.0, min((h - nh) / PEEK_H, 1.0)) * (1 - open_amount)
        shape = shape_path(cx, w, h, self._radius.x)
        NSGraphicsContext.saveGraphicsState()
        if h > nh + 2:  # lifted off the screen: a soft shadow, deeper as it opens
            shadow = NSShadow.alloc().init()
            shadow.setShadowBlurRadius_(10 + 18 * open_amount)
            shadow.setShadowOffset_((0, -(3 + 7 * open_amount)))
            shadow.setShadowColor_(NSColor.colorWithCalibratedWhite_alpha_(0, 0.3 + 0.25 * open_amount))
            shadow.set()
        BLACK.set()
        shape.fill()
        NSGraphicsContext.restoreGraphicsState()
        self._view.buttons = []
        self._draw_ears(left, right, nh, 1.0)
        if peek_amount > 0.05:
            self._draw_peek(left, right, nh, peek_amount)
        if open_amount > 0.05:
            NSGraphicsContext.saveGraphicsState()
            shape.addClip()
            self._draw_open(left, right, nh, open_amount ** 2)
            NSGraphicsContext.restoreGraphicsState()

    def _glyph(self, state, cx, cy, size=13, alpha=1.0):
        """The voice state's glyph; listening/speaking get live bars, others a soft pulse."""
        color = _rgb(STATE.get(state, "#ffffff"))
        t = time.time()
        if state in ("listening", "speaking", "dictating", "hearing"):
            scale = size / 13
            for i in range(4):
                bar_h = (3 + 9 * (0.5 + 0.5 * math.sin(t * 9 + i * 1.3)) * (1 if state != "hearing" else 0.5)) * scale
                color.colorWithAlphaComponent_(alpha).set()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    NSMakeRect(cx + (-8 + i * 4.6) * scale, cy - bar_h / 2, 2.6 * scale, bar_h), 1.3 * scale,
                    1.3 * scale).fill()
            return
        pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(t * 5)) if state in ANIMATED else 1.0
        _draw_symbol(GLYPH.get(state, "circle.fill"), size, color, cx, cy, alpha * pulse)

    def _draw_ears(self, left, right, nh, alpha):
        m = self.model
        cy = nh / 2
        if self._mode == "open":
            return  # the open body has its own header
        label = content.ear_label(m)
        if label:
            self._glyph(m.state, left + 22, cy)  # inset like the label on the right
            _text(label, F_EAR, _rgb(STATE.get(m.state, "#ffffff")), right - 13, cy - 8, alpha, align="right",
                  max_w=EAR_ACTIVE_MAX - 18)
            return
        # idle: tracking on the left, a small fatigue ring on the right
        ear = (self._w.x - self._notch_w) / 2
        _draw_symbol("eye.fill", 11, NAMED[content.TRACKING_COLOURS.get(m.tracking, "grey")], left + ear / 2 + 2, cy,
                     alpha)
        fatigue = m.health.get("fatigue")
        tone = {"low": "ok", "moderate": "warn", "high": "bad"}.get(m.health.get("level"))
        self._ring(right - ear / 2 - 2, cy, 6.5, None if fatigue is None else fatigue / 100, TONES[tone], 2.2)

    def _ring(self, cx, cy, r, progress, color, width, alpha=1.0):
        track = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - r, cy - r, 2 * r, 2 * r))
        track.setLineWidth_(width)
        color.colorWithAlphaComponent_(0.18 * alpha).set()
        track.stroke()
        if progress:
            arc = NSBezierPath.bezierPath()
            # flipped view: angles run clockwise; start at the top
            arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                (cx, cy), r, -90, -90 + 360 * min(progress, 1.0), False)
            arc.setLineWidth_(width)
            arc.setLineCapStyle_(1)  # round
            color.colorWithAlphaComponent_(alpha).set()
            arc.stroke()

    def _badge(self, x, y, d, color, alpha):
        """A tinted circle for a leading symbol, like a Live Activity's."""
        color.colorWithAlphaComponent_(0.18 * alpha).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x, y, d, d)).fill()

    def _chip(self, text, x_right, cy, color, alpha):
        """A small capsule label ending at x_right; returns its width."""
        w = text_width(text, F_CHIP) + 18
        color.colorWithAlphaComponent_(0.16 * alpha).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(x_right - w, cy - 10, w, 20), 10, 10).fill()
        _text(text, F_CHIP, color, x_right - w / 2, cy - 7.5, alpha, align="center")
        return w

    def _draw_peek(self, left, right, nh, alpha):
        line = content.peek_line(self.model, time.time())
        if line is None:
            return
        kind, text = line
        icon, color = {"confirm": ("hand.raised.fill", _rgb(STATE["waiting"])),
                       "speech": ("sparkles", _rgb(STATE["agent"])),
                       "reminder": ("eye.fill", TONES["warn"])}[kind]
        cy = nh + PEEK_H / 2 - 1
        self._badge(left + 12, cy - 11, 22, color, alpha)
        _draw_symbol(icon, 10.5, color, left + 23, cy, alpha)
        x0 = left + 42
        x1 = right - 14
        if kind == "confirm":
            x1 -= self._chip("say “confirm”", x1, cy, color, alpha) + 8
        avail = x1 - x0
        width = text_width(text, F_PEEK)
        if width <= avail or kind != "speech":
            _text(text, F_PEEK, WHITE, x0, cy - 8, alpha, max_w=avail)
            return
        started = self.model.caption[1] if self.model.caption else time.time()
        offset = max(0.0, min((time.time() - started - 0.8) * 45, width - avail + 12))
        NSGraphicsContext.saveGraphicsState()
        NSBezierPath.clipRect_(NSMakeRect(x0, cy - 10, avail, 20))
        _text(text, F_PEEK, WHITE, x0 - offset, cy - 8, alpha)
        NSGraphicsContext.restoreGraphicsState()

    def _button(self, cx, cy, symbol, on, action, alpha):
        WHITE.colorWithAlphaComponent_((0.14 if on else 0.06) * alpha).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - 12, cy - 12, 24, 24)).fill()
        _draw_symbol(symbol, 10.5, WHITE if on else FAINT, cx, cy, alpha)
        self._view.buttons.append(((cx - 14, cy - 14, 28, 28), action))

    def _draw_open(self, left, right, nh, alpha):
        m = self.model
        pad = 18
        x0, x1 = left + pad, right - pad
        cy = nh / 2
        tracking_color = NAMED[content.TRACKING_COLOURS.get(m.tracking, "grey")]
        # header in the notch band: a tracking dot and the name left, toggles right
        tracking_color.colorWithAlphaComponent_(alpha).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x0 + 2, cy - 3, 6, 6)).fill()
        _text("Gaze", F_HEADER, DIM, x0 + 14, cy - 7.5, alpha)
        bx = x1 - 12
        if self._on_reminders is not None:
            self._button(bx, cy, "bell.fill" if m.reminders_on else "bell.slash.fill", m.reminders_on,
                         self._on_reminders, alpha)
            bx -= 32
        if self._on_voice is not None and m.voice_on is not None:
            self._button(bx, cy, "mic.fill" if m.voice_on else "mic.slash.fill", m.voice_on, self._on_voice, alpha)

        # status row: a tinted badge, what it's doing, what it heard or said
        y = nh + 12
        title, sub = content.headline(m)
        color = _rgb(STATE[m.state]) if m.state in STATE else tracking_color
        self._badge(x0, y, 38, color, alpha)
        if m.state:
            self._glyph(m.state, x0 + 19, y + 19, size=15, alpha=alpha)
        else:
            _draw_symbol("eye.fill" if m.tracking == "tracking" else "eye.slash.fill", 14, color, x0 + 19, y + 19, alpha)
        tx = x0 + 50
        _text(title, F_TITLE, WHITE, tx, y + 1, alpha, max_w=x1 - tx)
        if sub:
            _text(sub, F_SUB, DIM, tx, y + 21, alpha, max_w=x1 - tx)

        # health: three small widgets
        y += 38 + 14
        tiles = content.tiles(m)
        gap, th = 8, 84
        tw = (x1 - x0 - gap * (len(tiles) - 1)) / len(tiles)
        for i, tile in enumerate(tiles):
            tx = x0 + i * (tw + gap)
            box = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(tx, y, tw, th), 14, 14)
            TILE.colorWithAlphaComponent_(TILE.alphaComponent() * alpha).set()
            box.fill()
            HAIR.colorWithAlphaComponent_(HAIR.alphaComponent() * alpha).set()
            box.setLineWidth_(0.5)
            box.stroke()
            color = TONES[tile.tone] if tile.tone else DIM
            _draw_symbol(tile.icon, 9.5, color, tx + 17, y + 15, alpha)
            _text(tile.title, F_TILE_TITLE, DIM, tx + 26, y + 8, alpha)
            value_color = color if tile.title == "Fatigue" or tile.tone in ("warn", "bad") else WHITE
            vw = _text(tile.value, F_TILE_VALUE, value_color, tx + 12, y + 26, alpha)
            if tile.unit:
                _text(tile.unit, F_UNIT, DIM, tx + 12 + vw + 3, y + 38, alpha)
            _text(tile.detail, F_SMALL, DIM, tx + 12, y + 62, alpha, max_w=tw - 22)
            if tile.title == "Fatigue":  # its last 30 minutes, beside the score
                self._sparkline(tx + tw - 66, y + 30, 54, 24, color, alpha)
            elif tile.progress is not None:
                self._ring(tx + tw - 22, y + 40, 10, tile.progress, color, 3.5, alpha)

        # footer: tracking quality left, the model being called (or the mic) right
        y += th + 12
        foot_l, foot_r = content.footer(m)
        state, _, rest = foot_l.partition(" · ")
        tracking_color.colorWithAlphaComponent_(alpha).set()
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(x0 + 2, y + 5, 5, 5)).fill()
        sw = _text(state, F_SMALL, DIM, x0 + 12, y, alpha)
        if rest:
            _text(rest, F_SMALL, FAINT, x0 + 12 + sw + 6, y, alpha, max_w=(x1 - x0) * 0.55 - sw)
        if foot_r:
            rw = _text(foot_r, F_SMALL, FAINT, x1, y, alpha, max_w=(x1 - x0) * 0.36, align="right")
            _draw_symbol("sparkles" if m.model else "mic.fill", 9, FAINT, x1 - rw - 9, y + 7, alpha)

    def _sparkline(self, x, y, w, h, color, alpha):
        pts = content.sparkline(self.model.fatigue_series, time.time())
        if len(pts) < 2:
            return
        k = max(1, len(pts) // 24)  # a few points per minute is plenty: smooth the jitter
        pts = [(sum(p[0] for p in pts[i:i + k]) / len(pts[i:i + k]), sum(p[1] for p in pts[i:i + k]) / len(pts[i:i + k]))
               for i in range(0, len(pts), k)]
        line = NSBezierPath.bezierPath()
        for i, (px, py) in enumerate(pts):
            point = (x + px * w, y + h - py * h)
            line.moveToPoint_(point) if i == 0 else line.lineToPoint_(point)
        area = line.copy()
        area.lineToPoint_((x + pts[-1][0] * w, y + h))
        area.lineToPoint_((x + pts[0][0] * w, y + h))
        area.closePath()
        NSGradient.alloc().initWithStartingColor_endingColor_(
            color.colorWithAlphaComponent_(0.35 * alpha), color.colorWithAlphaComponent_(0.0)).drawInBezierPath_angle_(
            area, 90)
        line.setLineWidth_(1.6)
        line.setLineJoinStyle_(1)
        line.setLineCapStyle_(1)
        color.colorWithAlphaComponent_(alpha).set()
        line.stroke()
        ex, ey = x + pts[-1][0] * w, y + h - pts[-1][1] * h  # now
        NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(ex - 2.5, ey - 2.5, 5, 5)).fill()

    def hide(self) -> None:
        self._panel.orderOut_(None)
