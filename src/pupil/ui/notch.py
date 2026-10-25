"""The notch: what the app is doing, for the person using it (the web dashboard is for debugging).

A black shape fused to the MacBook's notch (or hanging from the top centre on other screens), in the spirit of
notchy.dev's Dynamic Island, and only about the happy path: look at something, say what to do, see it heard
and done. Collapsed it has small "ears": whether it sees you and is listening, and while a command runs, what
it's doing. It grows a line for what you should see (the agent's reply, "say confirm"), and opens when you hover
it or look at it: what it's doing, your last few commands and what they did, gaze quality, a mic toggle.

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
ROW_H = 24  # a recent command
RESEARCH_ROW_H, RESEARCH_ANSWER_H = 22, 18  # a research job, and its answer under it
RESEARCH_MAX_H = 20 + 2 * (RESEARCH_ROW_H + RESEARCH_ANSWER_H) + 6  # the "Research" label and two jobs
RECENT_ROWS = 3
OPEN_TOP_H = 12 + 38 + 14 + 20  # status row, then the "Recent" label
OPEN_FOOT_H = 8 + 14 + 18  # gap, footer line, bottom margin
OPEN_BODY_H = OPEN_TOP_H + RESEARCH_MAX_H + RECENT_ROWS * ROW_H + OPEN_FOOT_H  # the tallest, below the notch band
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
         "dictating": "#fa5252", "error": "#fa5252", "researching": "#3bc9db"}
GLYPH = {"listening": "waveform", "hearing": "waveform", "thinking": "sparkle", "writing": "pencil",
         "agent": "sparkles", "acting": "cursorarrow.rays", "done": "checkmark", "waiting": "questionmark.bubble",
         "speaking": "speaker.wave.2.fill", "dictating": "mic.fill", "error": "exclamationmark.triangle.fill",
         "researching": "magnifyingglass"}
ANIMATED = {"listening", "hearing", "thinking", "agent", "speaking", "dictating", "waiting", "researching"}


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
F_ROW = _font(12.5, 0.3)
F_ROW_DIM = _font(12, 0.0)
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


def _open_url(url: str) -> None:
    from AppKit import NSURL, NSWorkspace

    NSWorkspace.sharedWorkspace().openURL_(NSURL.URLWithString_(url))


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
    def __init__(self, on_toggle_voice=None):
        self._on_voice = on_toggle_voice
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
        live = content.ear_state(model) in ANIMATED or self._mode == "peek"  # pulsing glyphs, scrolling text
        key = (self._mode, model.state, model.label, model.heard, model.pending, model.caption, model.tracking,
               model.voice_on, model.model, tuple((a.get("did"), round(time.time() - a.get("t", 0)) // 60)
                                                  for a in model.activity),
               tuple((j.get("phase"), j.get("answer")) for j in model.research))
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
            return OPEN_W, self._notch_h + self._open_body_h(now), RADIUS_OPEN
        collapsed = self._collapsed_w(self._ear())
        if self._mode == "peek":
            return max(collapsed, PEEK_W), self._notch_h + PEEK_H, RADIUS + 6
        return collapsed, self._notch_h, RADIUS

    def _open_body_h(self, now):
        rows = max(1, len(content.recent(self.model, now)))  # none yet: one row of suggestions
        return OPEN_TOP_H + self._research_h(now) + rows * ROW_H + OPEN_FOOT_H

    def _research_h(self, now):
        jobs = content.research_rows(self.model, now)
        if not jobs:
            return 0
        return 20 + sum(RESEARCH_ROW_H + (RESEARCH_ANSWER_H if j["answer"] else 0) for j in jobs) + 6

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
        open_amount = max(0.0, min((h - nh - PEEK_H) / (self._open_body_h(time.time()) - PEEK_H), 1.0))
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
        label, state = content.ear_label(m), content.ear_state(m)
        if label:
            self._glyph(state, left + 22, cy)  # inset like the label on the right
            _text(label, F_EAR, _rgb(STATE.get(state, "#ffffff")), right - 13, cy - 8, alpha, align="right",
                  max_w=EAR_ACTIVE_MAX - 18)
            return
        # idle: whether it sees you (left) and whether it's listening for commands (right)
        ear = (self._w.x - self._notch_w) / 2
        _draw_symbol("eye.fill", 11, NAMED[content.TRACKING_COLOURS.get(m.tracking, "grey")], left + ear / 2 + 2, cy,
                     alpha)
        if m.voice_on is not None:
            _draw_symbol("mic.fill" if m.voice_on else "mic.slash.fill", 10.5, DIM if m.voice_on else FAINT,
                         right - ear / 2 - 2, cy, alpha)

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
                       "research": ("magnifyingglass", _rgb(STATE["researching"]))}[kind]
        cy = nh + PEEK_H / 2 - 1
        self._badge(left + 12, cy - 11, 22, color, alpha)
        _draw_symbol(icon, 10.5, color, left + 23, cy, alpha)
        x0 = left + 42
        x1 = right - 14
        if kind == "confirm":
            x1 -= self._chip("say “confirm”", x1, cy, color, alpha) + 8
        avail = x1 - x0
        width = text_width(text, F_PEEK)
        if width <= avail or kind == "confirm":
            _text(text, F_PEEK, WHITE, x0, cy - 8, alpha, max_w=avail)
            return
        if kind == "research":  # scrolls from when it finished
            started = next((j["ended"] for j in self.model.research if j.get("ended")), time.time())
        else:
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
        _text("Pupil", F_HEADER, DIM, x0 + 14, cy - 7.5, alpha)
        if self._on_voice is not None and m.voice_on is not None:
            self._button(x1 - 12, cy, "mic.fill" if m.voice_on else "mic.slash.fill", m.voice_on, self._on_voice, alpha)

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

        y += 38 + 14
        # research running in the background (and the last answers): click a finished one for its top source
        jobs = content.research_rows(m, time.time())
        if jobs:
            _text("Research", F_TILE_TITLE, FAINT, x0, y, alpha)
            y += 20
            cyan = _rgb(STATE["researching"])
            for job in jobs:
                if job["phase"] in ("starting", "running"):
                    pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.time() * 5))
                    _draw_symbol("magnifyingglass", 10.5, cyan, x0 + 6, y + 8, alpha * pulse)
                elif job["phase"] == "done":
                    _draw_symbol("checkmark", 10, cyan, x0 + 6, y + 8, alpha)
                else:
                    _draw_symbol("xmark", 10, TONES["bad"], x0 + 6, y + 8, alpha)
                sw = _text(job["status"], F_SMALL, FAINT, x1, y + 1, alpha, align="right")
                _text(job["question"], F_ROW, WHITE, x0 + 20, y, alpha, max_w=x1 - sw - 12 - (x0 + 20))
                top = y
                y += RESEARCH_ROW_H
                if job["answer"]:
                    _text(job["answer"], F_ROW_DIM, DIM, x0 + 20, y - 3, alpha, max_w=x1 - (x0 + 20))
                    y += RESEARCH_ANSWER_H
                if job["url"]:
                    url = job["url"]
                    self._view.buttons.append(((x0, top - 2, x1 - x0, y - top), lambda url=url: _open_url(url)))
            y += 6

        # recent: what you said, what it did
        rows = content.recent(m, time.time())
        _text("Recent" if rows else "Try saying", F_TILE_TITLE, FAINT, x0, y, alpha)
        y += 20
        row_h = ROW_H
        if rows:
            for said, did, when, by_agent in rows:
                if by_agent:  # the agent's own summary
                    _draw_symbol("sparkles", 10.5, _rgb(STATE["agent"]), x0 + 6, y + 8, alpha)
                else:
                    _draw_symbol("checkmark", 10, TONES["ok"], x0 + 6, y + 8, alpha)
                ww = _text(when, F_SMALL, FAINT, x1, y + 1, alpha, align="right")
                sw = _text(f"“{said}”", F_ROW, WHITE, x0 + 20, y, alpha, max_w=(x1 - x0) * 0.42)
                _text(did, F_ROW_DIM, DIM, x0 + 20 + sw + 10, y + 1, alpha, max_w=x1 - ww - 12 - (x0 + 20 + sw + 10))
                y += row_h
        else:  # nothing yet: what to try
            hx = x0
            for hint in content.HINTS:
                w = text_width(hint, F_CHIP) + 18
                if hx + w > x1:
                    break
                WHITE.colorWithAlphaComponent_(0.08 * alpha).set()
                NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(hx, y - 2, w, 20), 10, 10).fill()
                _text(hint, F_CHIP, WHITE, hx + w / 2, y + 0.5, alpha * 0.85, align="center")
                hx += w + 6
            y += row_h
        y += 8

        # footer: tracking quality left, the model being called (or the mic) right
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

    def hide(self) -> None:
        self._panel.orderOut_(None)
