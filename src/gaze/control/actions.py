"""Mouse/keyboard on the real Mac. Direct helpers for the fast path + an executor for Claude's computer toolset."""

import subprocess
import time

import pyautogui

from .screen import image_block

pyautogui.PAUSE = 0.02  # pyautogui's FAILSAFE stays on: slam the mouse into the top-left corner to abort

# Claude uses xdotool-style key names; pyautogui uses its own.
_KEYS = {
    "return": "enter", "kp_enter": "enter", "escape": "esc", "backspace": "backspace", "delete": "delete",
    "page_down": "pagedown", "page_up": "pageup", "super": "command", "cmd": "command", "meta": "command",
    "control": "ctrl", "alt": "option", "caps_lock": "capslock", "minus": "-", "equal": "=", "plus": "+",
    "comma": ",", "period": ".", "slash": "/", "backslash": "\\", "bracketleft": "[", "bracketright": "]",
    "semicolon": ";", "apostrophe": "'", "grave": "`",
    **{f"kp_{i}": f"num{i}" for i in range(10)},  # numpad (Blender's view keys: KP_1 front, KP_3 side, KP_7 top)
}


def _key(name):
    n = name.strip().lower()
    return _KEYS.get(n, n)


def press(combo, repeat=1):
    keys = [_key(k) for k in combo.split("+")]
    for _ in range(repeat):
        pyautogui.hotkey(*keys) if len(keys) > 1 else pyautogui.press(keys[0])


def type_text(text):
    if text.isascii():
        pyautogui.write(text, interval=0.004)
        return
    subprocess.run(["pbcopy"], input=text.encode(), check=True)  # pyautogui can't type non-ASCII
    press("command+v")


def click(x, y, clicks=1, button="left", modifiers=()):
    pyautogui.moveTo(x, y)
    for m in modifiers:
        pyautogui.keyDown(m)
    try:
        pyautogui.click(clicks=clicks, interval=0.05, button=button)
    finally:
        for m in reversed(modifiers):
            pyautogui.keyUp(m)


_BUTTONS = {  # button -> (down, dragged, up event types, CG button number)
    "left": ("kCGEventLeftMouseDown", "kCGEventLeftMouseDragged", "kCGEventLeftMouseUp", 0),
    "right": ("kCGEventRightMouseDown", "kCGEventRightMouseDragged", "kCGEventRightMouseUp", 1),
    "middle": ("kCGEventOtherMouseDown", "kCGEventOtherMouseDragged", "kCGEventOtherMouseUp", 2),
}
_FLAGS = {"shift": "kCGEventFlagMaskShift", "ctrl": "kCGEventFlagMaskControl",
          "option": "kCGEventFlagMaskAlternate", "command": "kCGEventFlagMaskCommand"}


def _mouse_event(kind, x, y, button, flags):
    import Quartz

    event = Quartz.CGEventCreateMouseEvent(None, getattr(Quartz, kind), (x, y), button)
    Quartz.CGEventSetIntegerValueField(event, Quartz.kCGMouseEventButtonNumber, button)
    if flags:
        Quartz.CGEventSetFlags(event, flags)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


def drag(points, button="left", modifiers=(), duration=0.5):
    """Press `button` at points[0], move through the rest with real drag events, release at the end, holding
    `modifiers` throughout. Apps like Blender need this: middle-drag orbits, shift+middle pans, and a plain
    "move with the button down" (mouse-moved events) does nothing there."""
    import Quartz

    down, dragged, up, number = _BUTTONS[button]
    mods = [_key(m) for m in modifiers]
    flags = 0
    for m in mods:
        flags |= getattr(Quartz, _FLAGS[m]) if m in _FLAGS else 0
    pts = [(float(x), float(y)) for x, y in points]
    steps = max(int(duration * 60), 2 * (len(pts) - 1))  # ~60 Hz, at least a couple per segment
    per_segment = max(steps // max(len(pts) - 1, 1), 1)
    pyautogui.moveTo(*pts[0])
    for m in mods:
        pyautogui.keyDown(m)
    try:
        _mouse_event(down, *pts[0], number, flags)
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            for i in range(1, per_segment + 1):
                f = i / per_segment
                _mouse_event(dragged, x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, number, flags)
                time.sleep(duration / steps)
        _mouse_event(up, *pts[-1], number, flags)
    finally:
        for m in reversed(mods):
            pyautogui.keyUp(m)


def move_held(x, y, button="left"):
    """Move with a mouse button held down (after left_mouse_down): a drag event, which apps track, not a move."""
    _mouse_event(_BUTTONS[button][1], x, y, _BUTTONS[button][3], 0)


SCROLL_PX = 100  # pixels per "amount" step: "scroll down" (4) moves 400 px, "a little" (1) 100, "a lot" (10) 1000


def natural_scrolling():
    """macOS "natural scrolling" (on by default). It flips synthetic scroll events too, so pyautogui's sign is
    backwards with it on: "scroll down" scrolled up (or did nothing at the top of a page)."""
    from Foundation import NSUserDefaults

    value = NSUserDefaults.standardUserDefaults().objectForKey_("com.apple.swipescrolldirection")
    return True if value is None else bool(value)


def scroll_delta(direction, amount, natural):
    """(vertical, horizontal) pixel deltas for a CGEvent that moves the *content* the way "scroll down" means:
    down = show what's further down. Positive = down/right with natural scrolling, negative without."""
    px = int(amount) * SCROLL_PX * (1 if natural else -1)
    return {"down": (px, 0), "up": (-px, 0), "right": (0, px), "left": (0, -px)}[direction]


def scroll(x, y, direction, amount):
    import Quartz

    pyautogui.moveTo(x, y)  # scroll events go to the window under the pointer
    dy, dx = scroll_delta(direction, amount, natural_scrolling())
    event = Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitPixel, 2, dy, dx)
    Quartz.CGEventSetLocation(event, (x, y))
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)


_CLICKS = {
    "left_click": (1, "left"), "right_click": (1, "right"), "middle_click": (1, "middle"),
    "double_click": (2, "left"), "triple_click": (3, "left"),
}


class ComputerExecutor:
    """Runs one computer-toolset member call. Coordinates arrive in screenshot pixels."""

    def __init__(self, screen):
        self.screen = screen
        self.held = False  # left button down (left_mouse_down until left_mouse_up)

    def _pt(self, coordinate):
        return self.screen.to_points(*coordinate)

    def run(self, name, inp):
        ok = [{"type": "text", "text": "OK"}]
        mods = [_key(m) for m in inp.get("text", "").split("+") if m] if name != "type" else []

        if name == "screenshot":
            return [image_block(self.screen.screenshot())]
        if name == "zoom":
            return [image_block(self.screen.zoom(inp["region"]))]
        if name in _CLICKS:
            clicks, button = _CLICKS[name]
            x, y = self._pt(inp["coordinate"]) if "coordinate" in inp else pyautogui.position()
            click(x, y, clicks, button, mods)
            return ok
        if name == "left_click_drag":
            pyautogui.moveTo(*self._pt(inp["start_coordinate"]))
            pyautogui.dragTo(*self._pt(inp["coordinate"]), duration=0.3, button="left")
            return ok
        if name == "mouse_move":
            if self.held:  # moving with the button down is a drag: apps ignore plain moves then
                move_held(*self._pt(inp["coordinate"]))
            else:
                pyautogui.moveTo(*self._pt(inp["coordinate"]))
            return ok
        if name == "left_mouse_down":
            pyautogui.mouseDown()
            self.held = True
            return ok
        if name == "left_mouse_up":
            pyautogui.mouseUp()
            self.held = False
            return ok
        if name == "cursor_position":
            sx, sy = self.screen.to_shot(*pyautogui.position())
            return [{"type": "text", "text": f"X={round(sx)}, Y={round(sy)}"}]
        if name == "scroll":
            x, y = self._pt(inp["coordinate"]) if "coordinate" in inp else pyautogui.position()
            for m in mods:
                pyautogui.keyDown(m)
            try:
                scroll(x, y, inp["scroll_direction"], int(inp.get("scroll_amount", 3)))
            finally:
                for m in reversed(mods):
                    pyautogui.keyUp(m)
            return ok
        if name == "type":
            type_text(inp["text"])
            return ok
        if name == "key":
            press(inp["text"], int(inp.get("repeat", 1)))
            return ok
        if name == "hold_key":
            keys = [_key(k) for k in inp["text"].split("+")]
            for k in keys:
                pyautogui.keyDown(k)
            time.sleep(min(float(inp.get("duration", 1)), 10))
            for k in reversed(keys):
                pyautogui.keyUp(k)
            return ok
        if name == "wait":
            time.sleep(min(float(inp.get("duration", 1)), 10))
            return ok
        raise ValueError(f"unsupported computer action: {name}")
