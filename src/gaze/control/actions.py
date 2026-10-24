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


def scroll(x, y, direction, amount):
    pyautogui.moveTo(x, y)
    step = amount * 4  # macOS "lines" are small
    if direction in ("up", "down"):
        pyautogui.scroll(step if direction == "up" else -step)
    else:
        pyautogui.hscroll(step if direction == "right" else -step)


_CLICKS = {
    "left_click": (1, "left"), "right_click": (1, "right"), "middle_click": (1, "middle"),
    "double_click": (2, "left"), "triple_click": (3, "left"),
}


class ComputerExecutor:
    """Runs one computer-toolset member call. Coordinates arrive in screenshot pixels."""

    def __init__(self, screen):
        self.screen = screen

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
            pyautogui.moveTo(*self._pt(inp["coordinate"]))
            return ok
        if name == "left_mouse_down":
            pyautogui.mouseDown()
            return ok
        if name == "left_mouse_up":
            pyautogui.mouseUp()
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
