"""Snap a rough gaze point to the real UI elements around it (macOS Accessibility).

`candidates_near` walks the accessibility tree of the window (or menu bar) under the gaze and returns every
actionable element nearby, so the fast tier can pick the one the user *named* ("full screen") even when a
different one ("close") is closer. Point-sampling misses small, tightly packed controls; walking the tree doesn't.
"""

import math
import re
import time
from dataclasses import dataclass

from ApplicationServices import (
    AXUIElementCopyAttributeValue,
    AXUIElementCopyElementAtPosition,
    AXUIElementCopyMultipleAttributeValues,
    AXUIElementCreateSystemWide,
    AXValueGetValue,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)

from . import config

ACTIONABLE = {
    "AXButton", "AXLink", "AXMenuItem", "AXMenuBarItem", "AXMenuButton", "AXPopUpButton",
    "AXCheckBox", "AXRadioButton", "AXTab", "AXDisclosureTriangle", "AXComboBox",
    "AXTextField", "AXTextArea", "AXSearchField", "AXSlider", "AXIncrementor",
    "AXCell", "AXRow", "AXImage", "AXDockItem",
}
CONTAINERS = {"AXWindow", "AXMenuBar", "AXSheet", "AXDrawer", "AXList"}  # where the tree walk starts
SEARCH_RADIUS = 220  # points: how far from the gaze candidates may be
MAX_NODES = 2500  # tree-walk budget
MAX_WALK_S = 0.12

_SYSTEM = AXUIElementCreateSystemWide()
_WALK_ATTRS = ["AXRole", "AXPosition", "AXSize", "AXChildren"]


@dataclass(frozen=True)
class Element:
    role: str
    label: str
    x: float
    y: float
    w: float
    h: float
    kind: str = ""  # what macOS calls it: "tab", "close button", "text field"
    parent: str = ""  # label of the actionable element it sits inside ("Close" inside tab "YouTube")

    @property
    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def distance(self, px, py):
        """0 inside the element, else distance to its nearest edge."""
        dx = max(self.x - px, 0, px - (self.x + self.w))
        dy = max(self.y - py, 0, py - (self.y + self.h))
        return math.hypot(dx, dy)

    def describe(self):
        return f'{self.role.removeprefix("AX")} "{self.label}"' if self.label else self.role.removeprefix("AX")


def _attr(element, name):
    err, value = AXUIElementCopyAttributeValue(element, name, None)
    return value if err == 0 else None


def _point(value, kind):
    if value is None:
        return None
    try:
        ok, v = AXValueGetValue(value, kind, None)
    except Exception:  # an AXError placeholder from a multi-attribute fetch
        return None
    return v if ok else None


def _frame_of(pos, size):
    p, s = _point(pos, kAXValueCGPointType), _point(size, kAXValueCGSizeType)
    return (p.x, p.y, s.width, s.height) if p is not None and s is not None else None


def _frame(element):
    return _frame_of(_attr(element, "AXPosition"), _attr(element, "AXSize"))


def _walk_attrs(element):
    """Role, frame and children in one IPC round trip (falls back to one call per attribute)."""
    try:
        err, values = AXUIElementCopyMultipleAttributeValues(element, _WALK_ATTRS, 0, None)
    except Exception:
        err, values = 1, None
    if err != 0 or values is None:
        values = [_attr(element, a) for a in _WALK_ATTRS]
    role, pos, size, children = values
    role = role if isinstance(role, str) else ""
    children = list(children) if children is not None and hasattr(children, "__len__") else []
    return role, _frame_of(pos, size), children


def _text(element, name):
    value = _attr(element, name)
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _label(element):
    """Best human-readable name. Window buttons and menu-bar icons often have no title: fall back to the
    role description from the subrole ("full screen button"), then the identifier
    ("com.apple.menuextra.wifi" -> "wifi"), then help text."""
    for name in ("AXTitle", "AXDescription", "AXValue"):
        if text := _text(element, name):
            return text[:120]
    subrole = _text(element, "AXSubrole")
    if subrole and subrole != "AXUnknown":
        return (_text(element, "AXRoleDescription") or subrole.removeprefix("AX"))[:60]
    if identifier := _text(element, "AXIdentifier"):
        return re.sub(r"[._-]+", " ", identifier.rsplit(".", 1)[-1]).strip()[:60]
    return (_text(element, "AXHelp") or _text(element, "AXPlaceholderValue"))[:120]


def _hit(x, y):
    err, element = AXUIElementCopyElementAtPosition(_SYSTEM, float(x), float(y), None)
    return element if err == 0 else None


def element_at(x, y):
    """Nearest actionable ancestor of the deepest element at (x, y), in screen points."""
    element = _hit(x, y)
    for _ in range(8):
        if element is None:
            return None
        role = _attr(element, "AXRole") or ""
        if role in ACTIONABLE:
            frame = _frame(element)
            if frame:
                return Element(role, _label(element), *frame)
        element = _attr(element, "AXParent")
    return None


def window_close_button(x, y):
    """The close button of the window at (x, y), as an Element, if it has one."""
    element = _hit(x, y)
    for _ in range(40):
        if element is None:
            return None
        if (_attr(element, "AXRole") or "") == "AXWindow":
            button = _attr(element, "AXCloseButton")
            frame = _frame(button) if button is not None else None
            return Element("AXButton", "Close window", *frame, kind="close button") if frame else None
        element = _attr(element, "AXParent")
    return None


def _container(element):
    """The window / menu bar / list that holds the element under the gaze."""
    last = element
    for _ in range(40):
        role = _attr(element, "AXRole") or ""
        if role in CONTAINERS:
            return element
        if role == "AXApplication":
            return last
        last, element = element, _attr(element, "AXParent")
        if element is None:
            return last
    return last


def _rect_distance(frame, px, py):
    x, y, w, h = frame
    return math.hypot(max(x - px, 0, px - (x + w)), max(y - py, 0, py - (y + h)))


def _usable_hit(x, y):
    """An element with a role at or near (x, y). Gaps between controls (e.g. menu-bar icons) return an
    invalid element, so look a little further out until something real is there."""
    for r in (0, 12, 30, 60):
        for i in range(8 if r else 1):
            a = i * math.pi / 4
            element = _hit(x + r * math.cos(a), y + r * math.sin(a))
            if element is not None and _attr(element, "AXRole"):
                return element
    return None


def candidates_near(x, y, screen_w, screen_h, radius=SEARCH_RADIUS):
    """Actionable elements within `radius` of the gaze, nearest first."""
    hit = _usable_hit(x, y)
    if hit is None:
        return []
    max_area = config.SNAP_MAX_AREA_FRACTION * screen_w * screen_h
    found = {}
    queue, seen, deadline = [(_container(hit), "")], 0, time.perf_counter() + MAX_WALK_S
    while queue and seen < MAX_NODES and time.perf_counter() < deadline:
        node, parent = queue.pop()  # parent: label of the nearest actionable ancestor
        seen += 1
        role, frame, children = _walk_attrs(node)
        far = frame is not None and frame[2] * frame[3] > 0 and _rect_distance(frame, x, y) > radius
        if far and role not in CONTAINERS:
            continue  # this whole subtree is far from the gaze (children sit inside their parent; containers
            # are exempt because some, like the menu-extras bar, report a bogus frame)
        if role in ACTIONABLE and frame and 0 < frame[2] * frame[3] <= max_area:  # zero-size = not on screen
            key = (role, round(frame[0]), round(frame[1]), round(frame[2]), round(frame[3]))
            if key not in found:
                found[key] = Element(role, _label(node), *frame, kind=_text(node, "AXRoleDescription")[:40],
                                     parent=parent)
            parent = found[key].label or parent
        queue.extend((child, parent) for child in children)
    direct = element_at(x, y)  # the element right under the gaze, even if the walk ran out of budget
    if direct:
        found.setdefault((direct.role, round(direct.x), round(direct.y), round(direct.w), round(direct.h)), direct)
    # Nearest edge first; among ties (gaze inside several nested elements) prefer the smaller one.
    return sorted(found.values(), key=lambda e: (e.distance(x, y), e.w * e.h))


def snap(x, y, screen_w, screen_h):
    """(best element or None, all candidates). The Jev gate picks among candidates by what the user said."""
    candidates = candidates_near(x, y, screen_w, screen_h)
    best = candidates[0] if candidates and candidates[0].distance(x, y) <= config.SNAP_RADIUS else None
    return best, candidates
