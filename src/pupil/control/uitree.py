"""Any app, without pixels: its menu bar, its accessibility tree, and AppleScript (the agent's app-agnostic tools).

Native apps expose every command in their menus and every control in the accessibility tree, so the agent can
choose "File > Export > PNG" or press the "Send" button exactly instead of guessing coordinates from a
screenshot. Apps that draw their own UI (Blender, games, canvases) expose little here; the agent falls back to
screenshots, mouse and keys, or an app adapter (apps/).
"""
from __future__ import annotations

import subprocess
import time

from AppKit import NSWorkspace
from ApplicationServices import AXUIElementCreateApplication, AXUIElementPerformAction, AXUIElementSetAttributeValue

from .snapping import _attr, _frame, _label, _text

MAX_ELEMENTS = 150
MAX_NODES = 4000
APPLESCRIPT_TIMEOUT_S = 30
_INTERESTING = {
    "AXButton", "AXLink", "AXMenuButton", "AXPopUpButton", "AXCheckBox", "AXRadioButton", "AXTab", "AXSlider",
    "AXTextField", "AXTextArea", "AXSearchField", "AXComboBox", "AXCell", "AXRow", "AXDisclosureTriangle",
    "AXIncrementor", "AXStaticText", "AXImage", "AXSegmentedControl", "AXSwitch", "AXColorWell", "AXHeading",
}
_MODS = {0: "⌘", 1: "⇧⌘", 2: "⌥⌘", 3: "⌥⇧⌘", 4: "⌃⌘", 5: "⌃⇧⌘", 6: "⌃⌥⌘", 7: "⌃⌥⇧⌘", 8: ""}


def _app(name: str | None = None):
    """(AX application element, name): the named running app, else the frontmost one."""
    workspace = NSWorkspace.sharedWorkspace()
    if name:
        wanted = name.lower()
        for app in workspace.runningApplications():
            app_name = str(app.localizedName() or "")
            if app_name.lower() == wanted or wanted in app_name.lower():
                return AXUIElementCreateApplication(app.processIdentifier()), app_name
        raise LookupError(f"{name!r} isn't running (open_app first)")
    app = workspace.frontmostApplication()
    return AXUIElementCreateApplication(app.processIdentifier()), str(app.localizedName())


def front_context() -> tuple[str, str]:
    """Frontmost app and focused-window title, for validating short-lived voice references."""
    try:
        app_element, app_name = _app()
        window = _attr(app_element, "AXFocusedWindow") or (_attr(app_element, "AXWindows") or [None])[0]
        return app_name, _text(window, "AXTitle") if window is not None else ""
    except Exception:
        return "", ""


# ---- menus ---------------------------------------------------------------------------

def _children(element):
    return list(_attr(element, "AXChildren") or [])


def _menu_items(menu_bar_item):
    menus = [c for c in _children(menu_bar_item) if _attr(c, "AXRole") == "AXMenu"]
    return [i for i in _children(menus[0]) if _text(i, "AXTitle")] if menus else []


def _shortcut(item) -> str:
    char = _text(item, "AXMenuItemCmdChar")
    if not char:
        return ""
    mods = _attr(item, "AXMenuItemCmdModifiers")
    return f"{_MODS.get(int(mods) if mods is not None else 0, '⌘')}{char}"


def _describe_item(item) -> str:
    parts = [_text(item, "AXTitle")]
    if shortcut := _shortcut(item):
        parts.append(f"({shortcut})")
    if _attr(item, "AXEnabled") is False:
        parts.append("[disabled]")
    if _menu_items(item):
        parts.append("▸")
    return " ".join(parts)


def _walk_path(app_element, path: list[str]):
    """The menu bar item / menu item at `path` (titles, case-insensitive, prefix matches allowed)."""
    bar = _attr(app_element, "AXMenuBar")
    if bar is None:
        raise LookupError("this app has no menu bar")
    level = [c for c in _children(bar) if _text(c, "AXTitle")]
    item = None
    for i, title in enumerate(path):
        want = title.strip().lower().rstrip(".…")
        exact = [c for c in level if _text(c, "AXTitle").lower().rstrip(".…") == want]
        prefix = [c for c in level if _text(c, "AXTitle").lower().startswith(want)]
        matches = exact or prefix
        if not matches:
            have = ", ".join(_text(c, "AXTitle") for c in level[:40])
            raise LookupError(f"no {title!r} in {' > '.join(path[:i]) or 'the menu bar'}; there is: {have}")
        item = matches[0]
        level = _menu_items(item)
    return item


def menu(action: str = "list", path: str = "", app: str | None = None) -> str:
    """list: the menu bar, or the items under `path` ("File" or "File > Export"); choose: press the item."""
    app_element, app_name = _app(app)
    titles = [p for p in path.split(">") if p.strip()] if path else []
    if action == "choose":
        if not titles:
            raise ValueError("choose needs a path like \"File > Save\"")
        item = _walk_path(app_element, titles)
        if _attr(item, "AXEnabled") is False:
            raise RuntimeError(f"{path!r} is disabled right now")
        err = AXUIElementPerformAction(item, "AXPress")
        if err != 0:
            raise RuntimeError(f"couldn't press {path!r} (AX error {err})")
        return f"Chose {app_name}: {' > '.join(t.strip() for t in titles)}"
    if not titles:
        bar = _attr(app_element, "AXMenuBar")
        if bar is None:
            return f"{app_name} has no menu bar."
        tops = [_text(c, "AXTitle") for c in _children(bar) if _text(c, "AXTitle")]
        lines = [f"{app_name} menus: " + " | ".join(tops)]
        for top in _children(bar)[1:]:  # skip the Apple menu
            items = _menu_items(top)
            if items:
                lines.append(f"{_text(top, 'AXTitle')}: " + "; ".join(_describe_item(i) for i in items[:30]))
        return "\n".join(lines)
    item = _walk_path(app_element, titles)
    items = _menu_items(item)
    return f"{' > '.join(t.strip() for t in titles)}: " + "; ".join(_describe_item(i) for i in items) if items \
        else f"{path!r} is a command, not a submenu."


_index_cache: dict = {}  # pid -> (built at, {normalised title: menu item})
INDEX_TTL_S = 60


def _norm_title(title: str) -> str:
    return title.lower().replace("…", "").replace("...", "").strip()


def _menu_index(app_element, pid, rebuild=False) -> dict:
    """Every menu command of the app by normalised title (menus and one level of submenus), cached per app."""
    hit = _index_cache.get(pid)
    if hit and not rebuild and time.time() - hit[0] < INDEX_TTL_S:
        return hit[1]
    index = {}
    bar = _attr(app_element, "AXMenuBar")
    for top in (_children(bar) if bar is not None else [])[1:]:  # skip the Apple menu
        for item in _menu_items(top):
            index.setdefault(_norm_title(_text(item, "AXTitle")), item)
            for sub in _menu_items(item):
                index.setdefault(_norm_title(_text(sub, "AXTitle")), sub)
    _index_cache[pid] = (time.time(), index)
    return index


def menu_commands() -> tuple[str, list[str]]:
    """The frontmost app's name and its menu commands as "Title (shortcut)" (menus and one level of submenus)."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return "", []
    pid = app.processIdentifier()
    index = _menu_index(AXUIElementCreateApplication(pid), pid)
    return str(app.localizedName() or ""), [_describe_command(item) for item in index.values()]


def _describe_command(item) -> str:
    title = _text(item, "AXTitle")
    return f"{title} ({shortcut})" if (shortcut := _shortcut(item)) else title


def press_menu_command(command: str) -> str | None:
    """Press the frontmost app's menu command `command` (as menu_commands lists it). Returns its title, "" if it
    exists but is disabled right now, or None if the app has no such command."""
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return None
    pid = app.processIdentifier()
    app_element = AXUIElementCreateApplication(pid)
    for attempt in range(2):
        if attempt and time.time() - _index_cache.get(pid, (0,))[0] < 5:
            break  # just indexed: it really isn't there
        index = _menu_index(app_element, pid, rebuild=attempt > 0)
        item = next((it for it in index.values() if _describe_command(it) == command), None)
        if item is not None:
            if _attr(item, "AXEnabled") is False:
                return ""
            if AXUIElementPerformAction(item, "AXPress") == 0:
                return _text(item, "AXTitle")
        # not found, or a stale element (the menu was rebuilt): re-index once
    return None


# ---- the accessibility tree ---------------------------------------------------------------

_cache: list = []  # elements from the last ui_elements call, by id


def ui_elements(app: str | None = None, contains: str = "", to_shot=lambda x, y: (x, y)) -> str:
    """The controls and text in the app's focused window: one line each with an id for press/set_value."""
    global _cache
    app_element, app_name = _app(app)
    window = _attr(app_element, "AXFocusedWindow") or (_attr(app_element, "AXWindows") or [None])[0]
    if window is None:
        return f"{app_name} has no window open."
    want = contains.lower()
    found, queue, seen = [], [window], 0
    while queue and seen < MAX_NODES and len(found) < MAX_ELEMENTS:
        node = queue.pop(0)  # breadth first: the window's main structure before deep detail
        seen += 1
        role = _text(node, "AXRole")
        if role in _INTERESTING:
            frame = _frame(node)
            label = _label(node)
            if frame and frame[2] > 0 and frame[3] > 0 and (label or role != "AXStaticText"):
                if not want or want in label.lower():
                    found.append((node, role, label, frame))
        queue.extend(_children(node))
    _cache = [node for node, *_ in found]
    title = _text(window, "AXTitle")
    lines = [f"{app_name} — {title!r}: {len(found)} elements" + (" (more not shown)" if len(found) >= MAX_ELEMENTS else "")]
    for i, (node, role, label, (x, y, w, h)) in enumerate(found):
        kind = _text(node, "AXRoleDescription") or role.removeprefix("AX")
        value = _attr(node, "AXValue")
        extra = f" = {str(value)[:60]!r}" if value not in (None, "") and str(value) != label else ""
        disabled = " [disabled]" if _attr(node, "AXEnabled") is False else ""
        cx, cy = to_shot(x + w / 2, y + h / 2)
        lines.append(f"[{i}] {kind}: {label[:80]!r}{extra}{disabled} at ({round(cx)}, {round(cy)})")
    return "\n".join(lines)


def _cached(element_id: int):
    if not 0 <= element_id < len(_cache):
        raise LookupError(f"no element [{element_id}]: call ui_elements first (ids change each time)")
    return _cache[element_id]


def describe(element_id: int) -> str:
    """'button "Send"' for an id from the last ui_elements call (for the risk check)."""
    try:
        node = _cached(element_id)
    except LookupError:
        return f"element [{element_id}]"
    return f'{_text(node, "AXRoleDescription") or "element"} "{_label(node)[:60]}"'


def press_element(element_id: int) -> str:
    node = _cached(element_id)
    err = AXUIElementPerformAction(node, "AXPress")
    if err != 0:
        raise RuntimeError(f"[{element_id}] can't be pressed (AX error {err}): click its position instead")
    return f"Pressed [{element_id}] {_label(node)!r}"


def set_value(element_id: int, text: str) -> str:
    node = _cached(element_id)
    err = AXUIElementSetAttributeValue(node, "AXValue", text)
    if err != 0:
        raise RuntimeError(f"[{element_id}] doesn't take a value (AX error {err}): click it and type instead")
    return f"Set [{element_id}] to {text[:60]!r}"


# ---- AppleScript ---------------------------------------------------------------------------

def applescript(script: str) -> tuple[str, bool]:
    """Run AppleScript (or JXA if it starts with "// JavaScript"): (output, is_error)."""
    cmd = ["osascript", "-l", "JavaScript"] if script.lstrip().startswith("// JavaScript") else ["osascript"]
    try:
        done = subprocess.run(cmd + ["-"], input=script, capture_output=True, text=True,
                              timeout=APPLESCRIPT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"timed out after {APPLESCRIPT_TIMEOUT_S} s (a dialog may be waiting)", True
    if done.returncode != 0:
        return done.stderr.strip()[-2000:] or f"exit {done.returncode}", True
    return done.stdout.strip()[-4000:] or "OK (no output)", False
