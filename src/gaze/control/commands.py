"""Things the fast tier can do without an on-screen target: open apps and press standard shortcuts.

Both are enumerated here so the decision model only ever picks from a fixed list.
"""

import subprocess
import time
from pathlib import Path

from AppKit import NSApplicationActivationPolicyRegular, NSWorkspace

APP_DIRS = ["/Applications", "/Applications/Utilities", "/System/Applications",
            "/System/Applications/Utilities", str(Path.home() / "Applications")]
MAX_APPS = 250  # a Choice question takes up to 255 options

# id -> (what the model sees, key combo for actions.press)
SHORTCUTS = {
    "copy": ("Copy the selection", "command+c"),
    "paste": ("Paste", "command+v"),
    "cut": ("Cut the selection", "command+x"),
    "undo": ("Undo the last change", "command+z"),
    "redo": ("Redo", "command+shift+z"),
    "select_all": ("Select all", "command+a"),
    "save": ("Save the current document", "command+s"),
    "find": ("Find / search within the page or document", "command+f"),
    "new_tab": ("Open a new tab", "command+t"),
    "close_tab": ("Close the current tab", "command+w"),
    "reopen_tab": ("Reopen the last closed tab", "command+shift+t"),
    "next_tab": ("Go to the next tab", "control+tab"),
    "previous_tab": ("Go to the previous tab", "control+shift+tab"),
    "new_window": ("Open a new window", "command+n"),
    "private_window": ("Open a new private / incognito window", "command+shift+n"),
    "back": ("Go back to the previous page", "command+["),
    "forward": ("Go forward a page in the browser or file history", "command+]"),
    "refresh": ("Reload / refresh the page", "command+r"),
    "address_bar": ("Focus the browser address bar to type a URL or search", "command+l"),
    "zoom_in": ("Zoom in / make things bigger", "command+equal"),
    "zoom_out": ("Zoom out / make things smaller", "command+minus"),
    "actual_size": ("Reset zoom to actual size", "command+0"),
    "minimize": ("Minimize the current window", "command+m"),
    "full_screen": ("Toggle full screen for the current window", "control+command+f"),
    "hide_app": ("Hide the current app", "command+h"),
    "quit_app": ("Quit the app that's in front right now (not another app the command names)", "command+q"),
    "switch_app": ("Switch to the previously used app", "command+tab"),
    "spotlight": ("Open Spotlight search", "command+space"),
    "mission_control": ("Show Mission Control (all windows)", "control+up"),
    "escape": ("Press Escape / cancel / close a popup", "escape"),
    "enter": ("Press Enter / Return to confirm or submit", "return"),
    # plain keys: games, lists, sliders, text, media (repeatable)
    "key_left": ("Press the Left arrow key: move something left (a game piece, a selection, a slider), "
                 "including \"all the way left\"", "left"),
    "key_right": ("Press the Right arrow key: move something right (a game piece, a selection, a slider), "
                  "including \"all the way right\"", "right"),
    "key_up": ("Press the Up arrow key (move up, rotate in some games, previous item)", "up"),
    "key_down": ("Press the Down arrow key (move down, next item, drop in some games)", "down"),
    "key_space": ("Press Space (play/pause, jump, drop or place in a game, toggle a checkbox)", "space"),
    "key_tab": ("Press Tab (next field)", "tab"),
    "key_page_down": ("Press Page Down", "pagedown"),
    "key_page_up": ("Press Page Up", "pageup"),
    "key_home": ("Press Home: jump to the start of a line or document", "home"),
    "key_end": ("Press End: jump to the end of a line or document", "end"),
    "key_delete": ("Press Delete / Backspace (remove the character or selected item)", "backspace"),
}
ALWAYS_CONFIRM = {"quit_app"}  # loses everything the app had open: ask, whatever the risk score says
REPEATABLE = {k for k in SHORTCUTS if k.startswith("key_")} | {"next_tab", "previous_tab", "back", "forward",
                                                               "undo", "redo", "zoom_in", "zoom_out"}

# The same action is a menu command in nearly every Mac app, under slightly different names and shortcuts. Run it
# from the frontmost app's own menu (uitree.press_menu_command) and fall back to the keys above only if it has
# none. Each phrase matches a menu title containing all its words; tried in order.
MENU_PHRASES = {
    "new_tab": ["new tab"], "close_tab": ["close tab", "close current tab", "close editor"],
    "reopen_tab": ["reopen closed tab", "reopen tab", "reopen closed editor"],
    "next_tab": ["next tab", "next editor", "tab to the right"],
    "previous_tab": ["previous tab", "previous editor", "tab to the left"],
    "new_window": ["new window"], "private_window": ["new private window", "new incognito window"],
    "refresh": ["reload this page", "reload page", "reload", "refresh"],
    "back": ["back"], "forward": ["forward"], "find": ["find"], "save": ["save"],
    "zoom_in": ["zoom in"], "zoom_out": ["zoom out"], "actual_size": ["actual size", "reset zoom"],
    "full_screen": ["enter full screen", "toggle full screen", "full screen"], "minimize": ["minimize"],
    "address_bar": ["open location"], "select_all": ["select all"],
}


def _app_paths():
    found = {}
    for d in APP_DIRS:
        for app in sorted(Path(d).glob("*.app")) + sorted(Path(d).glob("*/*.app")):
            found.setdefault(app.stem, str(app))
    return found


_installed = {}
_installed_at = 0.0


def available_apps():
    """{name: path}: running apps first (most likely targets), then installed ones."""
    global _installed, _installed_at
    if time.time() - _installed_at > 60:  # a directory scan is cheap, but no need to repeat it every command
        _installed, _installed_at = _app_paths(), time.time()
    apps = {}
    for app in NSWorkspace.sharedWorkspace().runningApplications():
        url = app.bundleURL()
        if app.activationPolicy() == NSApplicationActivationPolicyRegular and app.localizedName() and url:
            apps.setdefault(str(app.localizedName()), str(url.path()))
    for name, path in _installed.items():
        if len(apps) >= MAX_APPS:
            break
        apps.setdefault(name, path)
    return apps


def open_app(path):
    """Launch the app, or bring it to the front if it's already running."""
    subprocess.Popen(["open", "-a", path])
