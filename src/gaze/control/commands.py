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
    "refresh": ("Reload / refresh the page", "command+r"),
    "address_bar": ("Focus the browser address bar to type a URL or search", "command+l"),
    "zoom_in": ("Zoom in / make things bigger", "command+equal"),
    "zoom_out": ("Zoom out / make things smaller", "command+minus"),
    "actual_size": ("Reset zoom to actual size", "command+0"),
    "minimize": ("Minimize the current window", "command+m"),
    "full_screen": ("Toggle full screen for the current window", "control+command+f"),
    "hide_app": ("Hide the current app", "command+h"),
    "quit_app": ("Quit the current app", "command+q"),
    "switch_app": ("Switch to the previously used app", "command+tab"),
    "spotlight": ("Open Spotlight search", "command+space"),
    "mission_control": ("Show Mission Control (all windows)", "control+up"),
    "escape": ("Press Escape / cancel / close a popup", "escape"),
    "enter": ("Press Enter / Return to confirm or submit", "return"),
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
