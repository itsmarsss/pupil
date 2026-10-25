"""Small macOS integrations: overlay window behavior, frontmost app, permission checks."""

import objc
from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSStatusWindowLevel,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowSharingNone,
    NSWorkspace,
)
from ApplicationServices import AXIsProcessTrusted


def run_as_accessory():
    """No Dock icon, and we never steal focus from the app the user is working in."""
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)


def activate_self():
    NSApplication.sharedApplication().activateIgnoringOtherApps_(True)


def float_overlay(widget):
    """Keep a Qt window above everything (incl. full-screen apps) and out of screenshots."""
    window = objc.objc_object(c_void_p=int(widget.winId())).window()
    window.setLevel_(NSStatusWindowLevel)
    window.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces
        | NSWindowCollectionBehaviorStationary
        | NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    window.setSharingType_(NSWindowSharingNone)  # keeps the overlay out of what Claude sees
    window.setIgnoresMouseEvents_(True)


def frontmost_app():
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return app.localizedName() if app else "unknown"


def has_accessibility():
    return bool(AXIsProcessTrusted())


def running_apps():
    """(name, pid) of the apps in the Dock (regular apps), frontmost first."""
    from AppKit import NSApplicationActivationPolicyRegular, NSWorkspace

    workspace = NSWorkspace.sharedWorkspace()
    front = workspace.frontmostApplication()
    apps = [a for a in workspace.runningApplications()
            if a.activationPolicy() == NSApplicationActivationPolicyRegular and a.localizedName()]
    apps.sort(key=lambda a: a != front)
    return [(str(a.localizedName()), a.processIdentifier()) for a in apps]


_PAGE_SCRIPTS = (  # (how to ask, for apps that answer it): Chromium browsers, then Safari-style
    'tell application "{app}" to return (URL of active tab of front window) & "\\n" & (title of active tab of front window)',
    'tell application "{app}" to return (URL of front document) & "\\n" & (name of front document)',
)


def front_page(timeout_s: float = 1.5):
    """(title, url) of the page in the frontmost browser, or None (not a browser, or it won't say)."""
    import subprocess

    app = frontmost_app()
    for script in _PAGE_SCRIPTS:
        try:
            out = subprocess.run(["osascript", "-e", script.format(app=app.replace('"', ""))], capture_output=True,
                                 text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return None
        url, _, title = out.stdout.strip().partition("\n")
        if out.returncode == 0 and url.startswith(("http://", "https://")):
            return title.strip(), url.strip()
    return None
