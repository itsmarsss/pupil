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
