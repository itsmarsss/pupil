"""Whether Pupil's on-screen overlays (notch, gaze blob, status pill, target boxes) show up in screen recordings.

They do by default, so a QuickTime or OBS recording of a demo shows what Pupil is doing. The agent still never sees
them: its screenshots are taken of every window except Pupil's own (control/screen.py). PUPIL_HIDE_FROM_RECORDINGS=1
hides them from every capture instead, as before.
"""
from __future__ import annotations

from .env import env


def overlay_sharing():
    """The NSWindow sharing type for an overlay window."""
    from AppKit import NSWindowSharingNone, NSWindowSharingReadOnly

    return NSWindowSharingNone if env("HIDE_FROM_RECORDINGS", "0") == "1" else NSWindowSharingReadOnly
