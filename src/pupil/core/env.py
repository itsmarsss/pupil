"""Settings from the environment: PUPIL_<NAME>, or the older GAZE_<NAME> (the app was called Gaze)."""
from __future__ import annotations

import os


def env(name: str, default=None):
    return os.environ.get(f"PUPIL_{name}", os.environ.get(f"GAZE_{name}", default))
