"""Settings from the environment as PUPIL_<NAME>."""
from __future__ import annotations

import os


def env(name: str, default=None):
    return os.environ.get(f"PUPIL_{name}", default)
