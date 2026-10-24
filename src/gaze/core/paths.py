"""Where models, profiles and logs live."""
from __future__ import annotations

import os
from pathlib import Path

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Gaze"
_REPO_MODELS = Path(__file__).resolve().parents[3] / "models"

GAZE_MODEL_FILE = "gaze_base.mnn"
FACE_LANDMARKER_FILE = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def models_dir() -> Path:
    """`$GAZE_MODELS_DIR`, else `models/` in a source checkout, else Application Support."""
    env = os.environ.get("GAZE_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    if _REPO_MODELS.is_dir():
        return _REPO_MODELS
    return APP_SUPPORT / "models"


def model_path(name: str) -> Path:
    return models_dir() / name


def ensure_face_landmarker() -> Path:
    """Return the MediaPipe face landmarker, downloading it on first use."""
    path = model_path(FACE_LANDMARKER_FILE)
    if path.exists() and path.stat().st_size > 0:
        return path
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, tmp)
    tmp.rename(path)
    return path


def profiles_dir() -> Path:
    return APP_SUPPORT / "profiles"


def recordings_dir() -> Path:
    return APP_SUPPORT / "recordings"
