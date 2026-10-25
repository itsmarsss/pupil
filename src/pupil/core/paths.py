"""Where models, profiles and logs live."""
from __future__ import annotations

from pathlib import Path
from .env import env

APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Pupil"
_OLD_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Gaze"  # before the rename


def migrate_app_support() -> None:
    """Legacy Application Support folder renamed to Pupil once (same disk, nothing copied); never overwrites Pupil."""
    if _OLD_APP_SUPPORT.is_dir() and not APP_SUPPORT.exists():
        _OLD_APP_SUPPORT.rename(APP_SUPPORT)
_REPO_MODELS = Path(__file__).resolve().parents[3] / "models"

PUPIL_GAZE_MODEL = "gaze_base.mnn"  # on-disk weights for the eye-gaze network
FACE_LANDMARKER_FILE = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def models_dir() -> Path:
    """`$PUPIL_MODELS_DIR`, else `models/` in a source checkout, else Application Support."""
    chosen = env("MODELS_DIR")
    if chosen:
        return Path(chosen).expanduser()
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
