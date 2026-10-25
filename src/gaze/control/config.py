import os
from pathlib import Path

from ..sinks.publisher import CMD_PORT, GAZE_PORT  # noqa: F401  (shared with the tracker)

ROOT = Path(__file__).resolve().parents[3]  # repo root: src/gaze/control/config.py
LOG_DIR = ROOT / ".logs"


def _load_dotenv(path=ROOT / ".env"):
    """Minimal .env support (KEY=value lines); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# Agent (LLM tier): only used when the fast tier can't handle a request
MODEL = os.environ.get("GAZE_MODEL", "claude-sonnet-5-5")
EFFORT = os.environ.get("GAZE_EFFORT", "low")  # low = fewer, faster steps
MAX_AGENT_STEPS = int(os.environ.get("GAZE_MAX_STEPS", "10"))

# Fast tier: a Jev-style decision model (TypeSafe System One API; any compatible base URL works)
GATE_MODEL = os.environ.get("GAZE_GATE_MODEL", "jev-1.14.0")  # pinned: aliases move and deprecations are short
GATE_BASE_URL = os.environ.get("TYPESAFE_BASE_URL")  # None = api.typesafe.ai; e.g. OpenRouter / Vercel AI Gateway
GATE_TIMEOUT_S = 2.0
# Yes/no thresholds for Jev's judgments live in gate.py (YES / NO) next to the truth table.
GATE_RISKY_MIN = 0.65  # ask to confirm at or above this (measured: harmless clicks ~0.4-0.46, delete/send 0.84-0.97)

# The tracker (`gaze --publish`, sinks/publisher.py) streams points to us over local UDP
GAZE_STALE_S = 1.0  # no point for this long = tracker lost you
GAZE_MIN_CONFIDENCE = 0.5  # below this (head outside its calibrated range) clicks and typing ask to confirm

# Voice: Cartesia Ink-2 streaming speech-to-text when CARTESIA_API_KEY is set; on-device Whisper otherwise or offline
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")
STT = os.environ.get("GAZE_STT", "cartesia" if CARTESIA_API_KEY else "whisper")  # cartesia | whisper
CARTESIA_MODEL = os.environ.get("GAZE_CARTESIA_MODEL", "ink-2")
# How sure Ink-2 must be that you're done before ending the turn (0.05-0.5, Cartesia's default 0.2). 0.5 = end
# soonest: measured ~0.95 s after speech vs ~1.45 s at 0.2; a long mid-sentence pause may split a request.
CARTESIA_END_THRESHOLD = os.environ.get("GAZE_CARTESIA_END_THRESHOLD", "0.5")
_LOCAL_WHISPER = ROOT / "models" / "whisper-base.en"  # downloaded with curl; HF's downloader stalls on bad wifi
WHISPER_MODEL = os.environ.get(
    "GAZE_WHISPER", str(_LOCAL_WHISPER) if (_LOCAL_WHISPER / "model.bin").exists() else "base.en")
MIC_NAME = os.environ.get("GAZE_MIC", "")  # force an input by name; empty = follow the macOS default (switches live,
# e.g. to AirPods), skipping silent virtual devices like "Virtual Desktop Mic" / "BlackHole"
VOICE_PROCESSING = os.environ.get("GAZE_VOICE_PROCESSING", "1") != "0"  # Apple's noise suppression + echo cancel
# on the system default input (not available for a mic picked by name)
LIP_GATE = os.environ.get("GAZE_LIP_GATE", "1") != "0"  # ignore speech while the camera sees your lips still
LIP_MOVING = 0.04  # p10-p90 range of mouth openness (inner-lip gap / width) that counts as talking. A first,
# deliberately low guess: episodes log the range for every utterance (`gaze episodes`), so tune it from those.
# The gate only rejects when the camera saw the face for most of the utterance.
WAKE_WORD = os.environ.get("GAZE_WAKE_WORD", "").strip().lower()  # e.g. "gaze"; empty = every utterance is a command

# Snapping
SNAP_RADIUS = 120  # points; how far from the gaze we look for a target
SNAP_MAX_AREA_FRACTION = 0.15  # ignore elements bigger than this share of the screen (windows, web areas)

# Screenshots sent to Claude (recommended computer-use size; coordinates are scaled back to points)
SCREENSHOT_MAX = (1280, 800)
