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
MAX_AGENT_STEPS = int(os.environ.get("GAZE_MAX_STEPS", "30"))  # room for bigger multi-step tasks
# Jev routes each agent task to a tier (gate.AGENT_TIER) and a step budget (gate.AGENT_BUDGET). A tier is a model and
# an effort; the model must support the computer toolset (Haiku 4.5 doesn't; Sonnet 5.5 and Opus 5.5 do).
AGENT_TIERS = {
    "fast": (os.environ.get("GAZE_MODEL_FAST", MODEL), "low"),
    "standard": (os.environ.get("GAZE_MODEL_STANDARD", MODEL), "medium"),
    "deep": (os.environ.get("GAZE_MODEL_DEEP", "claude-opus-5-5"), "medium"),
}
AGENT_BUDGETS = {"quick": 10, "normal": MAX_AGENT_STEPS, "ongoing": int(os.environ.get("GAZE_MAX_STEPS_ONGOING", "300"))}

# Fast tier: a Jev-style decision model (TypeSafe System One API; any compatible base URL works)
GATE_MODEL = os.environ.get("GAZE_GATE_MODEL", "jev-1.14.0")  # pinned: aliases move and deprecations are short
GATE_BASE_URL = os.environ.get("TYPESAFE_BASE_URL")  # None = api.typesafe.ai; e.g. OpenRouter / Vercel AI Gateway
GATE_TIMEOUT_S = 2.0
# Yes/no thresholds for Jev's judgments live in gate.py (YES / NO) next to the truth table.
GATE_RISKY_MIN = 0.65  # ask to confirm at or above this (measured: harmless clicks ~0.4-0.46, delete/send 0.84-0.97)

# The tracker (`gaze --publish`, sinks/publisher.py) streams points to us over local UDP
GAZE_STALE_S = 1.0  # no point for this long = tracker lost you
GAZE_MIN_CONFIDENCE = 0.5  # below this (head outside its calibrated range) clicks and typing ask to confirm

# Dictation ("type ...", dictation mode): a fast model turns the transcript into what was meant ("slash btw" ->
# "/btw" in a chat box), from the app and the text before the cursor. Off: typed as heard.
DICTATION_FORMAT = os.environ.get("GAZE_DICTATION_FORMAT", "1") != "0"
DICTATION_MODEL = os.environ.get("GAZE_DICTATION_MODEL", "claude-haiku-4-5-20251001")
DICTATION_TIMEOUT_S = 4.0  # then type it as heard

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
AVOID_BLUETOOTH_MIC = os.environ.get("GAZE_AVOID_BLUETOOTH_MIC", "1") != "0"  # following the default: when it's
# AirPods (Bluetooth), listen on the MacBook mic so they keep full-quality audio; pick AirPods in ◉ → Microphone
SAVE_AUDIO = os.environ.get("GAZE_SAVE_AUDIO", "1") != "0"  # each command's audio next to its episode (.logs/audio)
VOICE_PROCESSING = os.environ.get("GAZE_VOICE_PROCESSING", "0") == "1"  # Apple's noise suppression + echo cancel.
# Off by default: on AirPods it forces a call-style path that sounded and transcribed worse; ◉ → Voice isolation
# on the system default input (not available for a mic picked by name)
# Speaking back (speech.py): Claude's replies and questions read aloud with Cartesia Sonic; talking cuts it off
SPEAK = os.environ.get("GAZE_SPEAK", "1") != "0"
TTS_MODEL = os.environ.get("GAZE_TTS_MODEL", "sonic-3.6")
TTS_VOICE = os.environ.get("GAZE_TTS_VOICE", "db6b0ed5-d5d3-463d-ae85-518a07d3c2b4")  # Cartesia's example voice

# Lip gate (lips.py): speech counts only while you face the screen and move your lips. on | shadow (log what it
# would block, block nothing) | off. Eye tracking only; with --gaze mouse there is no face to watch.
LIP_GATE = os.environ.get("GAZE_LIP_GATE", "on")
WAKE_WORD = os.environ.get("GAZE_WAKE_WORD", "").strip().lower()  # e.g. "gaze"; empty = every utterance is a command

# Snapping
SNAP_RADIUS = 120  # points; how far from the gaze we look for a target
SNAP_MAX_AREA_FRACTION = 0.15  # ignore elements bigger than this share of the screen (windows, web areas)

# Screenshots sent to Claude (recommended computer-use size; coordinates are scaled back to points)
SCREENSHOT_MAX = (1280, 800)
