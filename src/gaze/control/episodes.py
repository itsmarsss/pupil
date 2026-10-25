"""Episodes: one record per spoken command (what was heard, seen, decided and done), labelled by the user's reaction.

Records are appended to `.logs/episodes.jsonl` as they finish; labels arrive later as separate records, so the file
is append-only and safe to tail. Implicit labels:
  corrected  "stop" / "no" / "undo" within CORRECT_S of an action
  repeated   the same request again within REPEAT_S (the first attempt probably didn't work)
  confirmed / declined   a pending action (or a question) the user confirmed or cancelled
  answered   the user answered the gate's question
Episodes with no label after REPEAT_S count as ok (a question nobody answered: unanswered). `gaze episodes` reads the file back (see `load`, `replay`).
"""
from __future__ import annotations

import itertools
import json
import re
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

from . import config

MAX_AUDIO_FILES = 1000  # newest kept; ~1 s of 16 kHz mono is 32 KB
CORRECT_S = 4.0
REPEAT_S = 10.0
CORRECTIONS = {"no", "nope", "undo", "undo that", "wrong", "wrong one", "not that", "no not that", "not that one"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def element_dict(el, gaze=None) -> dict | None:
    if el is None:
        return None
    d = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(el).items()}
    if gaze:
        d["distance"] = round(el.distance(*gaze), 1)
    return d


class EpisodeLog:
    def __init__(self, path: Path | None = None, source: str = "voice"):
        self.path = path or config.LOG_DIR / "episodes.jsonl"
        self.source = source
        self._lock = threading.Lock()
        self._recent: deque = deque(maxlen=20)  # finished episodes, for reaction labels
        self._ids = itertools.count()
        self._run = time.strftime("%Y%m%d-%H%M%S")

    def begin(self, text: str, t0: float, t1: float, kind: str) -> dict:
        """A new utterance; also labels earlier episodes by how this one reacts to them."""
        ep = {"type": "episode", "id": f"{self._run}-{next(self._ids)}", "t": time.time(), "text": text,
              "t0": t0, "t1": t1, "source": self.source, "kind": kind}
        norm = _norm(text)
        with self._lock:
            recent = list(self._recent)
        for prev in reversed(recent):
            since = t0 - prev.get("t_done", prev["t"])
            if prev.get("pending") and kind in ("confirm", "cancel") and not prev.get("label"):
                self.label(prev, "confirmed" if kind == "confirm" else "declined")
                break
            if (kind == "cancel" or norm in CORRECTIONS) and since <= CORRECT_S and prev.get("acted"):
                self.label(prev, "corrected")
                break
            if norm and norm == _norm(prev["text"]) and since <= REPEAT_S and kind not in ("cancel", "confirm"):
                self.label(prev, "repeated")
                break
        return ep

    def finish(self, ep: dict) -> None:
        ep["t_done"] = time.time()
        ep["ms"] = round((ep["t_done"] - ep["t1"]) * 1000)
        with self._lock:
            self._recent.append(ep)
        self._write({k: v for k, v in ep.items() if k != "label"})

    def save_audio(self, ep: dict, samples, rate: int = 16000) -> None:
        """The command's audio as a WAV beside the log (`.logs/audio/<id>.wav`), to replay misheard or wrongly
        gated commands later. Local only: .logs is git-ignored."""
        import wave

        import numpy as np

        if samples is None or not len(samples):
            return
        folder = self.path.parent / "audio"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{ep['id']}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        ep["audio"] = str(path.relative_to(self.path.parent))
        for old in sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime)[:-MAX_AUDIO_FILES]:
            old.unlink(missing_ok=True)

    def label(self, ep: dict, label: str) -> None:
        ep["label"] = label
        self._write({"type": "label", "id": ep["id"], "label": label, "t": time.time()})

    def outcome(self, ep: dict, **fields) -> None:
        """Something that finished after the episode was written (the agent's result)."""
        ep.update(fields)
        self._write({"type": "outcome", "id": ep["id"], "t": time.time(), **fields})

    def _write(self, record: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")


# ---- reading back ------------------------------------------------------------------------

def load(path: Path | None = None) -> list[dict]:
    """Episodes with their labels and outcomes merged in, oldest first."""
    path = path or config.LOG_DIR / "episodes.jsonl"
    episodes: dict[str, dict] = {}
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "episode":
            episodes[r["id"]] = r
        elif r.get("id") in episodes:
            if r["type"] == "label":
                episodes[r["id"]]["label"] = r["label"]
            elif r["type"] == "outcome":
                episodes[r["id"]].update({k: v for k, v in r.items() if k not in ("type", "id", "t")})
    now = time.time()
    for ep in episodes.values():
        if "label" not in ep and now - ep.get("t_done", ep["t"]) > REPEAT_S:
            ep["label"] = "unanswered" if ep.get("decision") == "clarify" else "ok"
    return list(episodes.values())


class _Answer:
    def __init__(self, value, attr):
        setattr(self, attr, value)


class _Answers:
    """Logged gate answers in the shape `gate._decide` reads."""

    def __init__(self, gate: dict):
        self.nouls = {k: _Answer(v, "noul") for k, v in gate["nouls"].items()}
        self.choices = {k: _Answer(v, "choice") for k, v in gate.get("choices", {}).items()}


def replay(episodes: list[dict]) -> list[tuple[dict, str, str]]:
    """Re-run the current truth table on each episode's logged Jev answers (offline, free).
    Returns (episode, logged route, route now) for every episode that went through the gate."""
    from .gate import _decide
    from .snapping import Element

    out = []
    for ep in episodes:
        gate = ep.get("gate")
        if not gate or "nouls" not in gate:
            continue
        elements = [Element(**{k: e[k] for k in ("role", "label", "x", "y", "w", "h")}) for e in ep.get("elements", [])]
        try:
            route, _, _ = _decide(_Answers(gate), elements, tuple(ep["gaze"]) if ep.get("gaze") else None, {},
                                  **gate.get("flags", {}))
        except KeyError as e:  # logged before a question existed
            route = f"? (missing {e})"
        out.append((ep, gate["route"], route))
    return out
