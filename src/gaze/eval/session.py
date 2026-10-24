"""Recorded calibration/validation sessions for offline replay.

Each frame shown while a target is on screen is saved as a JPEG of the region around
the face (not the whole frame), with its target and timing, so pipeline variants can be
compared on identical data.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..core.types import Frame, Sample

ROI_SCALE = 1.8  # crop around the face, generous enough to re-detect it


def face_roi(sample: Sample, w: int, h: int) -> tuple[int, int, int, int] | None:
    if sample.face is None:
        return None
    x, y, fw, fh = sample.face.face_rect
    cx, cy, half = x + fw / 2, y + fh / 2, max(fw, fh) * ROI_SCALE / 2
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    return x0, y0, x1 - x0, y1 - y0


class SessionWriter:
    """Tracker frame listener. The app sets `label(phase, target, onset_ns)` while a
    target is shown and `label(None)` otherwise; unlabelled frames aren't saved."""

    def __init__(self, root: Path, meta: dict):
        self.root = Path(root)
        (self.root / "frames").mkdir(parents=True, exist_ok=True)
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        self._rows = (self.root / "labels.jsonl").open("a")
        self._lock = threading.Lock()
        self._label: tuple[str, tuple[float, float], int] | None = None
        self.frames = 0

    def label(self, phase: str | None, target=None, onset_ns: int = 0) -> None:
        with self._lock:
            self._label = None if phase is None else (phase, tuple(map(float, target)), onset_ns)

    def on_frame(self, frame: Frame, sample: Sample) -> None:
        with self._lock:
            label = self._label
        if label is None:
            return
        roi = face_roi(sample, frame.width, frame.height)
        name = None
        if roi is not None:
            x, y, w, h = roi
            name = f"{frame.t_ns}.jpg"
            bgr = cv2.cvtColor(frame.image[y : y + h, x : x + w], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(self.root / "frames" / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
        phase, target, onset = label
        row = {"t_ns": frame.t_ns, "frame": [frame.width, frame.height], "roi": roi, "image": name,
               "phase": phase, "target": target, "onset_ns": onset}
        with self._lock:
            if not self._rows.closed:
                self._rows.write(json.dumps(row) + "\n")
                self.frames += 1

    def close(self) -> None:
        with self._lock:
            self._rows.close()


@dataclass
class SessionFrame:
    t_ns: int
    phase: str
    target: tuple[float, float]
    onset_ns: int
    image: np.ndarray | None  # full-size RGB canvas with the face region pasted in


def read_session(root: Path):
    """Yields SessionFrame in time order, rebuilding each frame on a black canvas."""
    root = Path(root)
    rows = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines() if line.strip()]
    for row in sorted(rows, key=lambda r: r["t_ns"]):
        canvas = None
        if row["image"]:
            bgr = cv2.imread(str(root / "frames" / row["image"]))
            if bgr is not None:
                w, h = row["frame"]
                x, y, rw, rh = row["roi"]
                canvas = np.zeros((h, w, 3), np.uint8)
                canvas[y : y + rh, x : x + rw] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[:rh, :rw]
        yield SessionFrame(row["t_ns"], row["phase"], tuple(row["target"]), row["onset_ns"], canvas)


def session_meta(root: Path) -> dict:
    return json.loads((Path(root) / "meta.json").read_text())
