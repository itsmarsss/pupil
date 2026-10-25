"""Writes every gaze sample to CSV. Always the unsnapped gaze."""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

from ..core.types import Sample

FIELDS = [
    "t_ns", "wall_time", "state", "x", "y", "raw_x", "raw_y", "fixation_id", "distance_mm",
    "eye_open_l", "eye_open_r", "yaw", "pitch", "roll", "event",
]


class PupilRecorder:
    """Attach `on_sample` as a tracker listener. Events (e.g. clicks) go in the same file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self._lock = threading.Lock()
        self.rows = 0
        # monotonic → wall clock, so rows line up with other logs
        self._offset = time.time() - time.monotonic_ns() / 1e9

    def _wall(self, t_ns: int) -> float:
        return round(self._offset + t_ns / 1e9, 4)

    def on_sample(self, s: Sample) -> None:
        p, f = s.point, s.face
        row = {
            "t_ns": s.t_ns, "wall_time": self._wall(s.t_ns), "state": p.state.value,
            "x": _r(p.x), "y": _r(p.y), "raw_x": _r(p.raw_x), "raw_y": _r(p.raw_y),
            "fixation_id": "" if p.fixation_id is None else p.fixation_id,
            "distance_mm": _r(p.distance_mm), "event": "",
        }
        if f is not None:
            row["eye_open_l"], row["eye_open_r"] = (round(v, 3) for v in f.eye_openness)
            if f.head_pose is not None:
                row["yaw"], row["pitch"], row["roll"] = (round(v, 1) for v in f.head_pose.yaw_pitch_roll_deg)
        self._write(row)

    def event(self, name: str, x: float | None = None, y: float | None = None) -> None:
        t = time.monotonic_ns()
        self._write({"t_ns": t, "wall_time": self._wall(t), "state": "event", "x": _r(x), "y": _r(y), "event": name})

    def _write(self, row: dict) -> None:
        with self._lock:
            if self._file.closed:
                return
            self._writer.writerow(row)
            self.rows += 1

    def close(self) -> None:
        with self._lock:
            self._file.close()


def _r(v):
    if v is None or v != v:  # None or NaN
        return ""
    return round(float(v), 2)
