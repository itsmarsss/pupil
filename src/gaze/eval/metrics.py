"""Accuracy and precision of gaze against known targets, in degrees of visual angle."""
from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..core.screen import Screen


@dataclass(frozen=True)
class TargetResult:
    target: tuple[float, float]
    n: int
    offset_pt: float  # distance from target to the mean gaze point
    offset_deg: float
    rms_s2s_deg: float  # sample-to-sample RMS: jitter
    std_deg: float  # spread around the mean


@dataclass(frozen=True)
class ValidationResult:
    accuracy_deg: float  # median of per-target offsets
    accuracy_p95_deg: float
    precision_rms_deg: float  # median per-target RMS-S2S
    accuracy_pt: float
    data_loss: float  # fraction of frames without a valid point
    distance_mm: float
    targets: list[TargetResult]

    def summary(self) -> str:
        return (
            f"accuracy {self.accuracy_deg:.2f}° (p95 {self.accuracy_p95_deg:.2f}°, {self.accuracy_pt:.0f} pt) · "
            f"precision {self.precision_rms_deg:.2f}° · loss {self.data_loss:.0%} · "
            f"distance {self.distance_mm / 10:.0f} cm"
        )


def score_target(target, points: np.ndarray, screen: Screen, distance_mm: float) -> TargetResult:
    """`points` is n x 2 screen points recorded while the target was shown."""
    if len(points) == 0:
        nan = float("nan")
        return TargetResult(tuple(target), 0, nan, nan, nan, nan)
    mean = points.mean(axis=0)
    offset = float(np.hypot(*(mean - np.asarray(target))))
    steps = np.hypot(*np.diff(points, axis=0).T) if len(points) > 1 else np.zeros(1)
    rms = float(np.sqrt(np.mean(steps**2)))
    spread = float(np.sqrt(np.mean(np.sum((points - mean) ** 2, axis=1))))
    to_deg = lambda pt: screen.pt_to_deg(pt, distance_mm)  # noqa: E731
    return TargetResult(tuple(target), len(points), offset, to_deg(offset), to_deg(rms), to_deg(spread))


def score(per_target: dict, screen: Screen, distance_mm: float, frames_total: int) -> ValidationResult:
    """`per_target` maps target (x, y) → n x 2 array of gaze points."""
    results = [score_target(t, np.asarray(p, float).reshape(-1, 2), screen, distance_mm) for t, p in per_target.items()]
    valid = [r for r in results if r.n > 0]
    offsets = np.array([r.offset_deg for r in valid]) if valid else np.array([math.nan])
    n_valid = sum(r.n for r in results)
    return ValidationResult(
        accuracy_deg=float(np.median(offsets)),
        accuracy_p95_deg=float(np.percentile(offsets, 95)),
        precision_rms_deg=float(np.median([r.rms_s2s_deg for r in valid])) if valid else math.nan,
        accuracy_pt=float(np.median([r.offset_pt for r in valid])) if valid else math.nan,
        data_loss=1 - n_valid / frames_total if frames_total else 1.0,
        distance_mm=distance_mm,
        targets=results,
    )


def append_result(path: Path, result: ValidationResult, **context) -> None:
    """Append one row to a CSV log of validation runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **context}
    row.update({k: v for k, v in asdict(result).items() if k != "targets"})
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
