"""Re-run the pipeline on a recorded session and score variants side by side."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..calibration.regression import Ridge, candidates, fit_best
from ..calibration.targets import TargetTiming
from ..core.screen import Screen
from ..core.types import Frame
from ..eval.metrics import ValidationResult, score
from ..eval.session import read_session, session_meta
from ..face.tracker import FaceTracker
from ..models.base import GazeModel

VGA = (640, 480)


@dataclass
class Extracted:
    phase: str
    target: tuple[float, float]
    features: np.ndarray  # n x d
    distance_mm: float
    frames: int  # frames shown for this target (incl. ones without a face)


def extract(root: Path, model: GazeModel, mode: str, timing: TargetTiming) -> list[Extracted]:
    """Features per (phase, target) inside the accepted sample window."""
    tracker = FaceTracker()
    buckets: dict[tuple, list] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)
    dists: dict[tuple, list] = defaultdict(list)
    for sf in read_session(root):
        if not timing.accepts(sf.onset_ns, sf.t_ns):
            continue
        key = (sf.phase, sf.target, sf.onset_ns)
        counts[key] += 1
        if sf.image is None:
            continue
        image = sf.image
        if mode == "vga" and (image.shape[1], image.shape[0]) != VGA:
            image = cv2.resize(image, VGA, interpolation=cv2.INTER_AREA)
        frame = Frame(sf.t_ns, image)
        face = tracker.process(frame)
        if face is None or face.is_blinking:
            continue
        feats = model.infer(frame, face)
        if feats is not None:
            buckets[key].append(feats.vector)
            if face.distance_mm:
                dists[key].append(face.distance_mm)
    tracker.close()
    out = []
    for key, n in counts.items():
        phase, target, _ = key
        f = np.stack(buckets[key]) if buckets[key] else np.zeros((0, 0), np.float32)
        out.append(Extracted(phase, target, f, float(np.median(dists[key])) if dists[key] else np.nan, n))
    return out


def evaluate(data: list[Extracted], screen: Screen, calibrator: str = "auto") -> tuple[str, ValidationResult]:
    calib = [d for d in data if d.phase == "calib" and len(d.features)]
    valid = [d for d in data if d.phase == "valid"]
    X = np.vstack([d.features for d in calib])
    Y = np.vstack([np.tile(d.target, (len(d.features), 1)) for d in calib])
    groups = np.concatenate([np.full(len(d.features), i) for i, d in enumerate(calib)])
    w = np.ones(len(X))
    if calibrator == "auto":
        model, report = fit_best(X, Y, w, groups)
        name = f"auto→{report.chosen}"
    else:
        model = next(c for c in candidates() if c.name == calibrator).fit(X, Y, w)
        name = calibrator
    # a list, not a dict: repeated checks show the same targets
    per_target = [(d.target, model.predict(d.features) if len(d.features) else np.zeros((0, 2))) for d in valid]
    dist = float(np.nanmedian([d.distance_mm for d in data])) if data else 550.0
    return name, score(per_target, screen, dist, sum(d.frames for d in valid))


def replay(root: Path, model: GazeModel, screen: Screen | None = None, modes=("hd", "vga")) -> list[tuple[str, str, ValidationResult]]:
    meta = session_meta(root)
    if screen is None:
        w, h = meta["screen_pt"]
        mw, mh = meta.get("screen_mm", (w * 0.2, h * 0.2))
        screen = Screen(w, h, mw, mh)
    timing = TargetTiming(**meta.get("timing", {}))
    results = []
    for mode in modes:
        data = extract(root, model, mode, timing)
        for cal in ["auto", *[c.name for c in candidates()]]:
            name, res = evaluate(data, screen, cal)
            results.append((mode, name, res))
    return results
