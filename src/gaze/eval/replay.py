"""Re-run the pipeline on a recorded session and score variants side by side."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..calibration.features import POSE_DIM, PoseRange, calibration_vector
from ..calibration.profile import WEIGHTS
from ..calibration.regression import candidates, fit_best
from ..calibration.targets import TargetTiming
from ..core.screen import Screen
from ..core.types import Frame
from ..eval.metrics import ValidationResult, score
from ..eval.session import read_session, session_meta
from ..face.tracker import FaceTracker
from ..models.base import GazeModel

VGA = (640, 480)
MOTION_CHUNK = 15

# Calibration variants: which phases to fit on, and whether to use head-pose features.
CONFIGS = {
    "dots": (("calib",), False),
    "dots+pose": (("calib",), True),
    "dots+pose+motion": (("calib", "motion"), True),
}


@dataclass
class Extracted:
    phase: str
    target: tuple[float, float]
    features: np.ndarray  # n x (model dim + POSE_DIM) calibration vectors
    distance_mm: float
    frames: int  # frames shown in the window (incl. ones without a face)


def extract(root: Path, model: GazeModel, mode: str, timings: dict[str, TargetTiming]) -> list[Extracted]:
    """Calibration vectors per shown target, inside each phase's accepted window."""
    tracker = FaceTracker()
    buckets: dict[tuple, list] = defaultdict(list)
    counts: dict[tuple, int] = defaultdict(int)
    dists: dict[tuple, list] = defaultdict(list)
    for sf in read_session(root):
        timing = timings.get(sf.phase, timings["calib"])
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
            buckets[key].append(calibration_vector(feats.vector, face))
            if face.distance_mm:
                dists[key].append(face.distance_mm)
    tracker.close()
    out = []
    for key, n in sorted(counts.items(), key=lambda kv: kv[0][2]):
        phase, target, _ = key
        f = np.stack(buckets[key]) if buckets[key] else np.zeros((0, 0), np.float32)
        out.append(Extracted(phase, target, f, float(np.median(dists[key])) if dists[key] else np.nan, n))
    return out


def evaluate(data: list[Extracted], screen: Screen, config: str = "dots+pose+motion",
             calibrator: str = "auto", valid_phase: str | None = None) -> tuple[str, ValidationResult, ValidationResult | None]:
    """Fit on the config's phases, score on validation. Also returns the score over only
    the validation targets where the head was inside the calibrated pose range."""
    phases, use_pose = CONFIGS[config]
    fit_on = [d for d in data if d.phase in phases and len(d.features)]
    valid = [d for d in data if d.phase.startswith("valid") and valid_phase in (None, d.phase)]

    X = np.vstack([d.features for d in fit_on])
    Y = np.vstack([np.tile(d.target, (len(d.features), 1)) for d in fit_on])
    groups = np.concatenate([np.full(len(d.features), i) for i, d in enumerate(fit_on)])
    subs = np.concatenate([
        i * 100_000 + (np.arange(len(d.features)) // MOTION_CHUNK if d.phase == "motion" else np.zeros(len(d.features), int))
        for i, d in enumerate(fit_on)
    ])
    w = np.concatenate([np.full(len(d.features), WEIGHTS["motion" if d.phase == "motion" else "explicit"]) for d in fit_on])
    if calibrator == "auto":
        model, report = fit_best(X, Y, w, groups, POSE_DIM, score_groups=subs, use_pose=use_pose)
        name = f"auto→{report.chosen}"
    else:
        model = next(c for c in candidates(POSE_DIM, use_pose) if c.name == calibrator).fit(X, Y, w)
        name = calibrator

    def points(d):
        if not len(d.features):
            return np.zeros((0, 2))
        return np.array([screen.clamp(x, y) for x, y in model.predict(d.features)])

    dist = float(np.nanmedian([d.distance_mm for d in data])) if data else 550.0
    pairs = [(d.target, points(d)) for d in valid]  # a list: repeated checks reuse targets
    overall = score(pairs, screen, dist, sum(d.frames for d in valid))

    pose_range = PoseRange(np.vstack([d.features for d in data if d.phase in ("calib", "motion") and len(d.features)]))
    inside = [(d.target, points(d)) for d in valid
              if len(d.features) and np.median([pose_range.outside_deg(f) for f in d.features]) <= PoseRange.MARGIN_DEG]
    in_range = score(inside, screen, dist, sum(len(p) for _, p in inside)) if inside else None
    return name, overall, in_range


def load(root: Path, screen: Screen | None = None):
    meta = session_meta(root)
    if screen is None:
        w, h = meta["screen_pt"]
        mw, mh = meta.get("screen_mm", (w * 0.2, h * 0.2))
        screen = Screen(w, h, mw, mh)
    timings = {"calib": TargetTiming(**meta.get("timing", {})), "valid": TargetTiming(**meta.get("timing", {}))}
    if "motion_timing" in meta:
        timings["motion"] = TargetTiming(**meta["motion_timing"])
    return screen, timings


def replay(root: Path, model: GazeModel, screen: Screen | None = None, modes=("hd", "vga"),
           all_calibrators: bool = False):
    """Rows of (mode, config, calibrator, validation phase, result, result with head in range).
    Phase "all" pools every check; sessions from the measure routine also get
    "valid_still" and "valid_moving" rows."""
    screen, timings = load(root, screen)
    rows = []
    for mode in modes:
        data = extract(root, model, mode, timings)
        has_motion = any(d.phase == "motion" for d in data)
        valid_phases = sorted({d.phase for d in data if d.phase.startswith("valid")})
        phases = [None] + (valid_phases if len(valid_phases) > 1 else [])
        for config, (_, use_pose) in CONFIGS.items():
            if "motion" in config and not has_motion:
                continue
            cals = ["auto"] + ([c.name for c in candidates(POSE_DIM, use_pose)] if all_calibrators else [])
            for cal in cals:
                for vp in phases:
                    name, overall, in_range = evaluate(data, screen, config, cal, vp)
                    rows.append((mode, config, name, vp or "all", overall, in_range))
    return rows
