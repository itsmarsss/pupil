"""Clicks as free calibration labels.

People look at what they click, just before clicking. Samples from shortly before each
click are labelled with the click position, but only when they pass a trust gate, so
clicks made without looking don't pull the fit off course.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..core.types import Sample, TrackingState

WINDOW_START_MS = 300  # look-back window before the click
WINDOW_END_MS = 40


@dataclass(frozen=True)
class ClickGate:
    max_error_deg: float = 6.0  # prediction must already be roughly right
    min_samples: int = 3
    min_fixation_share: float = 0.6  # most samples in one fixation


@dataclass(frozen=True)
class ClickDecision:
    accepted: bool
    reason: str
    features: np.ndarray | None = None
    error_deg: float | None = None
    measures: np.ndarray | None = None  # eyeball-model measurements of the same frames


def window(t_click_ns: int) -> tuple[int, int]:
    return t_click_ns - WINDOW_START_MS * 1_000_000, t_click_ns - WINDOW_END_MS * 1_000_000


def evaluate_click(samples: list[Sample], click_xy, pt_per_deg: float, gate: ClickGate = ClickGate()) -> ClickDecision:
    usable = [
        s for s in samples
        if s.calib is not None and s.face is not None and not s.face.is_blinking
    ]
    if len(usable) < gate.min_samples:
        return ClickDecision(False, "no_face")

    tracked = [s for s in usable if s.point.state is TrackingState.TRACKING]
    if len(tracked) < gate.min_samples:
        # not calibrated yet: there's no prediction to check against
        return ClickDecision(False, "uncalibrated")

    ids = Counter(s.point.fixation_id for s in tracked)
    fixation, count = ids.most_common(1)[0]
    if fixation is None or count / len(tracked) < gate.min_fixation_share:
        return ClickDecision(False, "not_fixating")

    in_fix = [s for s in tracked if s.point.fixation_id == fixation]
    pred = np.mean([[s.point.raw_x, s.point.raw_y] for s in in_fix], axis=0)
    error_deg = float(np.hypot(*(pred - np.asarray(click_xy)))) / pt_per_deg
    if error_deg > gate.max_error_deg:
        return ClickDecision(False, "too_far", error_deg=error_deg)

    features = np.stack([s.calib for s in in_fix])
    measures = np.stack([s.geo for s in in_fix]) if all(s.geo is not None for s in in_fix) else None
    return ClickDecision(True, "ok", features, error_deg, measures)
