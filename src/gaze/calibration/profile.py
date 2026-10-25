"""A user's calibration samples and fitted mapping, persisted between launches."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core.paths import profiles_dir
from ..geometry.calibrate import GeoCalibration
from ..geometry.calibrate import fit as fit_geometry
from ..geometry.eyeball import MEASURE_DIM, ScreenPlane, from_array
from .features import FEATURE_SET, POSE_DIM, PoseRange
from .regression import FitReport, Ridge, fit_best

log = logging.getLogger(__name__)

EXPLICIT, MOTION, CLICK = "explicit", "motion", "click"
# Per-sample weights. A head-motion sweep has ~6x the samples of one dot, so 0.5 makes
# it count like a few dots: enough to learn head compensation without dominating.
WEIGHTS = {EXPLICIT: 1.0, MOTION: 0.5, CLICK: 0.5}
# Final point = (1 - GEO_WEIGHT) x learned calibrator + GEO_WEIGHT x eyeball model. On 138 replayed validation
# dots this beat the learned calibrator alone on two thirds of them (mean 2.52° → 2.22°).
GEO_WEIGHT = 0.25


@dataclass
class ProfileMeta:
    name: str
    model_id: str
    camera: str
    screen_pt: tuple[float, float]
    features: str = FEATURE_SET
    lid_deg: float | None = None  # hinge angle at calibration, if the Mac has the sensor
    offset_pt: tuple[float, float] = (0.0, 0.0)  # drift correction added to every prediction
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    fit: dict = field(default_factory=dict)

    def matches(self, model_id: str, camera: str, screen_pt: tuple[float, float]) -> bool:
        return (
            self.model_id == model_id
            and self.camera == camera
            and self.features == FEATURE_SET
            and tuple(round(v) for v in self.screen_pt) == tuple(round(v) for v in screen_pt)
        )


class Profile:
    """Calibration samples (features → screen point) with their source and group.

    A group is one target: a calibration dot, the head-motion sweep, or one click.
    `subs` splits a group more finely for scoring (chunks of the head-motion sweep).
    Explicit and motion samples are kept; click samples are capped, oldest evicted first.
    """

    MAX_CLICK_GROUPS = 400

    def __init__(self, meta: ProfileMeta, dim: int):
        self.meta = meta
        self.X = np.zeros((0, dim), np.float32)
        self.M = np.zeros((0, MEASURE_DIM), np.float32)  # eyeball-model measurements of the same frames (NaN = none)
        self.Y = np.zeros((0, 2), np.float32)
        self.groups = np.zeros(0, np.int64)
        self.subs = np.zeros(0, np.int64)
        self.sources = np.zeros(0, "<U8")
        self.t = np.zeros(0, np.float64)
        self.model: Ridge | None = None
        self.report: FitReport | None = None
        self.pose_range: PoseRange | None = None
        self.geometry: GeoCalibration | None = None
        self.plane: ScreenPlane | None = None  # set by the app; needed to fit the eyeball model
        self._next_group = 0
        self._lock = threading.RLock()  # samples are added on the UI thread, fits run in the background

    @property
    def pose_dim(self) -> int:
        return POSE_DIM if self.meta.features == FEATURE_SET else 0

    # ---- samples -------------------------------------------------------------------
    def new_group(self) -> int:
        with self._lock:
            g = self._next_group
            self._next_group += 1
            return g

    def add(self, X: np.ndarray, target: tuple[float, float], source: str, group: int | None = None,
            chunk: int = 1, measures: np.ndarray | None = None) -> int:
        """Add samples for one target. `chunk` > 1 splits them into runs of that many
        samples for scoring (used for the head-motion sweep). `measures`: flattened eyeball-model
        measurements of the same frames."""
        X = np.atleast_2d(np.asarray(X, np.float32))
        M = (np.full((len(X), MEASURE_DIM), np.nan, np.float32) if measures is None
             else np.atleast_2d(np.asarray(measures, np.float32)))
        with self._lock:
            group = self.new_group() if group is None else group
            n = len(X)
            subs = group * 100_000 + (np.arange(n) // chunk if chunk > 1 else np.zeros(n, np.int64))
            self.X = np.vstack([self.X, X])
            self.M = np.vstack([self.M, M])
            self.Y = np.vstack([self.Y, np.tile(np.asarray(target, np.float32), (n, 1))])
            self.groups = np.concatenate([self.groups, np.full(n, group)])
            self.subs = np.concatenate([self.subs, subs])
            self.sources = np.concatenate([self.sources, np.full(n, source)])
            self.t = np.concatenate([self.t, np.full(n, time.time())])
            if source == CLICK:
                self._evict_clicks()
        return group

    def clear(self, *sources: str) -> None:
        """Remove samples from the given sources (all samples if none given)."""
        with self._lock:
            keep = ~np.isin(self.sources, sources) if sources else np.zeros(len(self.X), bool)
            self._keep(keep)

    def _keep(self, keep: np.ndarray) -> None:
        self.X, self.M, self.Y, self.groups, self.subs = self.X[keep], self.M[keep], self.Y[keep], self.groups[keep], self.subs[keep]
        self.sources, self.t = self.sources[keep], self.t[keep]

    def _evict_clicks(self) -> None:
        click_groups = np.unique(self.groups[self.sources == CLICK])
        excess = len(click_groups) - self.MAX_CLICK_GROUPS
        if excess > 0:
            self._keep(~np.isin(self.groups, click_groups[:excess]))

    @property
    def n_groups(self) -> int:
        return len(np.unique(self.groups))

    def count(self, source: str) -> int:
        return len(np.unique(self.groups[self.sources == source]))

    # ---- fitting -------------------------------------------------------------------
    def fit(self) -> FitReport | None:
        with self._lock:
            if self.count(EXPLICIT) < 3:
                return None
            X, M, Y, groups, subs, sources = self.X, self.M, self.Y, self.groups, self.subs, self.sources
        w = np.array([WEIGHTS[s] for s in sources])
        # Pose terms only help once the head has moved while gaze stayed put.
        use_pose = bool(self.pose_dim) and bool(np.any(sources == MOTION))
        model, report = fit_best(X, Y, w, groups, self.pose_dim, score_groups=subs, use_pose=use_pose)
        if self.pose_dim:
            self.pose_range = PoseRange(X[sources != CLICK])
        self.model, self.report = model, report
        self.geometry = self._fit_geometry(M, Y, groups, sources)
        self.meta.fit = asdict(report)
        self.meta.updated = time.time()
        log.info("calibration fit: %s (%d targets) cv=%s", report.chosen, report.n_groups,
                 {k: round(v, 1) for k, v in report.cv_error_pt.items()})
        return report

    def _fit_geometry(self, M, Y, groups, sources) -> GeoCalibration | None:
        if self.plane is None or not np.isfinite(M).any():
            return None

        def by_target(source):
            out = []
            for g in np.unique(groups[sources == source]):
                sel = (groups == g) & np.isfinite(M).all(axis=1)
                if sel.any():
                    out.append(([from_array(a) for a in M[sel]], tuple(map(float, Y[sel][0]))))
            return out

        dots = by_target(EXPLICIT)
        if len(dots) < 5:
            return None
        try:
            return fit_geometry(dots, self.plane, by_target(MOTION) or None)
        except np.linalg.LinAlgError:
            log.warning("eyeball model fit failed")
            return None

    @property
    def calibrated(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray, measure: np.ndarray | None = None) -> tuple[float, float] | None:
        """Learned calibrator, blended with the eyeball model when it has a point for this frame."""
        model = self.model
        if model is None:
            return None
        x, y = model.predict(np.asarray(features, np.float64))[0]
        geometry = self.geometry
        if geometry is not None and measure is not None and np.isfinite(measure).all():
            g = geometry.predict(from_array(measure))
            if g is not None:
                x, y = (1 - GEO_WEIGHT) * x + GEO_WEIGHT * g[0], (1 - GEO_WEIGHT) * y + GEO_WEIGHT * g[1]
        dx, dy = self.meta.offset_pt
        return float(x + dx), float(y + dy)

    def recentre(self, predictions: np.ndarray, target: tuple[float, float], max_shift_pt: float = 600.0) -> tuple[float, float]:
        """Drift correction from one dot: shift all predictions so this dot's median lands on it. Between
        sessions the error is mostly such a shift (posture, lid): on replayed session pairs a stale
        calibration was 3–9° off and one re-centred dot brought it back to ~2°.
        `predictions` are this profile's outputs (offset included) while the user looked at `target`.
        Returns the new offset; shifts beyond `max_shift_pt` are refused (recalibrate instead)."""
        miss = np.asarray(target, float) - np.median(np.asarray(predictions, float), axis=0)
        if np.hypot(*miss) > max_shift_pt:
            raise ValueError(f"off by {np.hypot(*miss):.0f} pt: too far for a re-centre, recalibrate")
        ox, oy = self.meta.offset_pt
        self.meta.offset_pt = (float(ox + miss[0]), float(oy + miss[1]))
        return self.meta.offset_pt

    def confidence(self, features: np.ndarray) -> float:
        """1 inside the head-pose range seen during calibration, falling to 0 outside it."""
        r = self.pose_range
        return 1.0 if r is None else r.confidence(np.asarray(features, np.float64))

    # ---- persistence ---------------------------------------------------------------
    @staticmethod
    def path_for(name: str, root: Path | None = None) -> Path:
        return (root or profiles_dir()) / name

    def save(self, root: Path | None = None) -> Path:
        d = self.path_for(self.meta.name, root)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            arrays = dict(X=self.X, M=self.M, Y=self.Y, groups=self.groups, subs=self.subs, sources=self.sources, t=self.t)
        np.savez_compressed(d / "samples.npz", **arrays)
        (d / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        return d

    @classmethod
    def load(cls, name: str, root: Path | None = None, plane: ScreenPlane | None = None) -> "Profile | None":
        d = cls.path_for(name, root)
        try:
            raw = json.loads((d / "meta.json").read_text())
            raw.setdefault("features", "")  # profiles from before head-pose features
            meta = ProfileMeta(**raw)
            data = np.load(d / "samples.npz")
        except (FileNotFoundError, ValueError, TypeError) as e:
            log.info("no usable profile %r: %s", name, e)
            return None
        p = cls(meta, data["X"].shape[1])
        p.X, p.Y, p.groups, p.sources, p.t = data["X"], data["Y"], data["groups"], data["sources"], data["t"]
        p.subs = data["subs"] if "subs" in data else p.groups.copy()
        p.M = data["M"] if "M" in data else np.full((len(p.X), MEASURE_DIM), np.nan, np.float32)
        p.plane = plane
        p._next_group = int(p.groups.max()) + 1 if len(p.groups) else 0
        if meta.features == FEATURE_SET:
            p.fit()
        return p
