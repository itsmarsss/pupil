"""A user's calibration samples and fitted mapping, persisted between launches."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..core.paths import profiles_dir
from .regression import FitReport, Ridge, fit_best

log = logging.getLogger(__name__)

EXPLICIT, CLICK = "explicit", "click"
WEIGHTS = {EXPLICIT: 1.0, CLICK: 0.5}


@dataclass
class ProfileMeta:
    name: str
    model_id: str
    camera: str
    screen_pt: tuple[float, float]
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    fit: dict = field(default_factory=dict)

    def matches(self, model_id: str, camera: str, screen_pt: tuple[float, float]) -> bool:
        return (
            self.model_id == model_id
            and self.camera == camera
            and tuple(round(v) for v in self.screen_pt) == tuple(round(v) for v in screen_pt)
        )


class Profile:
    """Calibration samples (features → screen point) with their source and group.

    A group is one target: a calibration dot or one click. Explicit samples are kept;
    click samples are capped and evicted oldest first.
    """

    MAX_CLICK_GROUPS = 400

    def __init__(self, meta: ProfileMeta, dim: int):
        self.meta = meta
        self.X = np.zeros((0, dim), np.float32)
        self.Y = np.zeros((0, 2), np.float32)
        self.groups = np.zeros(0, np.int64)
        self.sources = np.zeros(0, "<U8")
        self.t = np.zeros(0, np.float64)
        self.model: Ridge | None = None
        self.report: FitReport | None = None
        self._next_group = 0

    # ---- samples -------------------------------------------------------------------
    def new_group(self) -> int:
        g = self._next_group
        self._next_group += 1
        return g

    def add(self, X: np.ndarray, target: tuple[float, float], source: str, group: int | None = None) -> int:
        X = np.atleast_2d(np.asarray(X, np.float32))
        group = self.new_group() if group is None else group
        n = len(X)
        self.X = np.vstack([self.X, X])
        self.Y = np.vstack([self.Y, np.tile(np.asarray(target, np.float32), (n, 1))])
        self.groups = np.concatenate([self.groups, np.full(n, group)])
        self.sources = np.concatenate([self.sources, np.full(n, source)])
        self.t = np.concatenate([self.t, np.full(n, time.time())])
        if source == CLICK:
            self._evict_clicks()
        return group

    def clear(self, source: str | None = None) -> None:
        keep = np.zeros(len(self.X), bool) if source is None else self.sources != source
        self._keep(keep)

    def _keep(self, keep: np.ndarray) -> None:
        self.X, self.Y, self.groups = self.X[keep], self.Y[keep], self.groups[keep]
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
        if self.count(EXPLICIT) < 3:
            return None
        w = np.array([WEIGHTS[s] for s in self.sources])
        self.model, self.report = fit_best(self.X, self.Y, w, self.groups)
        self.meta.fit = asdict(self.report)
        self.meta.updated = time.time()
        log.info("calibration fit: %s (%d targets) cv=%s", self.report.chosen, self.report.n_groups,
                 {k: round(v, 1) for k, v in self.report.cv_error_pt.items()})
        return self.report

    @property
    def calibrated(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray) -> tuple[float, float] | None:
        if self.model is None:
            return None
        x, y = self.model.predict(np.asarray(features, np.float64))[0]
        return float(x), float(y)

    # ---- persistence ---------------------------------------------------------------
    @staticmethod
    def path_for(name: str, root: Path | None = None) -> Path:
        return (root or profiles_dir()) / name

    def save(self, root: Path | None = None) -> Path:
        d = self.path_for(self.meta.name, root)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "samples.npz", X=self.X, Y=self.Y, groups=self.groups, sources=self.sources, t=self.t)
        (d / "meta.json").write_text(json.dumps(asdict(self.meta), indent=2))
        return d

    @classmethod
    def load(cls, name: str, root: Path | None = None) -> "Profile | None":
        d = cls.path_for(name, root)
        try:
            meta = ProfileMeta(**json.loads((d / "meta.json").read_text()))
            data = np.load(d / "samples.npz")
        except (FileNotFoundError, ValueError, TypeError) as e:
            log.info("no usable profile %r: %s", name, e)
            return None
        p = cls(meta, data["X"].shape[1])
        p.X, p.Y, p.groups, p.sources, p.t = data["X"], data["Y"], data["groups"], data["sources"], data["t"]
        p._next_group = int(p.groups.max()) + 1 if len(p.groups) else 0
        p.fit()
        return p
