"""Feature → screen-point regressors, chosen by leave-one-target-out error.

With 5–9 calibration targets a 258-d ridge fit can memorise the targets and generalise
badly between them, so we fit several candidates and keep whichever predicts held-out
targets best. As click samples accumulate, richer models start winning on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# Pooled over 36 live validation dots, α=1 produced the worst outliers in every pipeline
# variant and the "rawpose" family was ~2x worse than the rest; both are left out.
ALPHAS = (10.0, 100.0, 1000.0)


def _design(X: np.ndarray, kind: str, pose_dim: int) -> np.ndarray:
    """Model inputs, where the last `pose_dim` columns are the head-pose block:
    'raw' = the model's own (x, y); 'model' = all model features (no pose);
    'full' = model features plus pose. ('rawpose' = raw plus pose; kept for replays.)"""
    n = X.shape[1] - pose_dim
    if kind == "raw":
        return X[:, :2]
    if kind == "rawpose":
        return np.hstack([X[:, :2], X[:, n:]])
    if kind == "model":
        return X[:, :n]
    return X


@dataclass
class Ridge:
    """Weighted ridge with standardised inputs and an unregularised intercept."""

    kind: str = "full"
    alpha: float = 10.0
    pose_dim: int = 0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coef: np.ndarray | None = None  # (d + 1) x 2, last row = intercept

    @property
    def name(self) -> str:
        return "raw" if self.kind == "raw" else f"{self.kind}/a={self.alpha:g}"

    def fit(self, X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> "Ridge":
        Z = _design(X, self.kind, self.pose_dim).astype(np.float64)
        w = np.ones(len(Z)) if w is None else np.asarray(w, dtype=np.float64)
        self.mean = np.average(Z, axis=0, weights=w)
        self.scale = np.sqrt(np.average((Z - self.mean) ** 2, axis=0, weights=w)) + 1e-6
        Zs = (Z - self.mean) / self.scale
        A = np.hstack([Zs, np.ones((len(Zs), 1))])
        reg = np.eye(A.shape[1]) * self.alpha
        reg[-1, -1] = 0.0
        Aw = A * w[:, None]
        self.coef = np.linalg.solve(A.T @ Aw + reg, Aw.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Zs = (_design(X, self.kind, self.pose_dim) - self.mean) / self.scale
        return np.hstack([Zs, np.ones((len(Zs), 1))]) @ self.coef


def candidates(pose_dim: int = 0, use_pose: bool = False) -> list[Ridge]:
    """Pose-aware candidates only when asked: fitting pose on data where the head never
    moved independently of gaze (dots only) makes extrapolation worse, not better."""
    cands = [Ridge("raw", 1e-3, pose_dim)]  # near-zero alpha keeps the 2-d affine map unbiased
    cands += [Ridge("model", a, pose_dim) for a in ALPHAS]
    if use_pose and pose_dim:
        cands += [Ridge("full", a, pose_dim) for a in ALPHAS]
    return cands


@dataclass
class FitReport:
    chosen: str
    cv_error_pt: dict[str, float] = field(default_factory=dict)  # median held-out error per candidate
    n_samples: int = 0
    n_groups: int = 0


MAX_FOLDS = 10


def group_cv_error(model: Ridge, X, Y, w, groups, score_groups=None) -> float:
    """Median error (screen points) predicting held-out targets from the rest.
    Leave-one-target-out up to MAX_FOLDS targets, grouped K-fold beyond that.
    `score_groups` (default: `groups`) sets how finely held-out samples are averaged
    before scoring, e.g. per half-second of a head-motion sweep that shares one target."""
    score_groups = groups if score_groups is None else score_groups
    unique = np.unique(groups)
    if len(unique) < 2:
        return float("inf")
    fold_of = {g: i % MAX_FOLDS for i, g in enumerate(np.random.default_rng(0).permutation(unique))}
    folds = np.array([fold_of[g] for g in groups])
    errors = []
    for f in np.unique(folds):
        test = folds == f
        m = replace(model, mean=None, scale=None, coef=None).fit(X[~test], Y[~test], w[~test])
        pred = m.predict(X[test])
        for g in np.unique(score_groups[test]):
            sel = score_groups[test] == g
            errors.append(float(np.hypot(*(pred[sel].mean(axis=0) - Y[test][sel][0]))))  # one estimate per target
    return float(np.median(errors))


# Within this factor of the best CV error, prefer the simpler candidate: with ~9 targets
# the CV ranking is noisy and the lowest-error fit is often the most overfit one.
CV_TOLERANCE = 1.10
_KIND_RANK = {"raw": 0, "model": 1, "full": 2}


def _simplicity(c: Ridge) -> tuple:
    return (_KIND_RANK[c.kind], -c.alpha)  # simpler kind first, then stronger regularisation


def fit_best(X: np.ndarray, Y: np.ndarray, w: np.ndarray, groups: np.ndarray, pose_dim: int = 0,
             score_groups: np.ndarray | None = None, use_pose: bool = False) -> tuple[Ridge, FitReport]:
    """Fit the simplest candidate whose group-CV error is within CV_TOLERANCE of the best.
    The 2-d "raw" map is only a fallback when there are too few targets to cross-validate."""
    report = FitReport(chosen="", n_samples=len(X), n_groups=len(np.unique(groups)))
    cands = candidates(pose_dim, use_pose)
    if report.n_groups >= 4:
        cands = [c for c in cands if c.kind != "raw"]
        errs = {c.name: group_cv_error(c, X, Y, w, groups, score_groups) for c in cands}
        report.cv_error_pt = errs
        best_err = min(errs.values())
        close = [c for c in cands if errs[c.name] <= best_err * CV_TOLERANCE]
        best = min(close, key=_simplicity)
    else:  # too few targets to cross-validate: the simplest model is safest
        best = cands[0]
    report.chosen = best.name
    return best.fit(X, Y, w), report
