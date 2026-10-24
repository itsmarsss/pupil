"""Feature → screen-point regressors, chosen by leave-one-target-out error.

With 5–9 calibration targets a 258-d ridge fit can memorise the targets and generalise
badly between them, so we fit several candidates and keep whichever predicts held-out
targets best. As click samples accumulate, richer models start winning on their own.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def _design(X: np.ndarray, kind: str) -> np.ndarray:
    """Model inputs: 'raw' = affine on the model's own (x, y); 'full' = all features."""
    if kind == "raw":
        return X[:, :2]
    return X


@dataclass
class Ridge:
    """Weighted ridge with standardised inputs and an unregularised intercept."""

    kind: str = "full"
    alpha: float = 10.0
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    coef: np.ndarray | None = None  # (d + 1) x 2, last row = intercept

    @property
    def name(self) -> str:
        return f"{self.kind}" if self.kind == "raw" else f"{self.kind}/a={self.alpha:g}"

    def fit(self, X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None) -> "Ridge":
        Z = _design(X, self.kind).astype(np.float64)
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
        Zs = (_design(X, self.kind) - self.mean) / self.scale
        return np.hstack([Zs, np.ones((len(Zs), 1))]) @ self.coef


def candidates() -> list[Ridge]:
    # A near-zero alpha keeps the 2-d affine map unbiased.
    return [Ridge("raw", 1e-3)] + [Ridge("full", a) for a in ALPHAS]


@dataclass
class FitReport:
    chosen: str
    cv_error_pt: dict[str, float] = field(default_factory=dict)  # median held-out error per candidate
    n_samples: int = 0
    n_groups: int = 0


MAX_FOLDS = 10


def group_cv_error(model: Ridge, X, Y, w, groups) -> float:
    """Median error (screen points) predicting held-out targets from the rest.
    Leave-one-target-out up to MAX_FOLDS targets, grouped K-fold beyond that."""
    unique = np.unique(groups)
    if len(unique) < 2:
        return float("inf")
    fold_of = {g: i % MAX_FOLDS for i, g in enumerate(np.random.default_rng(0).permutation(unique))}
    folds = np.array([fold_of[g] for g in groups])
    errors = []
    for f in np.unique(folds):
        test = folds == f
        m = Ridge(model.kind, model.alpha).fit(X[~test], Y[~test], w[~test])
        pred = m.predict(X[test])
        for g in np.unique(groups[test]):
            sel = groups[test] == g
            errors.append(float(np.hypot(*(pred[sel].mean(axis=0) - Y[test][sel][0]))))  # one estimate per target
    return float(np.median(errors))


def fit_best(X: np.ndarray, Y: np.ndarray, w: np.ndarray, groups: np.ndarray) -> tuple[Ridge, FitReport]:
    """Fit every candidate on all data; return the one with the lowest group-CV error."""
    report = FitReport(chosen="", n_samples=len(X), n_groups=len(np.unique(groups)))
    best, best_err = None, float("inf")
    for cand in candidates():
        err = group_cv_error(cand, X, Y, w, groups) if report.n_groups >= 4 else float("inf")
        report.cv_error_pt[cand.name] = err
        if err < best_err:
            best, best_err = cand, err
    if best is None:  # too few targets to cross-validate: the simplest model is safest
        best = candidates()[0]
    report.chosen = best.name
    return best.fit(X, Y, w), report
