"""Iris centre refined at full camera resolution.

MediaPipe locates the iris inside a downscaled face crop (iris ~12 px across), so sub-pixel biases there are 1-2°
of eye rotation and differ from one eye position to the next. Here MediaPipe only seeds the search: rays are cast
from its centre through the full-resolution eye image, each ray's strongest dark->bright step (iris -> sclera) is
taken as a limbus point, points outside the eye opening (eyelids) are dropped, and a circle is fitted to the rest.
"""
from __future__ import annotations

import cv2
import numpy as np

# Eye opening contours (MediaPipe face mesh), subject's right / left eye
R_EYE_CONTOUR = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
L_EYE_CONTOUR = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
R_IRIS = (468, 469, 470, 471, 472)
L_IRIS = (473, 474, 475, 476, 477)

N_RAYS = 48
REFINED_ROWS = (478, 479)  # where refined centres are appended to the landmark array (x, y normalised; z = 1 if refined)


def fit_circle(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) least-squares circle fit."""
    A = np.column_stack([2 * pts, np.ones(len(pts))])
    b = (pts ** 2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([cx, cy]), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def limbus_points(gray: np.ndarray, c0: np.ndarray, r0: float, contour: np.ndarray) -> np.ndarray:
    """Strongest iris->sclera edge along rays from c0, kept only inside the eye opening."""
    ang = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    radii = np.linspace(0.55 * r0, 1.6 * r0, max(int(1.05 * r0 * 2), 12))
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    xy = c0[None, None, :] + radii[None, :, None] * dirs[:, None, :]  # (rays, samples, 2)
    prof = cv2.remap(gray, xy[..., 0].astype(np.float32), xy[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    prof = cv2.GaussianBlur(prof, (1, 5), 0)  # smooth along each ray
    grad = np.diff(prof, axis=1)              # dark (iris) -> bright (sclera) is positive
    k = np.argmax(grad, axis=1)
    strength = grad[np.arange(N_RAYS), k]
    pts = xy[np.arange(N_RAYS), k + 1]
    # keep strong edges that lie inside the eye opening, slightly shrunk so lid edges don't count
    poly = contour.astype(np.float32)
    centre = poly.mean(axis=0)
    shrunk = centre + 0.85 * (poly - centre)
    inside = np.array([cv2.pointPolygonTest(shrunk, (float(p[0]), float(p[1])), False) > 0 for p in pts])
    keep = inside & (strength > max(4.0, 0.35 * np.median(strength[inside])) if inside.any() else inside)
    return pts[keep]


def refine_one(gray: np.ndarray, lms_px: np.ndarray, iris_idx, contour_idx) -> np.ndarray | None:
    c0 = lms_px[iris_idx[0]]
    r0 = float(np.mean(np.linalg.norm(lms_px[list(iris_idx[1:])] - c0, axis=1)))
    if r0 < 3:
        return None
    contour = lms_px[list(contour_idx)]
    c = c0.copy()
    for _ in range(2):  # re-centre once on the first fit
        pts = limbus_points(gray, c, r0, contour)
        if len(pts) < 8:
            return None
        # robust: drop points far from a first fit
        cc, r = fit_circle(pts)
        d = np.abs(np.linalg.norm(pts - cc, axis=1) - r)
        pts = pts[d < max(1.5, 2.5 * np.median(d))]
        if len(pts) < 6:
            return None
        cc, r = fit_circle(pts)
        if not (0.6 * r0 < r < 1.5 * r0) or np.linalg.norm(cc - c0) > 0.6 * r0:
            return None  # implausible: keep MediaPipe's centre
        c = cc
    return c


def refine(gray: np.ndarray, lms: np.ndarray) -> np.ndarray:
    """lms: (478, 3) normalised landmarks. Returns (2, 3) rows to append: refined right/left iris centre in
    normalised coords, z = 1.0 if refined, 0.0 if it fell back to MediaPipe's centre."""
    h, w = gray.shape
    px = lms[:, :2] * [w, h]
    out = np.zeros((2, 3))
    for row, (iris_idx, contour_idx) in enumerate(((R_IRIS, R_EYE_CONTOUR), (L_IRIS, L_EYE_CONTOUR))):
        c = refine_one(gray, px, iris_idx, contour_idx)
        if c is None:
            out[row] = [*lms[iris_idx[0], :2], 0.0]
        else:
            out[row] = [c[0] / w, c[1] / h, 1.0]
    return out
