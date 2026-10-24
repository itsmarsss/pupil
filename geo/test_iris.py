"""Synthetic eye images: iris partly hidden by eyelids, centre known. The refined centre should be close to the
truth, and unbiased vertically even when a lid covers the top of the iris (where a naive centroid isn't)."""
import cv2
import numpy as np

import iris


def render(cx, cy, r, lid_top, lid_bot, rng, size=(160, 260)):
    """Grey image with sclera, iris (dark ring + darker pupil), eyelids (skin) covering above lid_top/below lid_bot."""
    h, w = size
    img = np.full((h, w), 150, np.float32)                       # skin
    eye = np.zeros((h, w), np.uint8)
    cv2.ellipse(eye, (w // 2, h // 2), (100, 45), 0, 0, 360, 255, -1)
    img[eye > 0] = 215                                           # sclera
    yy, xx = np.mgrid[:h, :w]
    d = np.hypot(xx - cx, yy - cy)
    iris_mask = (d < r) & (eye > 0)
    img[iris_mask] = 90 - 20 * (d[iris_mask] / r)                # iris, darker toward the edge
    img[(d < 0.4 * r) & (eye > 0)] = 25                          # pupil
    lids = (yy < lid_top) | (yy > lid_bot)
    img[lids & (eye > 0)] = 150                                  # lids cover the eye
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), eye


def landmarks(w, h, cx, cy, r, lid_top, lid_bot, bias):
    """Fake MediaPipe landmarks: eye contour on the eye opening, iris centre/ring with a bias (as MediaPipe has)."""
    lms = np.zeros((478, 3))
    ang = np.linspace(0, 2 * np.pi, len(iris.R_EYE_CONTOUR), endpoint=False)
    pts = np.stack([w / 2 + 100 * np.cos(ang), np.clip(h / 2 + 45 * np.sin(ang), lid_top, lid_bot)], axis=1)
    for i, p in zip(iris.R_EYE_CONTOUR, pts):
        lms[i, :2] = p / [w, h]
    c = np.array([cx, cy]) + bias
    lms[468, :2] = c / [w, h]
    for i, a in zip(iris.R_IRIS[1:], (0, np.pi / 2, np.pi, 3 * np.pi / 2)):
        lms[i, :2] = (c + r * np.array([np.cos(a), np.sin(a)])) / [w, h]
    # left eye: copy, so refine() has something to chew on
    for a, b in zip(iris.R_EYE_CONTOUR + iris.R_IRIS, iris.L_EYE_CONTOUR + iris.L_IRIS):
        lms[b] = lms[a]
    return lms


def main():
    rng = np.random.default_rng(0)
    h, w = 160, 260
    errs_mp, errs_ref, dy_ref, refined = [], [], [], 0
    for _ in range(200):
        r = rng.uniform(16, 22)
        cx, cy = w / 2 + rng.uniform(-45, 45), h / 2 + rng.uniform(-12, 12)
        lid_top, lid_bot = cy - r * rng.uniform(0.45, 0.9), cy + r * rng.uniform(0.6, 1.1)  # lids hide top/bottom
        img, _ = render(cx, cy, r, lid_top, lid_bot, rng)
        bias = rng.normal(0, 1.5, 2) + [0, (lid_top - (cy - r)) * 0.25]  # MediaPipe-like error, dragged by the lid
        lms = landmarks(w, h, cx, cy, r, lid_top, lid_bot, bias)
        out = iris.refine(img, lms)
        c_ref = out[0, :2] * [w, h]
        refined += out[0, 2] > 0
        errs_mp.append(np.linalg.norm(bias))
        errs_ref.append(np.linalg.norm(c_ref - [cx, cy]))
        dy_ref.append(c_ref[1] - cy)
    print(f"refined {refined}/200 eyes | centre error: MediaPipe-like seed {np.mean(errs_mp):.2f} px -> refined {np.mean(errs_ref):.2f} px "
          f"| refined vertical bias {np.mean(dy_ref):+.2f} px")
    assert refined > 160 and np.mean(errs_ref) < 0.5 * np.mean(errs_mp)


if __name__ == "__main__":
    main()
