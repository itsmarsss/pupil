"""Synthetic round trips for the geometric model: faces whose visual axes hit known screen points."""
import numpy as np
import pytest

from pupil.core.screen import Screen
from pupil.core.types import FaceObservation
from pupil.geometry.calibrate import fit
from pupil.geometry.eyeball import (CENTRE_DEPTH_CM, CORNERS, EYEBALL_RADIUS_CM, IRIS, MEDIAPIPE_VFOV_DEG,
                                   ScreenPlane, from_angles, hits, measure, normalize, to_angles)

W, H, FOCAL = 1440, 1080, 1300.0
SCREEN = Screen(1512, 982, 302.0, 196.1)
PLANE = ScreenPlane(SCREEN, camera_below_top_cm=0.32)
EYE_CORNERS_HEAD = {"right": np.array([[-4.5, 3.0, 2.0], [-1.5, 3.0, 2.0]]),
                    "left": np.array([[1.5, 3.0, 2.0], [4.5, 3.0, 2.0]])}


def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def face_looking_at(target_pt, head_t, R, gain=(1.0, 1.0), kappa=(0.0, 0.0)):
    """Landmarks of a head at head_t (cm) rotated by R whose eyes' visual axes hit target_pt."""
    tan_x, tan_y = W / (2 * FOCAL), H / (2 * FOCAL)
    depth = abs(head_t[2])

    def project(p):
        d = -p[2]
        u = (p[0] / d / tan_x + 1) / 2
        v = (1 - p[1] / d / tan_y) / 2
        z = (d / depth - 1) / (2 * tan_x)
        return np.array([u * W, v * H, z * W])

    lm = np.zeros((478, 3))
    fwd = R[:, 2]
    B = np.stack([R[:, 0], R[:, 1], fwd], axis=1)
    target = np.array([*PLANE.to_cm(*target_pt), 0.0])
    for eye in ("right", "left"):
        cam = (R @ EYE_CORNERS_HEAD[eye].T).T + head_t
        a, b = CORNERS[eye]
        lm[a], lm[b] = project(cam[0]), project(cam[1])
        c = cam.mean(axis=0) - fwd * CENTRE_DEPTH_CM
        visual = to_angles(B, normalize(target - c))
        optical = (visual - np.array(kappa)) / np.array(gain)  # what the camera would measure
        lm[IRIS[eye]] = project(c + EYEBALL_RADIUS_CM * from_angles(B, optical))
    M = np.eye(4)
    M[:3, :3] = R
    f_mp = 1 / np.tan(np.radians(MEDIAPIPE_VFOV_DEG) / 2)
    M[:3, 3] = [head_t[0], head_t[1], head_t[2] * f_mp * tan_y]  # MediaPipe's depth for its assumed FOV
    return FaceObservation(0, (W, H), lm, (0, 0, 1, 1), (0, 0, 1, 1), (0, 0, 1, 1), (0.3, 0.3), 30.0,
                           focal_px=FOCAL, transform=M)


HEADS = [((0, -8, -55), 0), ((6, -5, -50), 12), ((-8, -10, -65), -18)]
POINTS = [(150, 120), (756, 491), (1360, 860), (200, 800)]


def test_plane_roundtrip():
    pts = np.array([[0.0, 0.0], [1512, 982], [123, 456]])
    assert np.allclose(PLANE.to_pt(PLANE.to_cm(pts[:, 0], pts[:, 1])), pts)
    top_centre = PLANE.to_cm(756, 0)
    assert np.allclose(top_centre, [0.0, 0.32])  # camera is the origin, 0.32 cm below the top edge


@pytest.mark.parametrize("head_t,yaw", HEADS)
def test_uncalibrated_geometry_recovers_target(head_t, yaw):
    for pt in POINTS:
        m = measure(face_looking_at(pt, np.array(head_t, float), rot_y(yaw)))
        got = PLANE.to_pt(hits(m, np.ones(2), np.zeros(2)))
        assert np.linalg.norm(got - pt) < 1.0


def test_fit_recovers_gain_and_kappa():
    gain, kappa = (1.2, 2.5), (3.0, -4.0)
    rng = np.random.default_rng(0)
    grid = [(x, y) for y in (100, 491, 880) for x in (150, 756, 1360)]
    dots = []
    for pt in grid:
        frames = [measure(face_looking_at(pt, np.array([0, -8, -55.0]) + rng.normal(0, 0.3, 3), rot_y(rng.normal(0, 1)),
                                          gain, kappa)) for _ in range(5)]
        dots.append((frames, pt))
    cal = fit(dots, PLANE)
    assert np.allclose(cal.gain, gain, atol=0.05) and np.allclose(cal.kappa, kappa, atol=0.3)
    test = measure(face_looking_at((500, 300), np.array([2, -7, -58.0]), rot_y(4), gain, kappa))
    assert np.linalg.norm(np.array(cal.predict(test)) - (500, 300)) < 10
