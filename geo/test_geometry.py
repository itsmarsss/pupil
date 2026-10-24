"""Synthetic round-trip: build landmarks for a head looking at a known screen point, check we recover it."""
import numpy as np

import geometry as g


def project(p: np.ndarray, head_depth: float, intr: g.Intrinsics) -> np.ndarray:
    d = -p[2]
    u = (p[0] / d / intr.tan_x + 1) / 2
    v = (1 - p[1] / d / intr.tan_y) / 2
    z = (d / head_depth - 1) / (2 * intr.tan_x)
    return np.array([u, v, z])


def rot_y(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def make_frame(head_t, head_R, target, intr, eye=g.EyeParams()):
    lms = np.zeros((478, 3))
    head_depth = abs(head_t[2])
    fwd = head_R[:, 2]
    # eye corners in head space (cm). The subject faces the camera, so their right eye is at camera -x.
    corners_head = {
        g.R_EYE_CORNERS: np.array([[-4.5, 3.0, 2.0], [-1.5, 3.0, 2.0]]),
        g.L_EYE_CORNERS: np.array([[1.5, 3.0, 2.0], [4.5, 3.0, 2.0]]),
    }
    for (a, b), iris_idx in ((g.R_EYE_CORNERS, g.R_IRIS_CENTER), (g.L_EYE_CORNERS, g.L_IRIS_CENTER)):
        cam = (head_R @ corners_head[(a, b)].T).T + head_t
        lms[a], lms[b] = project(cam[0], head_depth, intr), project(cam[1], head_depth, intr)
        center = cam.mean(axis=0) - fwd * eye.center_depth_cm
        iris = center + eye.eyeball_radius_cm * g.normalize(target - center)
        lms[iris_idx] = project(iris, head_depth, intr)
    pose = np.eye(4)
    pose[:3, :3] = head_R
    pose[:3, 3] = head_t
    return lms, pose


def main():
    intr = g.Intrinsics.from_hfov(1280, 720, 70.0)
    screen = g.Screen(30.2, 19.0, 1512, 945, cam_to_top_cm=0.6)
    worst = 0.0
    for head_t, yaw in [((0, -8, -60), 0), ((8, -5, -55), 15), ((-10, -12, -75), -20)]:
        R = rot_y(yaw)
        for px in [(100, 100), (756, 472), (1400, 850), (200, 800)]:
            target = screen.from_px(np.array(px, float))
            lms, pose = make_frame(np.array(head_t, float), R, target, intr)
            gf = g.compute(lms, pose, intr, screen, depth_scale=1.0)
            err = np.linalg.norm(gf.hit_px - np.array(px))
            worst = max(worst, err)
            print(f"head {head_t} yaw {yaw:+3d}°  target {px}  got {np.round(gf.hit_px, 1)}  err {err:.2f}px")
    # round-trip px <-> cm
    assert np.allclose(screen.to_px(screen.from_px(np.array([123.0, 456.0]))), [123, 456])
    # quaternion of a pure yaw
    q = g.rot_to_quat_xyzw(rot_y(30))
    assert np.allclose(q, [0, np.sin(np.radians(15)), 0, np.cos(np.radians(15))])
    print(f"worst error {worst:.3f}px")
    assert worst < 1.0


if __name__ == "__main__":
    main()
