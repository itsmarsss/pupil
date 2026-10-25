"""What the engine knows about each frame, as plain JSON-ready data for visualisation and tools.

`frame_state()` turns one tracker sample into a FrameState; `scene_info()` describes what doesn't change
per frame (screen, camera, lid). Both are versioned so clients can check they understand the format.

Coordinates:
  - image: MediaPipe-normalised (0..1 of the analysed frame, top-left origin)
  - screen: points, top-left origin of the main display
  - 3D: camera space in cm; +x image right (the user's left), +y up, +z toward the user's back,
    camera at the origin, screen on the plane z = 0. (Same convention as three.js.)
"""
from __future__ import annotations

import math
import time

import numpy as np

from .calibration.profile import CLICK, EXPLICIT, MOTION, Profile
from .core.device import Device
from .core.screen import Screen
from .core.types import Sample
from .geometry.eyeball import EYEBALL_RADIUS_CM, Reconstruction, ScreenPlane, eye_rays, face_pointing, from_array

SCHEMA = 1


def _r(a, nd=2):
    """Round arrays/floats for compact JSON; None stays None."""
    if a is None:
        return None
    if isinstance(a, (float, int, np.floating, np.integer)):
        v = float(a)
        return round(v, nd) if math.isfinite(v) else None
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def scene_info(screen: Screen, plane: ScreenPlane, device: Device, lid_deg: float | None) -> dict:
    corners_pt = np.array([[0, 0], [screen.width_pt, 0], [screen.width_pt, screen.height_pt], [0, screen.height_pt]])
    corners = np.column_stack([plane.to_cm(corners_pt[:, 0], corners_pt[:, 1]), np.zeros(4)])
    return {
        "type": "scene",
        "schema": SCHEMA,
        "screen": {"width_pt": screen.width_pt, "height_pt": screen.height_pt,
                   "width_mm": _r(screen.width_mm, 1), "height_mm": _r(screen.height_mm, 1),
                   "corners_cm": _r(corners)},
        "camera": {"focal_px_1080": _r(device.focal_px_1080, 1), "measured": device.measured,
                   "hfov_deg": _r(device.hfov_deg(1920, 1080), 1)},
        "lid_deg": _r(lid_deg, 1),
    }


def frame_state(sample: Sample, profile: Profile | None, plane: ScreenPlane, context: dict | None = None,
                fps: float = 0.0) -> dict:
    face, point = sample.face, sample.point
    state: dict = {
        "type": "frame",
        "schema": SCHEMA,
        "t": round(sample.t_ns / 1e9, 4),
        "latency_ms": round((time.monotonic_ns() - sample.t_ns) / 1e6, 1),
        "fps": round(fps, 1),
        "state": point.state.value,
        "confidence": _r(point.confidence),
        "distance_cm": _r(point.distance_mm / 10 if point.distance_mm else None, 1),
        "gaze": {  # screen points
            "fused": _r([point.raw_x, point.raw_y], 1) if point.valid else None,
            "smoothed": _r([point.x, point.y], 1) if point.valid else None,
            "fixation": point.fixation_id,
        },
        **(context or {}),
    }
    if face is None:
        return state

    w, h = face.frame_size
    norm = np.array([w, h])
    state["image"] = {
        "size": [w, h],
        "landmarks": _r(face.landmarks[:, :2] / norm, 4),
        "iris": _r(face.landmarks[[468, 473], :2] / norm, 4),
        "boxes": {name: _r(np.array(rect, float) / np.concatenate([norm, norm]), 4)
                  for name, rect in (("face", face.face_rect), ("left_eye", face.left_eye_rect),
                                     ("right_eye", face.right_eye_rect))},
    }
    state["eyes_open"] = _r(face.eye_openness, 3)
    if face.head_pose is not None:
        state["head_ypr"] = _r(face.head_pose.yaw_pitch_roll_deg, 1)

    rec: Reconstruction | None = sample.extra.get("reconstruction")
    if rec is not None:
        m = rec.measure
        scene = {
            "points": _r(rec.points, 2),
            "eyeballs": _r(m.centres, 2),
            "eyeball_radius": EYEBALL_RADIUS_CM,
            "iris": _r(rec.iris, 2),
            "head": {"origin": _r(rec.head_origin, 2), "axes": _r(m.basis.T, 3), "nose": _r(m.nose, 2)},
            "eye_angles": _r(m.angles, 2),
        }
        pointing = face_pointing(m)
        scene["face_pointing"] = _r([*pointing, 0.0], 2) if pointing is not None else None
        state["face_pointing_pt"] = _r(plane.to_pt(pointing), 1) if pointing is not None else None
        geometry = profile.geometry if profile is not None else None
        if geometry is not None:
            dirs, per_eye = eye_rays(m, geometry.gain, geometry.kappa)
            scene["rays"] = {"dirs": _r(dirs, 4), "hits": _r(per_eye, 2) if per_eye is not None else None}
        state["scene"] = scene

    if profile is not None and profile.calibrated and sample.calib is not None:
        ml = profile.model.predict(np.asarray(sample.calib, np.float64))[0]
        state["gaze"]["ml"] = _r(ml, 1)
        if profile.geometry is not None and sample.geo is not None:
            g = profile.geometry.predict(from_array(sample.geo))
            state["gaze"]["eyeball"] = _r(g, 1) if g is not None else None
            if g is not None:
                state["gaze"]["disagreement_pt"] = _r(float(np.hypot(*(np.array(g) - ml))), 1)
        hits_cm = {k: plane.to_cm(*v) for k, v in state["gaze"].items()
                   if k in ("ml", "eyeball", "fused") and v is not None}
        state.setdefault("scene", {})["hits"] = {k: _r([*v, 0.0], 2) for k, v in hits_cm.items()}
    return state


def calibration_info(profile: Profile | None) -> dict:
    """Summary of the active calibration (sent when it changes)."""
    if profile is None or not profile.calibrated:
        return {"type": "calibration", "schema": SCHEMA, "calibrated": False}
    report = profile.report
    g = profile.geometry
    return {
        "type": "calibration",
        "schema": SCHEMA,
        "calibrated": True,
        "profile": profile.meta.name,
        "chosen": report.chosen,
        "cv_error_pt": {k: _r(v, 1) for k, v in report.cv_error_pt.items()},
        "dots": profile.count(EXPLICIT),
        "head_motion": profile.count(MOTION) > 0,
        "clicks": profile.count(CLICK),
        "eyeball": None if g is None else {"gain": _r(g.gain, 3), "kappa_deg": _r(g.kappa, 2),
                                           "residual": g.residual is not None, "head_comp": g.head_comp is not None},
        "lid_deg": profile.meta.lid_deg,
        "offset_pt": _r(profile.meta.offset_pt, 1),
        "targets": _r(np.unique(profile.Y[profile.sources == EXPLICIT], axis=0), 1),
    }
