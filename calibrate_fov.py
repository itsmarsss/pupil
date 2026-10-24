"""One-time camera FOV measurement, for cameras that don't report it (FaceTime HD on M3 MacBook Pros).

No tape measure: use a sheet of printer paper as a spacer. Rest one short edge on the screen just below
the camera, touch the other end to your forehead between the eyebrows, then take the paper away and keep
still while it samples:

    uv run python calibrate_fov.py --paper a4        # or --paper letter, or --distance-cm N

The human iris is ~1.17 cm across with low variance between adults, so its size in pixels at a known distance
gives the focal length (to ~5%). Saved as camera_hfov_deg in device.json.
"""
from __future__ import annotations

import argparse

import cv2
import mediapipe as mp
import numpy as np

import device
import geometry as g
from calibrate_gaze import Display
from landmarks import make_landmarker

IRIS_DIAMETER_CM = 1.17
PAPER_CM = {"a4": 29.7, "letter": 27.94}
BROW_TO_EYE_CM = 1.5  # the eyes sit this far behind the brow the paper touches
R_IRIS_RING = (469, 470, 471, 472)
L_IRIS_RING = (474, 475, 476, 477)


def iris_diameter_px(lms: np.ndarray, w: int, h: int) -> float:
    ds = []
    for ring in (R_IRIS_RING, L_IRIS_RING):
        p = lms[list(ring), :2] * [w, h]
        ds.append(max(np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[1] - p[3])))
    return float(np.mean(ds))


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--paper", choices=PAPER_CM, help="paper sheet used as a spacer from screen to brow")
    src.add_argument("--distance-cm", type=float, help="measured eye-to-camera distance")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=90)
    args = ap.parse_args()

    distance = args.distance_cm or PAPER_CM[args.paper] + BROW_TO_EYE_CM
    info = device.load()
    cam = args.camera if args.camera is not None else device.camera_index(info["cameras"])
    cap = cv2.VideoCapture(cam, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("could not read from the camera (check System Settings > Privacy & Security > Camera)")
    h, w = frame.shape[:2]
    if args.paper:
        print(f"Spacer: {args.paper} paper ({PAPER_CM[args.paper]} cm) from the screen to your brow -> eyes at ~{distance:.1f} cm.")
        print("Set it up, take the paper away, keep still and look at the camera.")
        input("Press Enter when ready... ")
    print(f"camera {cam} ({info['cameras'][cam]['name'] if cam < len(info['cameras']) else '?'}), {w}x{h}")

    guess = g.Intrinsics.from_hfov(w, h, 70.0)  # only used for the small landmark-depth term
    screen = g.Screen(30, 19, 1512, 950)
    landmarker = make_landmarker()
    display = Display("calibrate camera FOV")
    iris_f, raw_depth = [], []
    t = 0
    while len(iris_f) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        t += 33
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t)
        msg = "no face"
        if res.face_landmarks:
            lms = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]])
            iris_f.append(iris_diameter_px(lms, w, h) * distance / IRIS_DIAMETER_CM)
            gf = g.compute(lms, np.asarray(res.facial_transformation_matrixes[0]), guess, screen, depth_scale=1.0)
            raw_depth.append(abs(gf.origin[2]))
            msg = f"{len(iris_f)}/{args.frames}  look at the camera, keep still"
        view = cv2.flip(frame, 1)
        cv2.putText(view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 160), 2)
        display.blit(cv2.resize(view, display.size()))
        if display.key() == 27:
            raise SystemExit("aborted")
    display.close()
    if not iris_f:
        raise SystemExit("no face detected")

    fx = float(np.median(iris_f))
    hfov_iris = float(np.degrees(2 * np.arctan(w / 2 / fx)))

    # cross-check from MediaPipe's head depth (assumes an average face size, so expect a few degrees off)
    k = distance / float(np.median(raw_depth))
    f_mp = 1 / np.tan(np.radians(g.MEDIAPIPE_VFOV_DEG) / 2)
    tan_y = 1 / (k * f_mp)
    hfov_depth = float(np.degrees(2 * np.arctan(tan_y * w / h)))

    print(f"\niris method:  fx {fx:.0f}px  ->  horizontal FOV {hfov_iris:.1f}°  (spread {np.std(iris_f) / fx * 100:.1f}%)")
    print(f"face-size cross-check: horizontal FOV {hfov_depth:.1f}° (assumes an average face; a few degrees off is normal)")

    info["camera_hfov_deg"] = round(hfov_iris, 2)
    info["camera_hfov_capture"] = [w, h]
    info.pop("depth_scale", None)
    device.save(info)
    print(f"saved camera_hfov_deg={hfov_iris:.2f} to {device.DEVICE_JSON.name}")


if __name__ == "__main__":
    main()
