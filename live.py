"""Live model-based gaze: webcam -> MediaPipe face/iris landmarks -> 3D eye model -> screen hit, shown in Rerun,
with the MacBook's live hinge angle. A small control window runs calibration, the accuracy test and free look
without leaving the live view.

    uv run python live.py                  # opens the Rerun viewer + control window
    uv run python live.py --camera 1       # another camera

Control window keys: C calibrate (25 dots), K 13 dots visited twice, M mouse-follow calibration, T accuracy test (+ comparison of every config), F free look, D drift correction
(one centre dot), V cycle config, O on-screen gaze overlay, H on-screen face-pointing overlay, S smoothing, Q quit.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import calib
import calibrate_gaze as cg
import device
import features as F
import geometry as g
from hinge import LidSensor
from overlay import Overlay



LOG_WIDTH = 960  # camera image sent to Rerun is downscaled to this width (capture is 1920x1080)


def setup_from_device(width: int, height: int, hfov_override: float | None):
    info = device.load()
    hfov = hfov_override or device.hfov_for(info["cameras"], width, height)
    if hfov is None and info.get("camera_hfov_deg"):
        hfov = info["camera_hfov_deg"]
        cw, ch = info.get("camera_hfov_capture", [width, height])
        if abs(cw / ch - width / height) > 0.01:
            print(f"warning: FOV was calibrated at {cw}x{ch}; a different aspect ratio may crop the field of view")
    if hfov is None:
        hfov = device.DEFAULT_HFOV_DEG
        print(f"warning: camera FOV unknown; assuming {hfov}°. Run calibrate_fov.py")
    disp, place = info["display"], info["camera_placement"]
    screen = g.Screen(
        width_cm=disp["usable_cm"][0], height_cm=disp["usable_cm"][1],
        width_px=int(disp["usable_pt"][0]), height_px=int(disp["usable_pt"][1]),
        cam_to_top_cm=place["cam_to_top_cm"], cam_x_offset_cm=place["cam_x_offset_cm"],
    )
    return g.Intrinsics.from_hfov(width, height, hfov), screen, hfov, info


def log_static(intr: g.Intrinsics, screen: g.Screen):
    rr.log("world", rr.ViewCoordinates.RUB, static=True)

    # Camera: pinhole frustum (RDF camera axes = world rotated 180° about x)
    rr.log("world/camera", rr.Transform3D(rotation=rr.RotationAxisAngle(axis=[1, 0, 0], degrees=180)), static=True)
    s = LOG_WIDTH / intr.width
    rr.log("world/camera", rr.Pinhole(focal_length=[intr.fx * s, intr.fy * s], width=LOG_WIDTH, height=round(intr.height * s),
                                      image_plane_distance=6.0), static=True)

    # Screen (usable area) in the lid plane
    c = screen.corners()
    rr.log("world/screen/outline", rr.LineStrips3D([np.vstack([c, c[:1]])], colors=[[90, 160, 255]], radii=0.08), static=True)

    # 2D screen view
    rr.log("screen/bounds", rr.Boxes2D(mins=[0, 0], sizes=[screen.width_px, screen.height_px], colors=[[90, 160, 255]]), static=True)

    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/image", name="Camera"),
                rrb.Spatial2DView(origin="screen", name="Screen"),
                rrb.TimeSeriesView(origin="plots", name="Angles"),
                rrb.TextDocumentView(origin="info", name="Numbers"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    ))


def log_hinge(screen: g.Screen, hinge_deg: float):
    """Lid outline and keyboard deck at the live hinge angle. Context only: the camera and the screen share the
    lid, so the hinge angle doesn't enter the gaze maths."""
    c = screen.corners()
    hinge_y = c[2, 1] - 1.0  # bottom bezel
    x0, x1 = c[0, 0] - 0.5, c[1, 0] + 0.5
    lid_top = c[0, 1] + 0.8
    a = np.radians(hinge_deg)
    d = np.array([0.0, np.cos(a), -np.sin(a)])  # from the hinge, away from the lid by the hinge angle
    deck = np.array([[x0, hinge_y, 0], [x1, hinge_y, 0], [x1, hinge_y, 0], [x0, hinge_y, 0]], float)
    deck[2:] += d * 21.0
    rr.log("world/keyboard", rr.LineStrips3D([np.vstack([deck, deck[:1]])], colors=[[120, 120, 120]], radii=0.08))
    rr.log("world/lid", rr.LineStrips3D([[[x0, lid_top, 0], [x1, lid_top, 0], [x1, hinge_y, 0], [x0, hinge_y, 0], [x0, lid_top, 0]]],
                                        colors=[[70, 70, 70]], radii=0.05))
    rr.log("plots/hinge", rr.Scalars(hinge_deg))


def log_image(rgb: np.ndarray, intr: g.Intrinsics):
    h = round(intr.height * LOG_WIDTH / intr.width)
    small = cv2.resize(rgb, (LOG_WIDTH, h), interpolation=cv2.INTER_AREA) if rgb.shape[1] != LOG_WIDTH else rgb
    rr.log("world/camera/image", rr.Image(small).compress(jpeg_quality=70))


def log_frame(gf: g.GazeFrame, lms: np.ndarray, rgb: np.ndarray, intr: g.Intrinsics, fps: float,
              hinge_deg: float | None = None, status: str = ""):
    log_image(rgb, intr)
    s = LOG_WIDTH / intr.width
    wh = np.array([intr.width, intr.height]) * s
    rr.log("world/camera/image/landmarks", rr.Points2D(lms[:478, :2] * wh, radii=0.7, colors=[[0, 255, 160]]))
    rr.log("world/camera/image/iris", rr.Points2D(lms[[g.R_IRIS_CENTER, g.L_IRIS_CENTER], :2] * wh, radii=2.0, colors=[[255, 60, 200]]))
    if len(lms) >= 480:  # full-resolution refined iris centres (iris.py), yellow where the refinement succeeded
        rr.log("world/camera/image/iris_refined", rr.Points2D(lms[478:480, :2] * wh, radii=2.0,
                                                              colors=[[255, 220, 0] if z > 0 else [120, 120, 120] for z in lms[478:480, 2]]))

    rr.log("world/face/points", rr.Points3D(gf.points, radii=0.08, colors=[[200, 200, 200]]))
    axes = gf.head_R.T * 6.0
    rr.log("world/face/axes", rr.Arrows3D(origins=[gf.head_t] * 3, vectors=axes,
                                          colors=[[255, 60, 60], [60, 255, 60], [60, 120, 255]]))

    eyes = [gf.eyes["right"], gf.eyes["left"]]
    rr.log("world/eyes/balls", rr.Ellipsoids3D(centers=[e.center for e in eyes],
                                               half_sizes=[[1.2, 1.2, 1.2]] * 2, colors=[[255, 255, 255, 60]]))
    rr.log("world/eyes/iris", rr.Points3D([e.iris for e in eyes], radii=0.25,
                                          colors=[[255, 60, 200] if e.iris_hit_sphere else [255, 160, 0] for e in eyes]))
    rays = []
    for name, e in gf.eyes.items():
        end = gf.eye_hits.get(name, e.center + e.direction * 80)
        rays.append([e.center, end])
    rr.log("world/eyes/rays", rr.LineStrips3D(rays, radii=0.05, colors=[[255, 220, 0]]))

    # where the face points: ray from the nose tip along the head's forward axis
    if gf.head_origin is not None:
        end = gf.head_hit if gf.head_hit is not None else gf.head_origin + gf.head_dir * 60
        rr.log("world/face/pointing", rr.LineStrips3D([[gf.head_origin, end]], radii=0.08, colors=[[255, 140, 0]]))
    if gf.head_hit_px is not None:
        rr.log("screen/face_pointing", rr.Points2D([gf.head_hit_px], radii=8, colors=[[255, 140, 0]]))
    else:
        rr.log("screen/face_pointing", rr.Clear(recursive=False))

    if gf.hit is not None:
        rr.log("world/screen/hit", rr.Points3D([gf.hit], radii=0.6, colors=[[0, 220, 220]]))
        rr.log("screen/hit", rr.Points2D([gf.hit_px], radii=12, colors=[[0, 220, 220]]))
        rr.log("screen/eye_hits", rr.Points2D(list(gf.eye_hits_px.values()), radii=5, colors=[[255, 220, 0]]))
    else:
        rr.log("world/screen/hit", rr.Clear(recursive=False))
        rr.log("screen/hit", rr.Clear(recursive=False))

    yaw, pitch, roll = gf.head_ypr_deg
    rr.log("plots/head/yaw", rr.Scalars(yaw))
    rr.log("plots/head/pitch", rr.Scalars(pitch))
    rr.log("plots/head/roll", rr.Scalars(roll))
    if gf.direction is not None:
        d = gf.direction
        rr.log("plots/gaze/yaw", rr.Scalars(np.degrees(np.arctan2(d[0], d[2]))))
        rr.log("plots/gaze/pitch", rr.Scalars(np.degrees(np.arctan2(d[1], np.hypot(d[0], d[2])))))

    q = gf.head_quat_xyzw
    dist = abs(gf.origin[2]) if gf.origin is not None else float("nan")
    hit = "off-screen / away" if gf.hit_px is None else f"({gf.hit_px[0]:.0f}, {gf.hit_px[1]:.0f}) pt"
    rr.log("info", rr.TextDocument(
        (f"**{status}**\n\n" if status else "") +
        f"**fps** {fps:.0f}\n\n"
        f"**hinge** {'n/a' if hinge_deg is None else f'{hinge_deg:.0f}°'}\n\n"
        f"**head position** x {gf.head_t[0]:+.1f}  y {gf.head_t[1]:+.1f}  z {gf.head_t[2]:+.1f} cm\n\n"
        f"**eye distance** {dist:.1f} cm\n\n"
        f"**head quaternion** (x, y, z, w) = ({q[0]:+.3f}, {q[1]:+.3f}, {q[2]:+.3f}, {q[3]:+.3f})\n\n"
        f"**head yaw / pitch / roll** {yaw:+.1f}° / {pitch:+.1f}° / {roll:+.1f}°\n\n"
        f"**gaze hit** {hit}\n\n"
        f"**iris on sphere** right {gf.eyes['right'].iris_hit_sphere}, left {gf.eyes['left'].iris_hit_sphere}",
        media_type=rr.MediaType.MARKDOWN))


class _FracScreen:
    """Adapter so calibration frames (stored as screen fractions) can be refitted against the live screen."""

    def __init__(self, geo: g.Screen):
        self.geo = geo

    def px(self, f):
        return int(f[0] * self.geo.width_px), int(f[1] * self.geo.height_px)


class Live:
    def __init__(self, args):
        self.info = device.load()
        cam = args.camera if args.camera is not None else device.camera_index(self.info["cameras"])
        self.trk = cg.Tracker(cam, args.width, args.height)
        self.intr, self.screen, self.hfov, _ = setup_from_device(self.trk.w, self.trk.h, args.hfov)
        self.base = g.EyeParams(eyeball_radius_cm=args.eye_radius, center_depth_cm=args.eye_depth)
        self.depth_scale = args.depth_scale or g.mediapipe_depth_scale(self.intr)
        self.uc = calib.UserCalib.load()
        self.calib_data = cg.latest_calibration()  # lets V/T refit configs without recalibrating
        self.lid = LidSensor()
        self.hinge: float | None = None
        self.mode = "live"
        self.frame = 0
        self.fps = 0.0
        self.last_t = time.monotonic()
        self.last_event = ""
        print(f"camera {cam}, capture {self.trk.w}x{self.trk.h}, hfov {self.hfov:.1f}°, depth scale {self.depth_scale:.3f}, "
              f"lid sensor {'found' if self.lid.available else 'not found'}")

        rr.init("gaze-geo", spawn=True)
        log_static(self.intr, self.screen)
        self.trk.on_frame = self.on_frame
        self.display = cg.Display("geo live")  # control window; goes fullscreen for calibrate / test / free look
        self.overlay: Overlay | None = None      # O: gaze dot drawn over everything on screen
        self.head_overlay: Overlay | None = None # H: where the face points, drawn over everything
        self.head_px = None
        self.smooth = True                       # S: One Euro smoothing for the overlay and free look
        self.filter = F.OneEuro()
        self.hit_px = None                       # latest gaze point (usable screen area, points)

    def status(self) -> str:
        if not self.uc:
            return f"{self.mode} | not calibrated (press C)"
        u = self.uc
        return (f"{self.mode} | {u.cfg.label()} | kappa {u.mean('kappa_yaw_deg'):+.1f}/{u.mean('kappa_pitch_deg'):+.1f}  "
                f"gain {u.mean('gain_yaw'):.2f}/{u.mean('gain_pitch'):.2f}")

    def on_frame(self, rgb, lms, pose):
        """Called by the tracker for every camera frame, in every mode (live, calibration, test, free look)."""
        now = time.monotonic()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_t, 1e-6)
        self.last_t = now
        rr.set_time("frame", sequence=self.frame)
        if self.frame % 5 == 0 and self.lid.available:
            self.hinge = self.lid.read()
            if self.hinge is not None:
                log_hinge(self.screen, self.hinge)
        self.frame += 1
        gf = calib.predict(lms, pose, self.intr, self.screen, self.uc, self.base, self.depth_scale) if lms is not None else None
        if lms is not None:
            self.head_px = g.compute(lms, pose, self.intr, self.screen, self.base, self.depth_scale).head_hit_px
        else:
            self.head_px = None
        if gf is None:  # no face, or a blink
            self.hit_px = None
            log_image(rgb, self.intr)
            return
        self.hit_px = gf.hit_px
        log_frame(gf, lms, rgb, self.intr, self.fps, self.hinge, self.status())

    def on_target(self, scr: cg.Screen):
        """Show the current calibration/test dot on the 3D screen and the 2D screen view."""
        def cb(f):
            if f is None:
                rr.log("world/screen/target", rr.Clear(recursive=False))
                rr.log("screen/target", rr.Clear(recursive=False))
                return
            px = np.array(scr.px(f), float)
            rr.log("world/screen/target", rr.Points3D([scr.geo.from_px(px)], radii=0.5, colors=[[255, 255, 255]]))
            rr.log("screen/target", rr.Points2D([[f[0] * self.screen.width_px, f[1] * self.screen.height_px]],
                                                radii=10, colors=[[255, 255, 255]]))
        return cb

    def fullscreen(self, mode: str, fn):
        """Run one fullscreen step (calibrate / test / free look / drift), then return to the control window."""
        self.mode = mode
        for ov in (self.overlay, self.head_overlay):
            if ov:
                ov.hide()  # they float above everything, including the fullscreen dots
        scr = cg.Screen(self.info, self.display)
        try:
            fn(scr)
        except SystemExit as e:  # ESC inside the fullscreen step: back to live, don't quit
            print(e)
        finally:
            self.display.windowed()
            self.mode = "live"
            self.filter.reset()

    def calibrate(self, scr: cg.Screen, points=cg.DENSE_POINTS, repeats: int = 1):
        cfg = calib.GazeConfig()  # the default config; V switches afterwards
        self.uc, self.calib_data = cg.run_calibration(scr, self.trk, self.intr, cfg, self.base, self.on_target(scr), points,
                                                      repeats=repeats)
        self.last_event = f"calibrated ({len(points)} dots x{repeats}), rms {self.uc.rms_cm:.2f} cm"

    def calibrate_quick(self, scr: cg.Screen):
        self.calibrate(scr, cg.CALIB_POINTS, repeats=2)

    def calibrate_mouse(self, scr: cg.Screen):
        cfg = calib.GazeConfig()
        self.uc, self.calib_data = cg.run_mouse_calibration(scr, self.trk, self.intr, cfg, self.base, self.on_target(scr))
        self.last_event = f"mouse-calibrated, rms {self.uc.rms_cm:.2f} cm"

    def test(self, scr: cg.Screen):
        if self.uc is None:
            self.last_event = "calibrate first (C)"
            return
        img = cg.run_test(scr, self.trk, self.intr, self.uc, self.hfov, self.calib_data, self.base, self.on_target(scr))
        while scr.show(img) not in (27, ord(" ")):
            self.trk.read()  # keep the live view streaming while the results are up

    def free_look(self, scr: cg.Screen):
        cg.free_look(scr, self.trk, self.intr, self.uc, self.smooth)

    def drift(self, scr: cg.Screen):
        """D: one centre dot; refits only a common offset (re-centres after you've moved)."""
        if self.uc is None:
            self.last_event = "calibrate first (C)"
            return
        frames = cg.collect(scr, self.trk, [(0.5, 0.5)], self.on_target(scr))
        before = self.uc
        self.uc = calib.drift_correct(self.uc, cg.fixations(frames, [(0.5, 0.5)], scr), self.intr, scr.geo, self.base)
        self.uc.save()
        dy = self.uc.mean("kappa_yaw_deg") - before.mean("kappa_yaw_deg")
        dp = self.uc.mean("kappa_pitch_deg") - before.mean("kappa_pitch_deg")
        self.last_event = f"drift corrected by {dy:+.1f} / {dp:+.1f} deg"

    def cycle_config(self):
        """V: switch to the next config of the ablation, refitted on the saved calibration frames."""
        if self.calib_data is None:
            self.last_event = "no calibration frames yet (press C)"
            return
        labels = [c.label() for c in calib.ABLATION]
        cur = self.uc.cfg.label() if self.uc else None
        cfg = calib.ABLATION[(labels.index(cur) + 1) % len(labels)] if cur in labels else calib.ABLATION[-1]
        fs = _FracScreen(self.screen)
        self.uc = calib.fit(self.calib_data.fixations(fs), self.intr, self.screen, cfg, self.base)
        self.uc.save()
        self.last_event = f"config: {cfg.label()}"

    def control_window(self):
        view = cv2.flip(self.trk.last_bgr, 1)
        w, h = self.display.size()
        view = cv2.resize(view, (w, h))
        lines = [self.status(),
                 f"hinge {'n/a' if self.hinge is None else f'{self.hinge:.0f} deg'}   fps {self.fps:.0f}   {self.last_event}",
                 "C calibrate (25)  K 13 dots x2  M mouse  T test  F free  D drift  V config",
                 f"O gaze overlay {'on' if self.overlay else 'off'}  H face overlay {'on' if self.head_overlay else 'off'}"
                 f"  S smoothing {'on' if self.smooth else 'off'}  Q quit"]
        for i, t in enumerate(lines):
            cv2.putText(view, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(view, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 160), 1, cv2.LINE_AA)
        self.display.blit(view)
        return self.display.key()

    def toggle_overlay(self):
        if self.overlay:
            self.overlay.close()
            self.overlay = None
        else:
            self.overlay = Overlay(self.info["display"])
            self.filter.reset()

    def toggle_head_overlay(self):
        if self.head_overlay:
            self.head_overlay.close()
            self.head_overlay = None
        else:
            self.head_overlay = Overlay(self.info["display"], radius=9, rgba=(1.0, 0.55, 0.0, 0.75))

    def update_overlay(self):
        if self.head_overlay:
            self.head_overlay.update(self.head_px)
        if not self.overlay:
            return
        if self.hit_px is None:
            self.overlay.update(None)
            return
        self.overlay.update(self.filter(self.hit_px, time.monotonic()) if self.smooth else self.hit_px)

    def run(self):
        while True:
            self.trk.read()
            self.update_overlay()
            k = self.control_window()
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                self.fullscreen("calibrating", self.calibrate)
            elif k == ord("k"):
                self.fullscreen("quick calibration", self.calibrate_quick)
            elif k == ord("m"):
                self.fullscreen("mouse calibration", self.calibrate_mouse)
            elif k == ord("t"):
                self.fullscreen("testing", self.test)
            elif k == ord("f"):
                self.fullscreen("free look", self.free_look)
            elif k == ord("d"):
                self.fullscreen("drift", self.drift)
            elif k == ord("v"):
                self.cycle_config()
            elif k == ord("o"):
                self.toggle_overlay()
            elif k == ord("h"):
                self.toggle_head_overlay()
            elif k == ord("s"):
                self.smooth = not self.smooth
                self.filter.reset()
        for ov in (self.overlay, self.head_overlay):
            if ov:
                ov.close()
        self.display.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="OpenCV camera index (default: the FaceTime camera)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--hfov", type=float, help="override the camera's horizontal FOV (degrees)")
    ap.add_argument("--eye-radius", type=float, default=1.2)
    ap.add_argument("--eye-depth", type=float, default=0.6, help="eyeball centre behind the eye-corner midpoint (cm)")
    ap.add_argument("--depth-scale", type=float, help="override the MediaPipe depth rescale factor")
    Live(ap.parse_args()).run()


if __name__ == "__main__":
    main()
