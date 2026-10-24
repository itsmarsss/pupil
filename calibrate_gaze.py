"""Fullscreen gaze calibration + accuracy test. No tape measure: the fit finds the depth scale and kappa.

    uv run python calibrate_gaze.py              # calibrate (25 dots), test (12 other dots), then free look
    uv run python calibrate_gaze.py --quick      # 13-dot calibration
    uv run python calibrate_gaze.py --test-only  # test with the saved calibration

Measure the camera FOV once first (calibrate_fov.py); without it a 62° prior is used.

Keys: SPACE start, ESC quit; after the results, free look (R toggles the uncalibrated dot, ESC quits).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk

import calib
import device
import features as F
import geometry as g
from landmarks import make_landmarker

# Calibration: corners and edge midpoints at a 4% margin (as close as the dot can sit and stay fully on screen),
# the centre, and extra rows at 30% / 70% to pin down the vertical gain. 13 dots.
M = 0.04
CALIB_POINTS = [(fx, fy) for fy in (M, 0.5, 1 - M) for fx in (M, 0.5, 1 - M)] + \
               [(0.27, 0.3), (0.73, 0.3), (0.27, 0.7), (0.73, 0.7)]
# Dense: 5x5 grid, twice the dots in the outer ring. On real runs, edge accuracy was limited by calibration data
# (fewer dots or frames hurt far points most; more flexible models didn't help), so this is the default.
DENSE_POINTS = [(fx, fy) for fy in (M, 0.27, 0.5, 0.73, 1 - M) for fx in (M, 0.27, 0.5, 0.73, 1 - M)]
# Test: none shared with calibration. Near-corner (8% in from both edges), near-edge (6% in), interior.
TEST_SET = [
    ((0.08, 0.08), "corner"), ((0.92, 0.08), "corner"), ((0.08, 0.92), "corner"), ((0.92, 0.92), "corner"),
    ((0.30, 0.06), "edge"), ((0.94, 0.30), "edge"), ((0.70, 0.94), "edge"), ((0.06, 0.70), "edge"),
    ((0.40, 0.40), "interior"), ((0.62, 0.40), "interior"), ((0.38, 0.62), "interior"), ((0.60, 0.60), "interior"),
]
TEST_POINTS = [p for p, _ in TEST_SET]
TEST_REGIONS = [r for _, r in TEST_SET]
SETTLE_S, HOLD_S = 0.8, 1.2
RUNS = Path(__file__).with_name("runs")

CYAN, MAGENTA, WHITE, GREY, YELLOW = (220, 220, 0), (255, 0, 255), (255, 255, 255), (110, 110, 110), (0, 210, 255)
SWEEP_POINTS = [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5)]
SWEEP_S = 7.0


class Tracker:
    def __init__(self, cam: int, width: int, height: int):
        self.cap = cv2.VideoCapture(cam, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("could not read from the camera (check System Settings > Privacy & Security > Camera)")
        self.h, self.w = frame.shape[:2]
        self.lm = make_landmarker()
        self.t0 = time.monotonic()
        self.last_ms = -1
        self.on_frame = None  # optional callback(rgb, lms | None, pose | None), e.g. to stream to Rerun
        self.last_bgr = frame

    def read(self):
        """-> (lms, pose) or None if no face."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.last_bgr = frame
        ms = max(int((time.monotonic() - self.t0) * 1000), self.last_ms + 1)
        self.last_ms = ms
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        out = None
        if res.face_landmarks:
            out = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]]), np.asarray(res.facial_transformation_matrixes[0])
        if self.on_frame:
            self.on_frame(rgb, *(out or (None, None)))
        return out


class Display:
    """The one app window, in Tk. (OpenCV's macOS windows are unreliable in fullscreen, and pygame's SDL clashes
    with the SDL bundled inside OpenCV's wheels.) Windowed for the control panel; fullscreen for calibration,
    testing and free look. Images are BGR numpy arrays drawn with OpenCV and shown here."""

    def __init__(self, title: str = "gaze"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg="black")
        self.label = tk.Label(self.root, bg="black", bd=0, highlightthickness=0)
        self.label.pack(fill="both", expand=True)
        self.keys: list[int] = []
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.keys.append(27))  # closing the window = Esc
        self.photo = None
        self.windowed()

    def _on_key(self, e):
        if e.keysym == "Escape":
            self.keys.append(27)
        elif e.keysym == "space":
            self.keys.append(32)
        elif len(e.char) == 1:
            self.keys.append(ord(e.char.lower()))

    def _settle(self, seconds: float):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.root.update()
            time.sleep(0.02)

    def windowed(self, size=(520, 340)):
        self.root.attributes("-fullscreen", False)
        self.root.geometry(f"{size[0]}x{size[1]}")
        self._settle(0.8)  # macOS animates leaving the fullscreen Space

    def fullscreen(self) -> tuple[int, int]:
        self.root.attributes("-fullscreen", True)
        self._settle(1.2)  # ...and entering it
        self.root.focus_force()
        return self.size()

    def size(self) -> tuple[int, int]:
        self.root.update_idletasks()
        return self.root.winfo_width(), self.root.winfo_height()

    def blit(self, bgr: np.ndarray):
        img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        size = self.size()
        if img.size != size and size[0] > 1:
            img = img.resize(size)
        self.photo = ImageTk.PhotoImage(img)  # keep a reference or Tk drops the image
        self.label.configure(image=self.photo)
        self.root.update()

    def key(self) -> int:
        """Next key as an ASCII code (27 = Esc, 32 = Space; closing the window counts as Esc), or 0."""
        self.root.update()
        return self.keys.pop(0) if self.keys else 0

    def pointer(self) -> tuple[int, int]:
        """Mouse position relative to the window, in window points."""
        r = self.root
        return r.winfo_pointerx() - r.winfo_rootx(), r.winfo_pointery() - r.winfo_rooty()

    def hide_cursor(self, hidden: bool):
        self.root.config(cursor="none" if hidden else "")

    def close(self):
        self.root.destroy()


class Screen:
    """Fullscreen canvas sized to the actual window. The window's physical extent is taken from its shape:
    full panel width, bottom-aligned (macOS puts fullscreen content below the notch)."""

    def __init__(self, info: dict, display: Display):
        self.disp, self.place = info["display"], info["camera_placement"]
        self.display = display
        self.W, self.H = 0, 0
        self.resize(*display.fullscreen())

    def resize(self, rw: int, rh: int):
        if (rw, rh) == (self.W, self.H) or rw <= 0 or rh <= 0:
            return
        self.W, self.H = rw, rh
        mm = self.disp["mm_per_pt"]
        panel_w_cm = self.disp["frame_pt"][0] * mm / 10
        panel_h_cm = self.disp["frame_pt"][1] * mm / 10
        win_h_cm = min(panel_w_cm * rh / rw, panel_h_cm)
        notch_half_cm = self.disp["safe_area_top_pt"] / 2 * mm / 10
        # camera sits notch_half below the panel top; the window top is (panel_h - win_h) below the panel top
        cam_to_top = (panel_h_cm - win_h_cm) - notch_half_cm
        self.geo = g.Screen(panel_w_cm, win_h_cm, rw, rh, cam_to_top_cm=cam_to_top, cam_x_offset_cm=self.place["cam_x_offset_cm"])
        print(f"fullscreen {rw}x{rh} -> {panel_w_cm:.1f} x {win_h_cm:.1f} cm, camera {cam_to_top:+.2f} cm above its top edge")

    def canvas(self) -> np.ndarray:
        img = np.zeros((self.H, self.W, 3), np.uint8)
        cv2.rectangle(img, (0, 0), (self.W - 1, self.H - 1), (60, 60, 60), 2)  # visible edges = no cropping
        return img

    def px(self, f: tuple[float, float]) -> tuple[int, int]:
        return int(f[0] * self.W), int(f[1] * self.H)

    def show(self, img: np.ndarray) -> int:
        self.display.blit(img)
        k = self.display.key()
        self.resize(*self.display.size())  # takes effect from the next frame
        return k


def text(img, s, y=None, color=WHITE, scale=0.9):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    y = y or img.shape[0] // 2
    cv2.putText(img, s, ((img.shape[1] - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def wait_key(scr: Screen, lines: list[str], keys=(ord(" "),)) -> int:
    while True:
        img = scr.canvas()
        for i, s in enumerate(lines):
            text(img, s, scr.H // 2 - 30 * (len(lines) - 1) + 60 * i)
        k = scr.show(img)
        if k == 27:
            raise SystemExit("aborted")
        if k in keys:
            return k


def collect(scr: Screen, trk: Tracker, points, on_target=None) -> list[list[tuple]]:
    """Show each point; return per point a list of (lms, pose) captured after the settle time.
    on_target(fraction | None) is called as each point appears and once at the end."""
    out = []
    for f in points:
        if on_target:
            on_target(f)
        start = time.monotonic()
        frames = []
        while (el := time.monotonic() - start) < SETTLE_S + HOLD_S:
            r = trk.read()
            if r is not None and el >= SETTLE_S:
                frames.append(r)
            img = scr.canvas()
            rad = int(6 + 24 * max(0.0, 1 - el / SETTLE_S))  # shrink onto the target
            cv2.circle(img, scr.px(f), rad, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, scr.px(f), 3, (0, 0, 0), -1, cv2.LINE_AA)
            if scr.show(img) == 27:
                raise SystemExit("aborted")
        out.append(frames)
    if on_target:
        on_target(None)
    return out


def angle_deg(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    u, v = g.normalize(a - o), g.normalize(b - o)
    return float(np.degrees(np.arccos(np.clip(u @ v, -1, 1))))


def score(frames_per_point, points, scr: Screen, intr, uc: calib.UserCalib | None, regions=None) -> list[dict]:
    """Per test dot: accuracy (mean gaze vs target, cm and degrees) and precision. uc=None scores the
    uncalibrated model. Frames rejected as blinks don't count."""
    res = []
    for i, (f, frames) in enumerate(zip(points, frames_per_point)):
        region = regions[i] if regions else None
        target = scr.geo.from_px(np.array(scr.px(f), float))
        hits, angs, origins = [], [], []
        for lms, pose in frames:
            gf = calib.predict(lms, pose, intr, scr.geo, uc)
            if gf is None or gf.hit is None:
                continue
            hits.append(gf.hit)
            origins.append(gf.origin)
            angs.append(angle_deg(gf.origin, target, gf.hit))
        if not hits:
            res.append({"target": list(f), "region": region, "n": 0})
            continue
        hits, origins = np.array(hits), np.array(origins)
        mh, mo = hits.mean(axis=0), origins.mean(axis=0)
        res.append({
            "target": list(f), "region": region, "n": len(hits),
            "mean_hit_px": scr.geo.to_px(mh).tolist(),
            "accuracy_cm": float(np.linalg.norm(mh[:2] - target[:2])),
            "accuracy_deg": angle_deg(mo, target, mh),
            "mean_sample_err_deg": float(np.mean(angs)),
            "precision_sd_deg": float(np.sqrt(np.mean([angle_deg(o, mh, h) ** 2 for o, h in zip(origins, hits)]))),
            "eye_distance_cm": float(abs(mo[2])),
        })
    return res


def summary(rs: list[dict]) -> dict:
    ok = [r for r in rs if r["n"]]
    m = lambda k, xs=ok: float(np.mean([r[k] for r in xs])) if xs else float("nan")
    out = {"accuracy_cm": m("accuracy_cm"), "accuracy_deg": m("accuracy_deg"),
           "median_deg": float(np.median([r["accuracy_deg"] for r in ok])) if ok else float("nan"),
           "precision_sd_deg": m("precision_sd_deg"), "eye_distance_cm": m("eye_distance_cm"), "points": f"{len(ok)}/{len(rs)}"}
    regions = sorted({r.get("region") for r in rs if r.get("region")})
    if regions:
        out["by_region"] = {reg: {"accuracy_cm": m("accuracy_cm", [r for r in ok if r.get("region") == reg]),
                                  "accuracy_deg": m("accuracy_deg", [r for r in ok if r.get("region") == reg]),
                                  "points": sum(1 for r in rs if r.get("region") == reg)} for reg in regions}
    return out


def results_screen(scr: Screen, raw: list[dict], cal: list[dict], s_raw: dict, s_cal: dict,
                   active: str = "", ablation: list[tuple[str, dict]] | None = None):
    img = scr.canvas()
    for r0, r1 in zip(raw, cal):
        t = scr.px(r0["target"])
        cv2.circle(img, t, 6, WHITE, -1, cv2.LINE_AA)
        for r, col in ((r0, MAGENTA), (r1, CYAN)):
            if r["n"]:
                h = tuple(int(v) for v in r["mean_hit_px"])
                cv2.line(img, t, h, col, 2, cv2.LINE_AA)
                cv2.circle(img, h, 5, col, -1, cv2.LINE_AA)
        if r1["n"]:
            cv2.putText(img, f"{r1['accuracy_deg']:.1f} deg", (t[0] + 10, t[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    lines = [
        (f"Accuracy test ({len(cal)} dots not used for calibration)", WHITE),
        (f"uncalibrated: {s_raw['accuracy_cm']:.2f} cm  {s_raw['accuracy_deg']:.2f} deg", MAGENTA),
        (f"calibrated [{active}]: {s_cal['accuracy_cm']:.2f} cm  {s_cal['accuracy_deg']:.2f} deg  "
         f"(median {s_cal['median_deg']:.2f}, precision SD {s_cal['precision_sd_deg']:.2f})", CYAN),
    ]
    for reg, v in s_cal.get("by_region", {}).items():
        lines.append((f"  {reg}: {v['accuracy_cm']:.2f} cm  {v['accuracy_deg']:.2f} deg", CYAN))
    if ablation:
        lines.append(("same data, each fix added in turn:", GREY))
        for label, sm in ablation:
            lines.append((f"  {label}: {sm['accuracy_cm']:.2f} cm  {sm['accuracy_deg']:.2f} deg", GREY))
    lines.append((f"eye distance {s_cal['eye_distance_cm']:.1f} cm      SPACE / ESC to continue", GREY))
    y0 = scr.H // 2 - 17 * len(lines)
    for i, (s, c) in enumerate(lines):
        text(img, s, y0 + 30 * i, c, 0.6)
    return img


def free_look(scr: Screen, trk: Tracker, intr, uc: calib.UserCalib | None, smooth: bool = True):
    show_raw, show_head = True, True
    filt = F.OneEuro()
    while True:
        img = scr.canvas()
        for fy in (0.05, 0.27, 0.5, 0.73, 0.95):
            for fx in (0.05, 0.27, 0.5, 0.73, 0.95):
                x, y = scr.px((fx, fy))
                cv2.line(img, (x - 8, y), (x + 8, y), GREY, 1)
                cv2.line(img, (x, y - 8), (x, y + 8), GREY, 1)
        r = trk.read()
        status = "no face"
        if r is not None:
            lms, pose = r
            gf = calib.predict(lms, pose, intr, scr.geo, uc)
            if gf is None:
                status = "blink"
            else:
                if gf.hit_px is not None:
                    p = filt(gf.hit_px, time.monotonic()) if smooth else gf.hit_px
                    cv2.circle(img, tuple(int(v) for v in p), 14, CYAN, -1, cv2.LINE_AA)
                status = f"eye distance {abs(gf.origin[2]):.0f} cm"
            raw = calib.predict(lms, pose, intr, scr.geo, None)
            if show_raw and raw.hit_px is not None:
                cv2.circle(img, tuple(int(v) for v in raw.hit_px), 10, MAGENTA, 2, cv2.LINE_AA)
            if show_head and raw.head_hit_px is not None:
                hp = tuple(int(v) for v in raw.head_hit_px)
                cv2.circle(img, hp, 16, YELLOW, 2, cv2.LINE_AA)
                cv2.line(img, (hp[0] - 6, hp[1]), (hp[0] + 6, hp[1]), YELLOW, 1)
                cv2.line(img, (hp[0], hp[1] - 6), (hp[0], hp[1] + 6), YELLOW, 1)
        text(img, f"cyan = calibrated{', magenta = uncalibrated' if show_raw else ''}{', yellow = face pointing' if show_head else ''}"
                  f"   {status}   R raw  H face  ESC back", scr.H - 20, GREY, 0.55)
        k = scr.show(img)
        if k == 27:
            return
        if k == ord("r"):
            show_raw = not show_raw
        if k == ord("h"):
            show_head = not show_head


@dataclass
class CalibData:
    """Calibration frames: per target point, its frames, its screen fraction and its averaging group
    (the dot index for dot calibration, the coverage tile for mouse-follow calibration)."""
    frames: list
    points: list
    groups: list

    def fixations(self, scr) -> list[calib.Fixation]:
        return [calib.Fixation(lms, pose, scr.geo.from_px(np.array(scr.px(f), float)), grp)
                for f, fr, grp in zip(self.points, self.frames, self.groups) for lms, pose in fr]


def fixations(frames_per_point, points, scr) -> list[calib.Fixation]:
    return CalibData(frames_per_point, points, list(range(len(points)))).fixations(scr)


def save_frames(path: Path, frames_per_point, points, regions=None, groups=None, extra: dict | None = None):
    """Raw landmarks + poses per point, for re-analysis offline (evaluate.py). `extra`: more arrays to store."""
    lms, pose, idx = [], [], []
    for i, fr in enumerate(frames_per_point):
        for l, p in fr:
            lms.append(l)
            pose.append(p)
            idx.append(i)
    np.savez_compressed(path, lms=np.array(lms, np.float32), pose=np.array(pose, np.float32), point=np.array(idx),
                        points=np.array(points), regions=np.array(regions or [""] * len(points)),
                        groups=np.array(groups if groups is not None else range(len(points))), **(extra or {}))


def load_frames(path: Path):
    """-> (frames per point, points, regions, groups)"""
    d = np.load(path)
    points = [tuple(float(v) for v in p) for p in d["points"]]
    frames = [[] for _ in points]
    for l, p, i in zip(d["lms"], d["pose"], d["point"]):
        frames[int(i)].append((l.astype(float), p.astype(float)))
    groups = [int(x) for x in d["groups"]] if "groups" in d else list(range(len(points)))
    return frames, points, [str(r) for r in d["regions"]], groups


def latest_calibration() -> CalibData | None:
    files = sorted(RUNS.glob("*-calib.npz"))
    if not files:
        return None
    frames, points, _, groups = load_frames(files[-1])
    return CalibData(frames, points, groups)


def head_sweep(scr: Screen, trk: Tracker, intr: g.Intrinsics, points=SWEEP_POINTS, seconds: float = SWEEP_S,
               on_target=None) -> list[list[tuple]]:
    """Eyes stay on a dot while the head moves: the yellow ring (where the face points, relative to where it
    pointed when the dot appeared) should travel around the circle. Teaches head compensation."""
    out = []
    radius = int(0.12 * scr.H)
    for f in points:
        if on_target:
            on_target(f)
        frames, start, origin = [], time.monotonic(), None
        dot = scr.px(f)
        while (el := time.monotonic() - start) < seconds + SETTLE_S:
            r = trk.read()
            img = scr.canvas()
            cv2.circle(img, dot, radius, GREY, 1, cv2.LINE_AA)
            # a marker that walks around the circle, so there's always a "next" place to point at
            a = 2 * np.pi * max(el - SETTLE_S, 0) / (seconds / 1.5)
            cv2.circle(img, (int(dot[0] + radius * np.cos(a)), int(dot[1] + radius * np.sin(a))), 5, GREY, -1, cv2.LINE_AA)
            if r is not None:
                gf = g.compute(r[0], r[1], intr, scr.geo)
                if gf.head_hit_px is not None:
                    origin = gf.head_hit_px if origin is None else origin
                    hp = np.array(dot) + (gf.head_hit_px - origin)
                    cv2.circle(img, tuple(int(v) for v in hp), 12, YELLOW, 2, cv2.LINE_AA)
                if el >= SETTLE_S:
                    frames.append(r)
            cv2.circle(img, dot, 8, WHITE, -1, cv2.LINE_AA)
            text(img, "eyes on the white dot - move your head so the yellow ring follows the grey marker around the circle",
                 scr.H - 20, GREY, 0.55)
            if scr.show(img) == 27:
                raise SystemExit("aborted")
        out.append(frames)
    if on_target:
        on_target(None)
    return out


def run_calibration(scr: Screen, trk: Tracker, intr: g.Intrinsics, cfg: calib.GazeConfig = calib.GazeConfig(),
                    base: g.EyeParams = g.EyeParams(), on_target=None, points=DENSE_POINTS, sweep: bool = True):
    """Dot calibration in the fullscreen window (25 dots by default, CALIB_POINTS for the quick 13).
    Saves user.json and the raw frames; returns (calibration, CalibData)."""
    wait_key(scr, [f"Calibration ({len(points)} dots): look at each dot until it moves on. Keep your head still.",
                   "SPACE to start   ESC to quit"])
    frames = collect(scr, trk, points, on_target)
    all_points, groups = list(points), list(range(len(points)))
    if sweep:
        wait_key(scr, ["Head sweep: keep your EYES on the white dot while you move your HEAD,",
                       "steering the yellow ring around the circle. 3 dots, ~8 s each.", "SPACE to start   ESC to quit"])
        frames += head_sweep(scr, trk, intr, on_target=on_target)
        all_points += SWEEP_POINTS
        groups += [calib.SWEEP_GROUP0 + i for i in range(len(SWEEP_POINTS))]
    data = CalibData(frames, all_points, groups)
    img = scr.canvas()
    text(img, "fitting...")
    scr.show(img)
    uc = calib.fit(data.fixations(scr), intr, scr.geo, cfg, base)
    print(f"calibration [{cfg.label()}]: kappa {uc.mean('kappa_yaw_deg'):+.2f}° / {uc.mean('kappa_pitch_deg'):+.2f}°, "
          f"gain {uc.mean('gain_yaw'):.2f} / {uc.mean('gain_pitch'):.2f}, rms {uc.rms_cm:.2f} cm")
    uc.save()
    RUNS.mkdir(exist_ok=True)
    print(f"head compensation: {'fitted' if uc.head_comp else 'not used'}")
    save_frames(RUNS / time.strftime("%Y%m%d-%H%M%S-calib.npz"), frames, all_points, groups=groups)
    return uc, data


class Coverage:
    """Screen split into tiles; each tile needs `need` accepted frames. Tiles are also the averaging groups."""

    def __init__(self, cols: int = 8, rows: int = 5, need: int = 12):
        self.cols, self.rows, self.need = cols, rows, need
        self.count = np.zeros((rows, cols), int)

    def tile(self, f: tuple[float, float]) -> int:
        c = min(int(f[0] * self.cols), self.cols - 1)
        r = min(int(f[1] * self.rows), self.rows - 1)
        return r * self.cols + c

    def add(self, f: tuple[float, float]) -> int:
        t = self.tile(f)
        self.count.flat[t] += 1
        return t

    def fraction(self) -> float:
        return float(np.mean(self.count >= self.need))

    def draw(self, img: np.ndarray):
        h, w = img.shape[:2]
        for r in range(self.rows):
            for c in range(self.cols):
                k = min(self.count[r, c] / self.need, 1.0)
                # dark red (empty) -> dark green (full), dim so the cursor dot stands out
                col = (int(20 + 30 * k), int(25 + 75 * k), int(70 * (1 - k) + 20))
                x0, y0 = c * w // self.cols, r * h // self.rows
                x1, y1 = (c + 1) * w // self.cols, (r + 1) * h // self.rows
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, -1)
                cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (45, 45, 45), 1)


def run_mouse_calibration(scr: Screen, trk: Tracker, intr: g.Intrinsics, cfg: calib.GazeConfig = calib.GazeConfig(),
                          base: g.EyeParams = g.EyeParams(), on_target=None, lag_s: float = 0.1,
                          max_speed: float = 350.0, finish_at: float = 0.85):
    """Mouse-follow calibration: move the mouse slowly and keep your eyes on the dot under it. Frames are used while
    the mouse moves slowly (fast moves = eyes jumping to catch up), matched to where the mouse was `lag_s` earlier
    (eyes trail a moving target). Tiles turn green as they fill. SPACE finishes, ESC cancels."""
    wait_key(scr, ["Mouse calibration: move the mouse SLOWLY all over the screen and keep your eyes on the dot.",
                   "Keep your head still. Tiles turn green as they fill. SPACE when mostly green.",
                   "SPACE to start   ESC to quit"])
    cov = Coverage()
    hist: deque = deque()  # (time, fx, fy)
    frames, points, groups = [], [], []
    # full record for exact offline re-tuning of lag / speed limit: every face frame (accepted or not) + mouse path
    rec_lms, rec_pose, rec_t, rec_speed, path = [], [], [], [], []
    scr.display.hide_cursor(True)
    try:
        while True:
            now = time.monotonic()
            mx, my = scr.display.pointer()
            f = (min(max(mx / scr.W, 0.0), 1.0), min(max(my / scr.H, 0.0), 1.0))
            hist.append((now, *f))
            path.append((now, *f))
            while hist and now - hist[0][0] > 1.0:
                hist.popleft()
            r = trk.read()
            # where the mouse was lag_s ago, and how fast it was moving around then (points/s)
            past = min(hist, key=lambda h: abs(now - lag_s - h[0]))
            near = [h for h in hist if abs(h[0] - past[0]) <= 0.1]
            speed = 0.0
            if len(near) >= 2 and near[-1][0] > near[0][0]:
                dx = (near[-1][1] - near[0][1]) * scr.W
                dy = (near[-1][2] - near[0][2]) * scr.H
                speed = float(np.hypot(dx, dy) / (near[-1][0] - near[0][0]))
            if r is not None:
                rec_lms.append(r[0])
                rec_pose.append(r[1])
                rec_t.append(now)
                rec_speed.append(speed)
            ok = r is not None and speed <= max_speed
            if ok:
                tf = (past[1], past[2])
                frames.append([r])
                points.append(tf)
                groups.append(cov.tile(tf))
                if on_target:
                    on_target(tf)
                cov.add(tf)
            img = scr.canvas()
            cov.draw(img)
            dot = (int(f[0] * scr.W), int(f[1] * scr.H))
            cv2.circle(img, dot, 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, dot, 3, (0, 0, 0), -1, cv2.LINE_AA)
            frac = cov.fraction()
            msg = ("no face" if r is None else "too fast - slow down" if speed > max_speed else "")
            text(img, f"coverage {frac * 100:.0f}%  ({len(frames)} frames)   {msg}   "
                      f"{'SPACE to finish' if frac >= finish_at else 'keep going'}   ESC cancel",
                 scr.H - 20, WHITE if frac >= finish_at else GREY, 0.55)
            k = scr.show(img)
            if k == 27:
                raise SystemExit("aborted")
            if k == ord(" ") and frames:
                break
    finally:
        scr.display.hide_cursor(False)
        if on_target:
            on_target(None)

    img = scr.canvas()
    text(img, "fitting...")
    scr.show(img)
    data = CalibData(frames, points, groups)
    uc = calib.fit(data.fixations(scr), intr, scr.geo, cfg, base)
    print(f"mouse calibration [{cfg.label()}], {len(frames)} frames, coverage {cov.fraction() * 100:.0f}%: "
          f"kappa {uc.mean('kappa_yaw_deg'):+.2f}° / {uc.mean('kappa_pitch_deg'):+.2f}°, "
          f"gain {uc.mean('gain_yaw'):.2f} / {uc.mean('gain_pitch'):.2f}, rms {uc.rms_cm:.2f} cm")
    uc.save()
    RUNS.mkdir(exist_ok=True)
    save_frames(RUNS / time.strftime("%Y%m%d-%H%M%S-calib.npz"), frames, points, groups=groups, extra={
        "rec_lms": np.array(rec_lms, np.float32), "rec_pose": np.array(rec_pose, np.float32),
        "rec_t": np.array(rec_t), "rec_speed": np.array(rec_speed),
        "mouse_path": np.array(path),  # (time, fx, fy) every UI frame
        "lag_s": np.array(lag_s), "max_speed": np.array(max_speed)})
    return uc, data


def run_test(scr: Screen, trk: Tracker, intr: g.Intrinsics, uc: calib.UserCalib, hfov: float,
             calib_data: CalibData | None = None, base: g.EyeParams = g.EyeParams(), on_target=None) -> np.ndarray:
    """12-dot accuracy test (near-corner, near-edge, interior). Also fits each config of the ablation on the same
    calibration frames and scores it on the same test frames. Saves results + raw frames to runs/."""
    wait_key(scr, ["Accuracy test: look at each dot. Keep your head still.", "SPACE to start   ESC to quit"])
    frames = collect(scr, trk, TEST_POINTS, on_target)
    img = scr.canvas()
    text(img, "scoring...")
    scr.show(img)
    raw = score(frames, TEST_POINTS, scr, intr, None, TEST_REGIONS)
    cal = score(frames, TEST_POINTS, scr, intr, uc, TEST_REGIONS)
    s_raw, s_cal = summary(raw), summary(cal)
    ablation = []
    if calib_data is not None:
        fix = calib_data.fixations(scr)
        for cfg in calib.ABLATION:
            ablation.append((cfg.label(), summary(score(frames, TEST_POINTS, scr, intr, calib.fit(fix, intr, scr.geo, cfg, base), TEST_REGIONS))))
    print("uncalibrated:", json.dumps(s_raw))
    print(f"calibrated [{uc.cfg.label()}]:", json.dumps(s_cal))
    for label, sm in ablation:
        print(f"  {label:40s} {sm['accuracy_cm']:.2f} cm  {sm['accuracy_deg']:.2f}°  " +
              "  ".join(f"{k} {v['accuracy_cm']:.2f}" for k, v in sm.get("by_region", {}).items()))
    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    save_frames(RUNS / f"{stamp}-test.npz", frames, TEST_POINTS, TEST_REGIONS)
    out = RUNS / f"{stamp}.json"
    out.write_text(json.dumps({"calibration": asdict(uc), "hfov_deg": hfov, "capture": [intr.width, intr.height], "screen": asdict(scr.geo),
                               "uncalibrated": {"summary": s_raw, "points": raw},
                               "calibrated": {"summary": s_cal, "points": cal},
                               "ablation": [{"config": label, "summary": sm} for label, sm in ablation]}, indent=2))
    print(f"saved {out}")
    return results_screen(scr, raw, cal, s_raw, s_cal, uc.cfg.label(), ablation)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--mouse", action="store_true", help="mouse-follow calibration instead of dots")
    ap.add_argument("--quick", action="store_true", help="13 dots instead of 25")
    ap.add_argument("--no-sweep", action="store_true", help="skip the head-sweep step")
    args = ap.parse_args()

    info = device.load()
    cam = args.camera if args.camera is not None else device.camera_index(info["cameras"])
    trk = Tracker(cam, args.width, args.height)
    hfov = device.hfov_for(info["cameras"], trk.w, trk.h) or info.get("camera_hfov_deg")
    if hfov is None:
        hfov = device.DEFAULT_HFOV_DEG
        print(f"warning: camera FOV not measured, assuming {hfov:.0f}° (run calibrate_fov.py --paper a4)")
    intr = g.Intrinsics.from_hfov(trk.w, trk.h, hfov)
    print(f"camera {cam} {trk.w}x{trk.h}, hfov {hfov:.1f}°")
    display = Display("gaze calibration")
    scr = Screen(info, display)

    if args.test_only:
        uc = calib.UserCalib.load()
        if uc is None:
            raise SystemExit("no user.json yet; run without --test-only first")
        calib_data = latest_calibration()
    elif args.mouse:
        uc, calib_data = run_mouse_calibration(scr, trk, intr)
    else:
        uc, calib_data = run_calibration(scr, trk, intr, points=CALIB_POINTS if args.quick else DENSE_POINTS,
                                         sweep=not args.no_sweep)
    img = run_test(scr, trk, intr, uc, hfov, calib_data)
    while True:
        k = scr.show(img)
        if k in (27, ord(" ")):
            break
    free_look(scr, trk, intr, uc)
    display.close()


if __name__ == "__main__":
    main()
