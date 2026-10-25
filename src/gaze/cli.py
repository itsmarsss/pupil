"""Command line: `gaze` runs the app; `gaze cameras` and `gaze doctor` help set up."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_tracker(args):
    from .capture.camera import CameraSource, FileSource
    from .core.paths import GAZE_MODEL_FILE, model_path
    from .core.screen import main_screen
    from .face.tracker import FaceTracker
    from .models.mnn_face_eye import MNNFaceEyeModel
    from .tracker import Tracker

    if args.source:
        source = FileSource(args.source)
    else:
        camera = int(args.camera) if args.camera.isdigit() else args.camera
        source = CameraSource(mode=args.mode, camera=camera)
    if args.appearance == "network" and not args.model and not model_path(GAZE_MODEL_FILE).exists():
        print("No models/gaze_base.mnn: tracking with per-person eye patches (--appearance patches).", flush=True)
        args.appearance = "patches"
    if args.appearance == "patches":
        from .models.eye_patch import EyePatchModel

        model = EyePatchModel()
    else:
        model = MNNFaceEyeModel(args.model)
    if args.normalise != "none":
        from .face.normalize import NormalisedModel

        model = NormalisedModel(model, args.normalise)
    return Tracker(source, FaceTracker(), model, main_screen(), smoothing=args.smoothing)


def _single_instance():
    """Hold an exclusive lock for the app's lifetime; None if another copy already has it."""
    import fcntl

    from .core.paths import APP_SUPPORT

    APP_SUPPORT.mkdir(parents=True, exist_ok=True)
    lock = open(APP_SUPPORT / "app.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        return None
    return lock


def cmd_run(args) -> int:
    from .ui.app import GazeApp

    lock = _single_instance()
    if lock is None:
        print("gaze is already running (◉ in the menu bar). Quit it first: ◉ → Quit.", file=sys.stderr)
        return 1
    tracker = _build_tracker(args)
    GazeApp(
        tracker,
        profile_name=args.profile,
        n_targets=args.targets,
        recalibrate=args.recalibrate,
        results_csv=Path(args.results) if args.results else None,
        learn_from_clicks=not args.no_clicks,
        snap=args.snap,
        record_path=Path(args.record) if args.record else None,
        session_dir=_session_dir(args.save_session),
        measure=args.measure,
        publish=args.publish,
        dashboard_port=args.dashboard_port if args.dashboard else None,
        voice=args.voice,
        voice_typed=args.typed,
    ).run()
    return 0


def _session_dir(value):
    if value is None:
        return None
    if value == "auto":
        import time

        from .core.paths import recordings_dir

        return recordings_dir() / time.strftime("session-%Y%m%d-%H%M%S")
    return Path(value)


def cmd_replay(args) -> int:
    from .eval.replay import replay
    from .models.mnn_face_eye import MNNFaceEyeModel

    if args.dashboard:
        import threading
        import webbrowser

        from .dashboard.replay import build_replay
        from .dashboard.server import DashboardServer

        print("running the pipeline over the session (~1 minute)…")
        feed = build_replay(Path(args.session), MNNFaceEyeModel(args.model))
        server = DashboardServer(feed, port=args.dashboard_port)
        threading.Timer(1.5, webbrowser.open, args=(server.url,)).start()
        print(f"replay at {server.url}  (Ctrl+C to stop)")
        server.serve_forever()
        return 0

    rows = replay(Path(args.session), MNNFaceEyeModel(args.model), modes=args.modes.split(","),
                  all_calibrators=args.all)
    print(f"{'mode':8} {'config':17} {'calibrator':22} {'checks':13} {'accuracy':>9} {'p95':>7} {'jitter':>7} {'loss':>5}  head in range")
    for mode, config, cal, phase, r, inr in rows:
        in_range = f"{inr.accuracy_deg:.2f}° ({len(inr.targets)} dots)" if inr else "-"
        print(f"{mode:8} {config:17} {cal:22} {phase:13} {r.accuracy_deg:8.2f}° {r.accuracy_p95_deg:6.2f}° "
              f"{r.precision_rms_deg:6.2f}° {r.data_loss:5.0%}  {in_range}")
    return 0


def cmd_fov(args) -> int:
    """Measure the camera's focal length from iris size at a known distance (paper sheet as the spacer)."""
    import math
    import time

    import numpy as np

    from .capture.camera import CameraSource
    from .core.device import load_device
    from .face.tracker import FaceTracker, IRIS_DIAMETER_MM

    paper_cm = {"a4": 29.7, "letter": 27.94}
    brow_to_eye_cm = 1.5  # the eyes sit this far behind the brow the paper touches
    distance_mm = 10 * (args.distance_cm if args.distance_cm else paper_cm[args.paper] + brow_to_eye_cm)
    if args.paper:
        print(f"Rest one short edge of a sheet of {args.paper.upper()} paper on the screen just below the camera,")
        print("touch the other end to your forehead between the eyebrows, then take the paper away")
        print(f"and keep still, looking at the camera (eyes ~{distance_mm / 10:.1f} cm from it).")
    input("Press Enter when you're in position... ")

    device = load_device()
    source = CameraSource(mode="hd", camera=int(args.camera) if args.camera.isdigit() else args.camera)
    tracker = FaceTracker(detect_width=1440, smooth=False, device=device)  # full resolution: iris size is the measurement
    source.start()
    focals, depth_focals, last, deadline = [], [], 0, time.monotonic() + 20
    try:
        while len(focals) < args.frames and time.monotonic() < deadline:
            frame = source.latest(after_t_ns=last, timeout=1.0)
            if frame is None:
                continue
            last = frame.t_ns
            face = tracker.process(frame)
            if face is None or face.is_blinking:
                continue
            lm = face.landmarks
            iris = np.mean([max(math.dist(lm[a, :2], lm[c, :2]), math.dist(lm[b, :2], lm[d, :2]))
                            for a, b, c, d in ((469, 470, 471, 472), (474, 475, 476, 477))])
            focals.append(iris * distance_mm / IRIS_DIAMETER_MM * 1080 / frame.height)
            if face.transform is not None:  # MediaPipe solves pose assuming a 63° vertical field of view
                f_mp = (frame.height / 2) / math.tan(math.radians(31.5))
                depth_focals.append(f_mp * distance_mm / (10 * abs(face.transform[2, 3])) * 1080 / frame.height)
            print(f"\r{len(focals)}/{args.frames} frames", end="", flush=True)
    finally:
        source.stop()
        tracker.close()
    print()
    if len(focals) < 10:
        print("Not enough frames with your face and open eyes; try again with more light.")
        return 1
    f = float(np.median(focals))
    spread = float(np.std(focals) / f * 100)
    hfov_full = math.degrees(2 * math.atan(960 / f))
    hfov_crop = math.degrees(2 * math.atan(720 / f))
    print(f"focal length {f:.0f} px at 1080p (spread {spread:.1f} %)")
    print(f"field of view: {hfov_full:.1f}° across the full 1920 frame, {hfov_crop:.1f}° across the 1440 crop")
    if depth_focals:
        fd = float(np.median(depth_focals))
        print(f"cross-check from MediaPipe's head depth (assumes an average head): {fd:.0f} px "
              f"({100 * (fd / f - 1):+.0f} %; within ~15 % is normal)")
    print(f"previous value: {device.focal_px_1080:.0f} px ({'measured' if device.measured else 'assumed'})")
    if args.dry_run:
        return 0
    device.focal_px_1080, device.measured = f, True
    device.method = f"iris at {distance_mm / 10:.1f} cm, {len(focals)} frames, spread {spread:.1f} %"
    device.save()
    print("saved. Recalibrate (`gaze --recalibrate`) so the calibration uses it.")
    return 0


def cmd_episodes(args) -> int:
    from .control import evaluate

    if args.eval is not None:
        return evaluate.evaluate(Path(args.eval)) if args.eval else evaluate.evaluate()
    return evaluate.replay() if args.replay else evaluate.show(args.n)


def cmd_cameras(_args) -> int:
    from .capture.camera import list_cameras

    for i, name in enumerate(list_cameras()):
        print(f"{i}: {name}")
    return 0


def cmd_doctor(args) -> int:
    import AVFoundation as AV

    from .capture.camera import find_camera, list_cameras
    from .core.paths import FACE_LANDMARKER_FILE, GAZE_MODEL_FILE, model_path, models_dir
    from .core.screen import main_screen

    ok = True

    def check(cond: bool, good: str, bad: str) -> None:
        nonlocal ok
        ok &= cond
        print(("✓ " if cond else "✗ ") + (good if cond else bad))

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    check(status == 3, "camera access granted", "camera access not granted (System Settings → Privacy & Security → Camera)")
    cams = list_cameras()
    idx = find_camera(args.camera)
    check(bool(cams), f"camera: {cams[idx] if cams else '-'}", "no cameras found")
    check(model_path(GAZE_MODEL_FILE).exists(), f"gaze model in {models_dir()}", f"gaze model missing: {model_path(GAZE_MODEL_FILE)}")
    lm = model_path(FACE_LANDMARKER_FILE)
    print(("✓ " if lm.exists() else "• ") + ("face landmarker present" if lm.exists() else "face landmarker downloads on first run"))
    s = main_screen()
    print(f"• screen {s.width_pt:.0f}x{s.height_pt:.0f} pt, {s.width_mm:.0f}x{s.height_mm:.0f} mm")
    from .core.device import load_device

    dev = load_device()
    if dev.measured:
        print(f"✓ camera focal length {dev.focal_px_1080:.0f} px at 1080p, measured ({dev.method})")
    else:
        print(f"• camera field of view not measured (assuming {dev.hfov_deg(1920, 1080):.0f}°); run `gaze fov --paper a4`")
    from .core.lid import LidSensor

    lid = LidSensor().angle()
    print(f"• lid angle {lid:.0f}°" if lid is not None else "• no lid angle sensor (external display or older Mac)")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["control"]:  # its own options; it starts the tracker itself
        from .control.__main__ import main as control_main

        return control_main(argv[1:])
    parser = argparse.ArgumentParser(prog="gaze", description="Webcam gaze tracking for macOS")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="start the tracker (default)")
    for p in (parser, run):
        p.add_argument("--mode", choices=["hd", "vga"], default="hd", help="capture resolution (default hd)")
        p.add_argument("--camera", default="FaceTime", help="camera name substring or index")
        p.add_argument("--source", help="replay a video/image instead of the camera")
        p.add_argument("--model", help="path to the gaze model")
        p.add_argument("--profile", default="default", help="calibration profile name")
        p.add_argument("--targets", type=int, default=16, choices=[5, 9, 13, 16, 25], help="calibration dots (default 16)")
        p.add_argument("--recalibrate", action="store_true", help="ignore the saved calibration")
        p.add_argument("--no-clicks", action="store_true", help="don't learn from clicks")
        p.add_argument("--results", help="CSV to append accuracy checks to")
        p.add_argument("--snap", action="store_true", help="snap the ring to nearby buttons (needs Accessibility access)")
        p.add_argument("--record", help="write every gaze sample to this CSV")
        p.add_argument("--save-session", nargs="?", const="auto", metavar="DIR",
                       help="save calibration/validation frames for `gaze replay`")
        p.add_argument("--measure", action="store_true", help="run the head-still + head-moving accuracy measurement")
        p.add_argument("--publish", action="store_true", help="stream gaze over local UDP for `gaze control`")
        p.add_argument("--dashboard", action="store_true", help="serve the web dashboard on 127.0.0.1 and open it")
        p.add_argument("--voice", action="store_true",
                       help="voice control: look at something and say what to do (needs: uv sync --extra control)")
        p.add_argument("--typed", action="store_true", help="with --voice: type commands in the terminal instead")
        p.add_argument("--dashboard-port", type=int, default=8766)
        p.add_argument("--appearance", choices=["network", "patches"], default="network",
                       help="eye appearance features: the pretrained network (default) or eye patches "
                            "fitted per person (no pretrained weights; less accurate)")
        p.add_argument("--smoothing", choices=["off", "light", "medium", "strong"], default="medium",
                       help="ring smoothing; medium holds steady during fixations (default)")
        p.add_argument("--normalise", choices=["roll", "frontal", "none"], default="roll",
                       help="straighten the face before the gaze model (default: level the eyes)")
    sub.add_parser("control", help="shortcut for `gaze --voice` (see `gaze control -h`)")
    sub.add_parser("cameras", help="list cameras")
    fov = sub.add_parser("fov", help="measure the camera's field of view (once per camera)")
    spacer = fov.add_mutually_exclusive_group(required=True)
    spacer.add_argument("--paper", choices=["a4", "letter"], help="paper sheet used as the spacer")
    spacer.add_argument("--distance-cm", type=float, help="measured eye-to-camera distance instead")
    fov.add_argument("--camera", default="FaceTime")
    fov.add_argument("--frames", type=int, default=90)
    fov.add_argument("--dry-run", action="store_true", help="measure and print, don't save")
    doctor = sub.add_parser("doctor", help="check camera access, models and screen")
    doctor.add_argument("--camera", default="FaceTime")
    ep = sub.add_parser("episodes", help="logged voice commands: what was heard, decided and done")
    ep.add_argument("-n", type=int, default=20, help="how many recent episodes to show")
    ep.add_argument("--replay", action="store_true", help="re-run the routing rules on logged answers (offline)")
    ep.add_argument("--eval", nargs="?", const="", metavar="CASES",
                    help="score the gate live on labelled cases (default tests/agent_cases.jsonl)")
    rp = sub.add_parser("replay", help="score pipeline variants on a saved session")
    rp.add_argument("session")
    rp.add_argument("--modes", default="hd+roll,hd", help="e.g. hd, vga, hd+roll, hd+frontal")
    rp.add_argument("--all", action="store_true", help="also score every calibrator, not just the auto choice")
    rp.add_argument("--dashboard", action="store_true", help="play the session back in the web dashboard instead")
    rp.add_argument("--dashboard-port", type=int, default=8766)
    rp.add_argument("--model", help="path to the gaze model")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    commands = {"cameras": cmd_cameras, "doctor": cmd_doctor, "replay": cmd_replay, "fov": cmd_fov,
                "episodes": cmd_episodes}
    return commands.get(args.command, cmd_run)(args)


if __name__ == "__main__":
    sys.exit(main())
