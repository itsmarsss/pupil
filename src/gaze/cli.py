"""Command line: `gaze` runs the app; `gaze cameras` and `gaze doctor` help set up."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_tracker(args):
    from .capture.camera import CameraSource, FileSource
    from .core.screen import main_screen
    from .face.tracker import FaceTracker
    from .models.mnn_face_eye import MNNFaceEyeModel
    from .tracker import Tracker

    if args.source:
        source = FileSource(args.source)
    else:
        camera = int(args.camera) if args.camera.isdigit() else args.camera
        source = CameraSource(mode=args.mode, camera=camera)
    return Tracker(source, FaceTracker(), MNNFaceEyeModel(args.model), main_screen())


def cmd_run(args) -> int:
    from .ui.app import GazeApp

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

    rows = replay(Path(args.session), MNNFaceEyeModel(args.model), modes=args.modes.split(","))
    print(f"{'mode':5} {'calibrator':22} {'accuracy':>9} {'p95':>7} {'precision':>9} {'loss':>6}")
    for mode, cal, r in rows:
        print(f"{mode:5} {cal:22} {r.accuracy_deg:8.2f}° {r.accuracy_p95_deg:6.2f}° {r.precision_rms_deg:8.2f}° {r.data_loss:6.0%}")
    return 0


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
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
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
        p.add_argument("--targets", type=int, default=9, choices=[5, 9, 13], help="calibration dots")
        p.add_argument("--recalibrate", action="store_true", help="ignore the saved calibration")
        p.add_argument("--no-clicks", action="store_true", help="don't learn from clicks")
        p.add_argument("--results", help="CSV to append accuracy checks to")
        p.add_argument("--snap", action="store_true", help="snap the ring to nearby buttons (needs Accessibility access)")
        p.add_argument("--record", help="write every gaze sample to this CSV")
        p.add_argument("--save-session", nargs="?", const="auto", metavar="DIR",
                       help="save calibration/validation frames for `gaze replay`")
    sub.add_parser("cameras", help="list cameras")
    doctor = sub.add_parser("doctor", help="check camera access, models and screen")
    doctor.add_argument("--camera", default="FaceTime")
    rp = sub.add_parser("replay", help="score pipeline variants on a saved session")
    rp.add_argument("session")
    rp.add_argument("--modes", default="hd,vga")
    rp.add_argument("--model", help="path to the gaze model")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    commands = {"cameras": cmd_cameras, "doctor": cmd_doctor, "replay": cmd_replay}
    return commands.get(args.command, cmd_run)(args)


if __name__ == "__main__":
    sys.exit(main())
