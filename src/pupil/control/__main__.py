"""`pupil control`: the Pupil app with voice control on (a shortcut for `pupil --voice`).

    pupil control                # eye tracking + voice commands, one process
    pupil control --dashboard    # ... and the web dashboard
    pupil control --typed        # type commands in the terminal instead of speaking (dev)
    pupil control --gaze mouse   # cursor = gaze; webcam on for the lip gate (same voice rules as eyes)
    pupil control --gaze mouse --no-camera   # cursor only, no webcam (dev fallback)
    pupil control --calibrate    # recalibrate on start (eyes only)

Any other tracker option (--camera, --mode, …) is passed through in mouse mode too.
"""

import argparse
import signal
import sys
import webbrowser


def _parse_tracker_args(tracker_argv: list[str]):
    from ..cli import add_tracker_arguments, build_tracker

    p = argparse.ArgumentParser(add_help=False)
    add_tracker_arguments(p)
    return build_tracker(p.parse_args(tracker_argv))


def run_mouse_mode(typed: bool, dashboard_port: int | None, tracker_argv: list[str], *, camera: bool) -> int:
    """Voice control with the mouse as gaze; by default the webcam feeds the lip gate like eye mode."""
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSTimer
    from PyObjCTools import AppHelper

    from ..dashboard.mouse_feed import MouseControlFeed
    from ..dashboard.mouse_hybrid_feed import MouseHybridFeed
    from ..dashboard.server import DashboardServer
    from ..ui.app import DASHBOARD_PORT
    from ..ui.services import VoiceLog
    from .controller import VoiceController
    from .sources import MouseGaze, tracker_lips_listener

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    voice = VoiceLog()
    gaze = MouseGaze(report=voice.add, lip_gate=camera)
    gaze.start()

    tracker = None
    if camera:
        tracker = _parse_tracker_args(tracker_argv)
        tracker.start()

    controller = VoiceController(gaze, typed=typed)
    if tracker is not None:
        tracker.add_listener(tracker_lips_listener(controller.lips, lambda: False))
    controller.start()

    dashboard = None
    if dashboard_port is not None:
        port = dashboard_port or DASHBOARD_PORT
        ctx = lambda: {"voice": voice.snapshot(), "busy": False}
        if tracker is not None:
            feed = MouseHybridFeed(tracker, gaze, context=ctx)
        else:
            feed = MouseControlFeed(gaze, context=ctx)
        dashboard = DashboardServer(feed, commands={"clear_research": voice.clear_research}, port=port)
        dashboard.start()
        webbrowser.open(dashboard.url)
        print(f"Dashboard: {dashboard.url}", flush=True)

    cam = "webcam + lip gate" if camera else "no camera (--no-camera)"
    print(f"Mouse control ({cam}): cursor = gaze. Say commands out loud; Ctrl+C quits.", flush=True)

    def quit(*_):
        if dashboard is not None:
            dashboard.stop()
        if tracker is not None:
            tracker.stop()
        gaze.stop()
        AppHelper.stopEventLoop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, quit)
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1.0, True, lambda _t: None)
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 30, True, lambda _t: controller.tick())
    try:
        AppHelper.runEventLoop()
    finally:
        if dashboard is not None:
            dashboard.stop()
        if tracker is not None:
            tracker.stop()
        gaze.stop()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupil control", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaze", choices=["eyes", "face", "mouse"], default="eyes",
                        help="eyes = webcam eye tracking (default); face = where your head points, with ~0.6° "
                             "emulated inaccuracy (demos, no calibration); mouse = the cursor stands in for gaze")
    parser.add_argument("--no-camera", action="store_true",
                        help="with --gaze mouse: no webcam (no lip gate; mic-only voice, dev fallback)")
    parser.add_argument("--calibrate", action="store_true", help="recalibrate the tracker on start (eyes only)")
    parser.add_argument("--typed", action="store_true", help="type commands in the terminal instead of speaking")
    parser.add_argument("--dashboard", action="store_true", help="serve the web dashboard on 127.0.0.1 and open it")
    parser.add_argument("--dashboard-port", type=int, default=8766)
    args, tracker_args = parser.parse_known_args(argv)
    dashboard = ["--dashboard"] if args.dashboard else []
    port = [f"--dashboard-port={args.dashboard_port}"] if args.dashboard else []
    if args.gaze == "mouse":
        return run_mouse_mode(args.typed, args.dashboard_port if args.dashboard else None, tracker_args,
                              camera=not args.no_camera)
    from ..cli import main as pupil_main

    face = ["--pointing=face"] if args.gaze == "face" else []
    return pupil_main(["--voice", *face, *(["--typed"] if args.typed else []),
                      *(["--recalibrate"] if args.calibrate else []), *dashboard, *port, *tracker_args])


if __name__ == "__main__":
    sys.exit(main())
