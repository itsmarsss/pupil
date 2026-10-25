"""`gaze control`: the gaze app with voice control on (a shortcut for `gaze --voice`).

    gaze control                # eye tracking + voice commands, one process
    gaze control --dashboard    # ... and the web dashboard
    gaze control --typed        # type commands in the terminal instead of speaking (dev)
    gaze control --gaze mouse   # the mouse cursor stands in for gaze: no camera, no calibration (dev)
    gaze control --calibrate    # recalibrate on start

Any other option is passed to the tracker (e.g. --targets 25 --snap).
"""

import argparse
import sys


def run_mouse_mode(typed: bool) -> int:
    """Voice control with the mouse as gaze: no tracker, just the overlay and the command tiers."""
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory, NSTimer
    from PyObjCTools import AppHelper

    from .controller import VoiceController
    from .sources import MouseGaze

    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    gaze = MouseGaze()
    gaze.start()
    controller = VoiceController(gaze, typed=typed)
    controller.start()
    NSTimer.scheduledTimerWithTimeInterval_repeats_block_(1 / 30, True, lambda _t: controller.tick())
    AppHelper.runEventLoop(installInterrupt=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gaze control", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaze", choices=["eyes", "mouse"], default="eyes",
                        help="eyes = webcam eye tracking (default); mouse = the cursor stands in for gaze")
    parser.add_argument("--calibrate", action="store_true", help="recalibrate the tracker on start (eyes only)")
    parser.add_argument("--typed", action="store_true", help="type commands in the terminal instead of speaking")
    args, tracker_args = parser.parse_known_args(argv)
    if args.gaze == "mouse":
        return run_mouse_mode(args.typed)
    from ..cli import main as gaze_main

    return gaze_main(["--voice", *(["--typed"] if args.typed else []),
                      *(["--recalibrate"] if args.calibrate else []), *tracker_args])


if __name__ == "__main__":
    sys.exit(main())
