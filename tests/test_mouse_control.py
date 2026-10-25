import argparse
import json
import time
from pupil.control.controller import VoiceController
from pupil.control.sources import MouseGaze
from pupil.dashboard.mouse_feed import MouseControlFeed


def test_mouse_mode_parser_accepts_dashboard_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaze", choices=["eyes", "mouse"], default="eyes")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--dashboard-port", type=int, default=8766)
    args, _ = parser.parse_known_args(["--gaze", "mouse", "--dashboard", "--dashboard-port", "8767"])
    assert args.gaze == "mouse" and args.dashboard and args.dashboard_port == 8767


def test_mouse_gaze_forwards_voice_events():
    seen = []
    gaze = MouseGaze(report=seen.append)
    gaze.report("heard", "hello")
    assert seen == [{"event": "heard", "text": "hello"}]


def test_mouse_feed_streams_gaze_and_voice_context():
    gaze = MouseGaze()
    gaze.start()
    time.sleep(0.05)
    feed = MouseControlFeed(gaze, context=lambda: {"voice": {"connected": True, "heard": "hello", "log": []}})
    hello = json.loads(feed.hello()[0])
    assert hello["type"] == "scene"
    assert len(hello["screen"]["corners_cm"]) == 4
    frames = feed.poll()
    assert frames
    frame = json.loads(frames[0])
    assert frame["gaze"]["smoothed"] is not None
    assert frame["voice"]["heard"] == "hello"


def test_mouse_hybrid_enables_lip_gate_by_default():
    from pupil.control.config import LIP_GATE

    if LIP_GATE == "off":
        return
    assert VoiceController(MouseGaze(lip_gate=True), typed=True).lips is not None
    assert VoiceController(MouseGaze(lip_gate=False), typed=True).lips is None


def test_tts_echo_ignored_without_lip_gate():
    controller = VoiceController(MouseGaze(lip_gate=False), typed=True)
    controller._last_tts_text = "Lost: JBL headphones and luggage."
    controller._tts_echo_until = time.time() + 10
    assert controller._tts_echo("Loss.")
    assert not controller._tts_echo("Open Safari.")


def test_mouse_mode_tick_without_lip_gate_does_not_crash():
    gaze = MouseGaze()
    gaze.start()
    controller = VoiceController(gaze, typed=True)
    controller.lips = None
    controller.events.put(("speech", ("click here", time.time(), time.time(), None)))
    controller.tick()  # would raise if lip_gate reported with lips=None
