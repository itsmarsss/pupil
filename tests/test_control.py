import json
import socket
import time

from gaze.control import router
from gaze.sinks.publisher import GazePublisher
from gaze.ui.services import VoiceLog


def test_tracker_phrases_route_to_commands():
    assert router.route("re-centre").kind == "recentre"
    assert router.route("recenter").kind == "recentre"
    assert router.route("check accuracy").kind == "check"
    assert router.route("calibrate").kind == "calibrate"
    assert router.route("click this").kind == "click"


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_publisher_routes_commands_and_events():
    port = _free_port()
    got, events = [], []
    GazePublisher(port=_free_port()).listen({"recentre": lambda: got.append("recentre")}, port=port,
                                            on_event=events.append)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    out.sendto(json.dumps({"cmd": "recentre"}).encode(), ("127.0.0.1", port))
    out.sendto(json.dumps({"event": "heard", "text": "click this"}).encode(), ("127.0.0.1", port))
    out.sendto(json.dumps({"cmd": "format_disk"}).encode(), ("127.0.0.1", port))  # unknown: ignored
    deadline = time.time() + 2
    while time.time() < deadline and (not got or not events):
        time.sleep(0.02)
    assert got == ["recentre"]
    assert events == [{"event": "heard", "text": "click this"}]


def test_voice_log_tracks_state_and_history():
    v = VoiceLog(keep=3)
    assert v.snapshot() is None  # no controller yet
    v.add({"event": "mic", "text": "listening"})
    v.add({"event": "pending", "text": "click Delete"})
    v.add({"event": "heard", "text": "confirm"})  # a new utterance clears the pending prompt
    v.add({"event": "target", "target": {"x": 1, "y": 2, "w": 3, "h": 4, "label": "button Delete"}})
    for i in range(4):
        v.add({"event": "status", "text": f"s{i}"})
    snap = v.snapshot()
    assert snap["mic"] == "listening" and snap["heard"] == "confirm" and "pending" not in snap
    assert snap["target"]["label"] == "button Delete" and snap["connected"]
    assert [e["text"] for e in snap["log"]] == ["s1", "s2", "s3"]


def test_tracker_gaze_reads_samples_in_process():
    from gaze.control.sources import TrackerGaze
    from gaze.core.types import GazePoint, Sample, TrackingState

    calls, reports = [], []
    g = TrackerGaze(busy=lambda: False, commands={"recentre": lambda: calls.append("recentre")}, report=reports.append)
    now_ns = time.monotonic_ns()
    for i, (x, conf) in enumerate([(100, 1.0), (110, 0.9), (120, 0.2)]):
        g.on_sample(Sample(now_ns + i * 33_000_000, None, None,
                           GazePoint(now_ns, TrackingState.TRACKING, x, 50.0, x, 50.0, confidence=conf)))
    g.on_sample(Sample(now_ns, None, None, GazePoint(now_ns, TrackingState.NO_FACE)))  # ignored for points
    assert g.connected and g.point() == (120, 50.0)
    t = time.time()
    assert g.point_during(t - 5, t + 5) == (110, 50.0)  # median while "speaking"
    assert abs(g.confidence_during(t - 5, t + 5) - 0.9) < 1e-9
    g.send({"cmd": "recentre"})
    g.report("heard", "click this")
    assert calls == ["recentre"] and reports == [{"event": "heard", "text": "click this"}]
