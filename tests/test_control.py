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
