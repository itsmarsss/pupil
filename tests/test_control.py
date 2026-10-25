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


def test_split_clauses():
    assert router.split_clauses("open Safari and then make a new tab") == ["open Safari", "make a new tab"]
    assert router.split_clauses("copy this, then switch to Notes") == ["copy this", "switch to Notes"]
    assert router.split_clauses("select all and copy") == ["select all", "copy"]


class _A:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _answers(choices=None, **nouls):
    base = dict(is_command=0.95, enough_context=0.9, is_compound=0.05, needs_agent=0.05, wants_element=0.05,
                wants_app=0.05, wants_shortcut=0.05, wants_scroll=0.05, risky=0.05)
    base.update(nouls)
    ch = {"missing": "who", "app": "Notes", "shortcut": "copy", "scroll_direction": "down",
          "element_action": "activate", **(choices or {})}
    return _A(nouls={k: _A(noul=v) for k, v in base.items()}, choices={k: _A(choice=v) for k, v in ch.items()})


def test_truth_table_clarify_and_compound():
    from gaze.control.gate import _decide

    assert _decide(_answers(enough_context=0.04), [], None, {})[0] == "clarify"
    assert _decide(_answers(enough_context=0.04), [], None, {}, allow_clarify=False)[0] == "agent"  # asks once
    assert _decide(_answers(enough_context=0.24, wants_shortcut=0.9), [], None, {})[0] == "shortcut"  # terse is fine
    assert _decide(_answers(is_compound=0.95), [], None, {})[0] == "compound"
    assert _decide(_answers(is_compound=0.95, wants_app=0.9), [], None, {}, allow_compound=False)[0] == "open_app"
    assert _decide(_answers(wants_scroll=0.95, wants_shortcut=0.8), [], None, {})[0] == "scroll_down"


def test_episodes_label_reactions_and_replay(tmp_path):
    from gaze.control import episodes

    log = episodes.EpisodeLog(tmp_path / "e.jsonl")
    t = time.time()
    a = log.begin("click this", t, t + 0.5, "click")
    a.update(acted=True)
    log.finish(a)
    log.begin("undo", t + 1, t + 1.3, "unknown")  # a correction right after an action
    b = log.begin("open notes", t + 2, t + 2.4, "unknown")
    b.update(gate={"route": "open_app", "nouls": {k: v.noul for k, v in _answers(wants_app=0.9).nouls.items()},
                   "choices": {"app": "Notes", "missing": "who", "shortcut": "copy", "scroll_direction": "down",
                               "element_action": "activate"}}, elements=[], gaze=None)
    log.finish(b)
    log.begin("open notes", t + 3, t + 3.4, "unknown")  # said again: the first one didn't work
    eps = {e["text"]: e for e in episodes.load(tmp_path / "e.jsonl")}
    assert eps["click this"]["label"] == "corrected" and eps["open notes"]["label"] == "repeated"
    [(ep, was, now)] = episodes.replay(list(eps.values()))
    assert was == now == "open_app"


def test_compound_hands_the_rest_to_the_agent(monkeypatch):
    from gaze.control import controller as ctl
    from gaze.control.gate import Decision

    done, started = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._ep, c.pending = {}, None
    c.gaze = _A(point=lambda: (10, 10))
    c.screen = _A(w_pt=100, h_pt=100)
    c.agent = _A(available=True, start=lambda req, *a, **k: started.append(req))
    c._status = lambda text: None
    routes = {"open Notes": Decision("open_app", "", 0, app="Notes", app_path="/Applications/Notes.app"),
              "write a shopping list": Decision("agent", "", 0)}
    c.gate = _A(available=True, decide=lambda clause, *a, **k: routes[clause])
    monkeypatch.setattr(ctl.snapping, "snap", lambda *a: (None, []))
    monkeypatch.setattr(ctl.macos, "frontmost_app", lambda: "Notes")
    monkeypatch.setattr(ctl.commands, "open_app", lambda path: done.append(path))
    monkeypatch.setattr(ctl, "SETTLE_S", {}), monkeypatch.setattr(ctl, "DEFAULT_SETTLE_S", 0)
    c._compound("open Notes and write a shopping list", (10, 10), None, [], "Safari")
    assert done == ["/Applications/Notes.app"]
    assert started == ["open Notes and write a shopping list (Already done: open Notes. "
                       "Do the rest: write a shopping list.)"]
    assert [s["route"] for s in c._ep["clauses"]] == ["open_app", "agent"]
