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


def test_resampler_keeps_speech_band_and_blocks_aliasing():
    import numpy as np

    from gaze.control.audio import Resampler

    for f, want in ((1000, 0.707), (11000, 0.0)):  # 11 kHz is above 16 kHz audio's Nyquist: must be removed
        t = np.arange(48000) / 24000
        x = np.sin(2 * np.pi * f * t).astype(np.float32)
        r = Resampler(24000)
        y = np.concatenate([r(x[i:i + 2400]) for i in range(0, len(x), 2400)])  # streamed in 100 ms blocks
        assert abs(len(y) - 32000) <= 32
        assert abs(np.sqrt(np.mean(y[500:-500] ** 2)) - want) < 0.01


def test_speech_detector_ignores_silence_and_noise():
    import numpy as np

    from gaze.control.audio import CHUNK, SpeechDetector

    vad = SpeechDetector()
    rng = np.random.default_rng(0)
    probs = [vad((rng.standard_normal(CHUNK) * 0.02).astype(np.float32)) for _ in range(30)]
    assert max(probs) < 0.5


def test_chunks_carry_capture_times_and_clip_cuts_by_them():
    import queue

    import numpy as np

    from gaze.control.audio import CHUNK, RATE
    from gaze.control.voice import Voice

    v = Voice(lambda *a: None, lambda s: None)
    blocks = queue.Queue()
    t0 = 1000.0
    for i in range(10):  # 10 blocks of 0.1 s, block i holds the value i, captured ending at t0 + 0.1 (i + 1)
        blocks.put((t0 + 0.1 * (i + 1), np.full(RATE // 10, i, np.float32)))
    chunks = []
    for t, c in v._chunks(blocks, None):
        chunks.append((t, c))
        if len(chunks) == 25:
            break
    # a chunk's time is when its last sample was captured, however late it's processed
    assert abs(chunks[0][0] - (t0 + CHUNK / RATE)) < 1e-6
    assert abs(chunks[24][0] - (t0 + 25 * CHUNK / RATE)) < 1e-6
    clip = v.clip(t0 + 0.3, t0 + 0.5)
    assert len(clip) == int(0.2 * RATE) and clip[0] == 3 and clip[-1] == 4


def test_automatic_mic_skips_bluetooth(monkeypatch):
    from gaze.control import config, voice

    devices = [{"name": "AirPods", "max_input_channels": 1, "index": 0},
               {"name": "MacBook Pro Microphone", "max_input_channels": 1, "index": 1}]
    monkeypatch.setattr(voice.sd, "query_devices",
                        lambda i=None, kind=None: devices[0] if kind == "input" else devices if i is None else devices[i])
    monkeypatch.setattr(voice, "default_input_is_bluetooth", lambda: True)
    monkeypatch.setattr(config, "AVOID_BLUETOOTH_MIC", True)
    assert voice.pick_mic() == (1, "MacBook Pro Microphone")
    assert voice.pick_mic("AirPods") == (0, "AirPods")  # chosen by name: honoured
    monkeypatch.setattr(config, "AVOID_BLUETOOTH_MIC", False)
    assert voice.pick_mic() == (0, "AirPods")


def test_episode_audio_is_saved_as_wav(tmp_path):
    import wave

    import numpy as np

    from gaze.control.episodes import EpisodeLog

    log = EpisodeLog(tmp_path / "e.jsonl")
    ep = log.begin("click this", 0, 1, "click")
    log.save_audio(ep, np.zeros(16000, np.float32))
    with wave.open(str(tmp_path / ep["audio"])) as w:
        assert (w.getframerate(), w.getnframes(), w.getsampwidth()) == (16000, 16000, 2)


def test_scroll_direction_follows_natural_scrolling():
    from gaze.control.actions import SCROLL_PX, scroll_delta

    # measured: with natural scrolling on, +300 px scrolled a window's content down by 300
    assert scroll_delta("down", 4, natural=True) == (4 * SCROLL_PX, 0)
    assert scroll_delta("down", 4, natural=False) == (-4 * SCROLL_PX, 0)
    assert scroll_delta("up", 1, natural=True) == (-SCROLL_PX, 0)
    assert scroll_delta("right", 2, natural=True) == (0, 2 * SCROLL_PX)


def test_drag_sends_down_drags_up_with_the_button_and_modifiers(monkeypatch):
    from gaze.control import actions

    events, keys = [], []
    monkeypatch.setattr(actions, "_mouse_event", lambda kind, x, y, button, flags: events.append((kind, button, flags)))
    monkeypatch.setattr(actions.pyautogui, "moveTo", lambda *a: None)
    monkeypatch.setattr(actions.pyautogui, "keyDown", lambda k: keys.append(("down", k)))
    monkeypatch.setattr(actions.pyautogui, "keyUp", lambda k: keys.append(("up", k)))
    monkeypatch.setattr(actions.time, "sleep", lambda s: None)
    actions.drag([(0, 0), (100, 0)], button="middle", modifiers=["shift"], duration=0.1)
    kinds = [k for k, _, _ in events]
    assert kinds[0] == "kCGEventOtherMouseDown" and kinds[-1] == "kCGEventOtherMouseUp"
    assert set(kinds[1:-1]) == {"kCGEventOtherMouseDragged"} and len(kinds) > 3
    assert all(b == 2 and f != 0 for _, b, f in events)  # middle button, shift flag on every event
    assert keys == [("down", "shift"), ("up", "shift")]


def test_moving_with_the_button_held_is_a_drag(monkeypatch):
    from gaze.control import actions

    moved, held = [], []
    monkeypatch.setattr(actions.pyautogui, "moveTo", lambda *a: moved.append(a))
    monkeypatch.setattr(actions.pyautogui, "mouseDown", lambda: None)
    monkeypatch.setattr(actions.pyautogui, "mouseUp", lambda: None)
    monkeypatch.setattr(actions, "move_held", lambda x, y: held.append((x, y)))
    ex = actions.ComputerExecutor(_A(to_points=lambda x, y: (x, y)))
    ex.run("mouse_move", {"coordinate": [1, 1]})
    ex.run("left_mouse_down", {})
    ex.run("mouse_move", {"coordinate": [5, 5]})
    ex.run("left_mouse_up", {})
    ex.run("mouse_move", {"coordinate": [9, 9]})
    assert moved == [(1, 1), (9, 9)] and held == [(5, 5)]


def test_open_app_by_approximate_name(monkeypatch):
    from gaze.control import apps

    opened = []
    monkeypatch.setattr(apps.commands, "available_apps", lambda: {"Notes": "/N.app", "Blender": "/B.app"})
    monkeypatch.setattr(apps, "open_path", lambda name, path: opened.append(path) or f"Opened {name}.")
    assert apps.open_app("notes") == ("Opened Notes.", False) and opened == ["/N.app"]
    assert apps.open_app("blend")[1] is False and opened[-1] == "/B.app"
    assert apps.open_app("Photoshop")[1] is True


def test_blender_python_without_a_bridge_explains_what_to_do(monkeypatch, tmp_path):
    from gaze.control.apps import blender

    monkeypatch.setattr(blender, "TOKEN_FILE", tmp_path / "token")  # no token file: nothing to connect with
    monkeypatch.setattr(blender, "_running", lambda: True)
    assert "wasn't started by voice control" in blender.run("1")["error"]
    monkeypatch.setattr(blender, "_running", lambda: False)
    assert "open_app" in blender.run("1")["error"]


def test_type_text_uses_shift_return_for_new_lines_and_restores_the_clipboard(monkeypatch):
    from gaze.control import actions

    calls, clipboard = [], [b"mine"]
    monkeypatch.setattr(actions.pyautogui, "write", lambda s, interval=0: calls.append(("write", s)))
    monkeypatch.setattr(actions.pyautogui, "hotkey", lambda *k: calls.append(("hotkey", k)))
    monkeypatch.setattr(actions, "press", lambda combo, repeat=1: calls.append(("press", combo)))
    monkeypatch.setattr(actions.time, "sleep", lambda s: None)

    def run(cmd, input=None, capture_output=False, check=False):
        if cmd == ["pbpaste"]:
            return _A(stdout=clipboard[-1])
        clipboard.append(input)
        return _A(stdout=b"")

    monkeypatch.setattr(actions.subprocess, "run", run)
    actions.type_text("/btw café\nline two")
    assert calls == [("press", "command+v"), ("hotkey", ("shift", "enter")), ("write", "line two")]
    assert clipboard[1] == "/btw café".encode() and clipboard[-1] == b"mine"  # pasted, then put back


def test_dictation_join_spacing():
    from gaze.control.dictation import join

    assert join("We test on Monday", "and then we ship it.") == " and then we ship it."
    assert join("Hello ", "world") == "world"
    assert join("Done", ". Next") == ". Next"
    assert join("", "/btw") == "/btw"


def test_menu_paths_match_titles_and_report_what_exists(monkeypatch):
    import pytest

    from gaze.control import uitree

    def node(role, title="", children=(), enabled=True):
        return {"AXRole": role, "AXTitle": title, "AXChildren": list(children), "AXEnabled": enabled}

    export = node("AXMenuItem", "Export…", [node("AXMenu", children=[node("AXMenuItem", "PNG"),
                                                                       node("AXMenuItem", "PDF", enabled=False)])])
    file_menu = node("AXMenuBarItem", "File", [node("AXMenu", children=[node("AXMenuItem", "Save"), export])])
    app = {"AXMenuBar": node("AXMenuBar", children=[node("AXMenuBarItem", "Apple"), file_menu])}
    pressed = []
    monkeypatch.setattr(uitree, "_attr", lambda e, name: e.get(name))
    monkeypatch.setattr(uitree, "_text", lambda e, name: e.get(name) or "")
    monkeypatch.setattr(uitree, "_app", lambda name=None: (app, "Demo"))
    monkeypatch.setattr(uitree, "AXUIElementPerformAction", lambda e, a: pressed.append(e["AXTitle"]) or 0)

    assert uitree.menu("list", "file > export") == "file > export: PNG; PDF [disabled]"
    assert uitree.menu("choose", "File > Export > PNG") == "Chose Demo: File > Export > PNG"
    assert pressed == ["PNG"]
    with pytest.raises(RuntimeError, match="disabled"):
        uitree.menu("choose", "File > Export > PDF")
    with pytest.raises(LookupError, match="there is: Save, Export"):
        uitree.menu("choose", "File > Print")


def test_applescript_reports_errors():
    from gaze.control import uitree

    assert uitree.applescript("return 6 * 7") == ("42", False)
    out, error = uitree.applescript("this is not applescript")
    assert error and "error" in out.lower()


def test_adapter_tools_only_when_their_app_is_in_play(monkeypatch):
    from gaze.control import apps

    monkeypatch.setattr(apps, "_running", lambda app: False)
    assert apps.tools_for("make a slide about this") == []
    assert [t["name"] for t in apps.tools_for("build me a car in blender")] == ["blender_python"]
    monkeypatch.setattr(apps, "_running", lambda app: app == "Blender")
    assert [t["name"] for t in apps.tools_for("make the cube red")] == ["blender_python"]
    assert apps.call("not_an_adapter", {}) is None


def test_close_never_guesses_a_whole_window(monkeypatch):
    from gaze.control import controller as ctl
    from gaze.control.snapping import Element

    clicks, keys = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._ep, c._status = {}, lambda text: None
    c._do = lambda kind, args, gaze, target: clicks.append(target.label)
    monkeypatch.setattr(ctl.actions, "press", lambda combo: keys.append(combo))
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    row = Element("AXRow", "Chrome Helper", 0, 0, 100, 20, kind="row")
    dialog_close = Element("AXButton", "close", 90, 0, 10, 10, parent="Update", subrole="AXCloseButton")
    not_now = Element("AXButton", "Not Now", 0, 40, 60, 20, parent="Update")
    tab = Element("AXRadioButton", "Figma", 0, 0, 100, 30, kind="Registerkarte", parent_role="AXTabGroup")  # any language
    c._candidates = [row, dialog_close, not_now, tab]
    assert c._close(row) is False and clicks == []  # a list row: hand to the agent, don't close its window
    assert c._close(not_now) and clicks == ["close"]  # the dialog's own close control, found as a sibling
    assert c._close(tab) and clicks[-1] == "Figma" and keys == ["command+w"]


def test_menu_commands_are_found_by_their_words_in_any_app():
    from gaze.control.commands import MENU_PHRASES
    from gaze.control.uitree import find_menu_title

    chrome = ["select previous tab", "select next tab", "reload this page", "close tab", "close other tabs", "back"]
    wezterm = ["activate the tab to the left", "activate the tab to the right", "reload configuration", "close current tab"]
    finder = ["show previous tab", "open in new tab", "new tab", "back"]
    assert find_menu_title(chrome, MENU_PHRASES["previous_tab"]) == "select previous tab"
    assert find_menu_title(wezterm, MENU_PHRASES["previous_tab"]) == "activate the tab to the left"
    assert find_menu_title(finder, MENU_PHRASES["previous_tab"]) == "show previous tab"
    assert find_menu_title(chrome, MENU_PHRASES["close_tab"]) == "close tab"  # not "close other tabs"
    assert find_menu_title(wezterm, MENU_PHRASES["close_tab"]) == "close current tab"
    assert find_menu_title(wezterm, MENU_PHRASES["refresh"]) is None  # not "reload configuration"
    assert find_menu_title(finder, MENU_PHRASES["new_tab"]) == "new tab"
