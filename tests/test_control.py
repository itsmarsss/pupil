import json
import socket
import time

from pupil.control import router
from pupil.sinks.publisher import GazePublisher
from pupil.ui.services import VoiceLog


def test_tracker_phrases_route_to_commands():
    assert router.route("re-centre").kind == "recentre"
    assert router.route("recenter").kind == "recentre"
    assert router.route("check accuracy").kind == "check"
    assert router.route("calibrate").kind == "calibrate"
    assert router.route("stop").kind == "cancel" and router.route("do it").kind == "confirm"
    for said in ("click this", "scroll down", "go to github.com", "type hello", "again", "quit games", "please"):
        assert router.route(said).kind == "unknown", said  # Jev reads everything else


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
    from pupil.control.sources import TrackerGaze
    from pupil.core.types import GazePoint, Sample, TrackingState

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


def test_tracker_gaze_boosts_only_a_sustained_final_fixation():
    from pupil.control.sources import TrackerGaze
    from pupil.core.types import GazePoint, Sample, TrackingState

    g = TrackerGaze(busy=lambda: False, commands={})
    now_ns = time.monotonic_ns()
    for i in range(8):
        fixation = 1 if i < 4 else 2
        x = 100.0 if fixation == 1 else 300.0
        t_ns = now_ns + i * 33_000_000
        g.on_sample(Sample(t_ns, None, None, GazePoint(t_ns, TrackingState.TRACKING, x, 50.0, x, 50.0,
                                                       fixation_id=fixation)))
    cloud = g.cloud_during(time.time() - 5, time.time() + 5)
    assert cloud.weights == [1.0] * 4 + [2.0] * 4

    g2 = TrackerGaze(busy=lambda: False, commands={})
    for i in range(4):
        fixation = 1 if i < 2 else 2
        t_ns = now_ns + i * 33_000_000
        g2.on_sample(Sample(t_ns, None, None, GazePoint(t_ns, TrackingState.TRACKING, i * 100.0, 50.0,
                                                        i * 100.0, 50.0, fixation_id=fixation)))
    assert g2.cloud_during(time.time() - 5, time.time() + 5).weights == [1.0] * 4


class _A:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _answers(choices=None, **nouls):
    base = dict(is_command=0.95, enough_context=0.9, cut_off=0.05, is_compound=0.05, needs_agent=0.05, wants_element=0.05,
                wants_app=0.05, wants_shortcut=0.05, wants_scroll=0.05, risky=0.05)
    base.update(nouls)
    ch = {"missing": "who", "app": "Notes", "shortcut": "copy", "scroll_direction": "down",
          "element_action": "activate", **(choices or {})}
    return _A(nouls={k: _A(noul=v) for k, v in base.items()}, choices={k: _A(choice=v) for k, v in ch.items()})


def test_truth_table_clarify_and_compound():
    from pupil.control.gate import _decide

    assert _decide(_answers(enough_context=0.04), [], None, {})[0] == "clarify"
    assert _decide(_answers(enough_context=0.04), [], None, {}, allow_clarify=False)[0] == "agent"  # asks once
    assert _decide(_answers(enough_context=0.24, wants_shortcut=0.9), [], None, {})[0] == "shortcut"  # terse is fine
    assert _decide(_answers(is_compound=0.95), [], None, {})[0] == "compound"
    assert _decide(_answers(is_compound=0.95, wants_app=0.9), [], None, {}, allow_compound=False)[0] == "open_app"
    route, _, extras = _decide(_answers(wants_scroll=0.95, wants_shortcut=0.8), [], None, {})
    assert route == "scroll" and extras["args"] == {"direction": "down", "amount": 4}


def test_truth_table_unsure_commands_and_follow_ups():
    from pupil.control.controller import _last_sentence
    from pupil.control.gate import _decide

    # "play the best move": which move is for the screen to settle, not a question
    assert _decide(_answers(enough_context=0.1, needs_agent=0.9, choices={"missing": "what"}), [], None, {})[0] == "agent"
    assert _decide(_answers(enough_context=0.04, choices={"missing": "cut_off"}), [], None, {})[0] == "clarify"
    assert _decide(_answers(is_command=0.5), [], None, {})[0] == "maybe"  # unsure, nothing direct: offer, don't drop
    assert _decide(_answers(is_command=0.2), [], None, {})[0] == "ignore"
    # "knight g2" after "play the best move": a follow-up, so a command
    assert _decide(_answers(is_command=0.5, follows_up=0.9), [], None, {})[0] == "agent"
    assert _decide(_answers(is_command=0.5), [], None, {}, assume_command=True)[0] == "agent"  # "do it"
    assert _decide(_answers(enough_context=0.05, cut_off=0.9, choices={"missing": "what"}), [], None, {})[0] == "clarify"
    assert _last_sentence("is now beginning. Click join.") == "Click join."
    assert _last_sentence("Click join.") is None and _last_sentence("stuff. Yeah.") is None


def test_episodes_label_reactions_and_replay(tmp_path):
    from pupil.control import episodes

    log = episodes.EpisodeLog(tmp_path / "e.jsonl")
    t = time.time()
    a = log.begin("click this", t, t + 0.5, "click")
    a.update(acted=True)
    log.finish(a)
    no = log.begin("no, not that", t + 1, t + 1.3, "unknown")
    log.corrected(no)  # Jev read it as saying the last action was wrong
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
    from pupil.control import controller as ctl
    from pupil.control.gate import Decision

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
    monkeypatch.setattr(ctl.extract, "steps", lambda request: ["open Notes", "write a shopping list"])
    c._busy_with = None
    c._compound("open Notes and write a shopping list", (10, 10), None, [], "Safari")
    assert done == ["/Applications/Notes.app"]
    assert started == ["open Notes and write a shopping list (Already done: open Notes. "
                       "Do the rest: write a shopping list.)"]
    assert [s["route"] for s in c._ep["clauses"]] == ["open_app", "agent"]


def test_resampler_keeps_speech_band_and_blocks_aliasing():
    import numpy as np

    from pupil.control.audio import Resampler

    for f, want in ((1000, 0.707), (11000, 0.0)):  # 11 kHz is above 16 kHz audio's Nyquist: must be removed
        t = np.arange(48000) / 24000
        x = np.sin(2 * np.pi * f * t).astype(np.float32)
        r = Resampler(24000)
        y = np.concatenate([r(x[i:i + 2400]) for i in range(0, len(x), 2400)])  # streamed in 100 ms blocks
        assert abs(len(y) - 32000) <= 32
        assert abs(np.sqrt(np.mean(y[500:-500] ** 2)) - want) < 0.01


def test_speech_detector_ignores_silence_and_noise():
    import numpy as np

    from pupil.control.audio import CHUNK, SpeechDetector

    vad = SpeechDetector()
    rng = np.random.default_rng(0)
    probs = [vad((rng.standard_normal(CHUNK) * 0.02).astype(np.float32)) for _ in range(30)]
    assert max(probs) < 0.5


def test_chunks_carry_capture_times_and_clip_cuts_by_them():
    import queue

    import numpy as np

    from pupil.control.audio import CHUNK, RATE
    from pupil.control.voice import Voice

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
    from pupil.control import config, voice

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

    from pupil.control.episodes import EpisodeLog

    log = EpisodeLog(tmp_path / "e.jsonl")
    ep = log.begin("click this", 0, 1, "click")
    log.save_audio(ep, np.zeros(16000, np.float32))
    with wave.open(str(tmp_path / ep["audio"])) as w:
        assert (w.getframerate(), w.getnframes(), w.getsampwidth()) == (16000, 16000, 2)


def test_scroll_direction_follows_natural_scrolling():
    from pupil.control.actions import SCROLL_PX, scroll_delta

    # measured: with natural scrolling on, +300 px scrolled a window's content down by 300
    assert scroll_delta("down", 4, natural=True) == (4 * SCROLL_PX, 0)
    assert scroll_delta("down", 4, natural=False) == (-4 * SCROLL_PX, 0)
    assert scroll_delta("up", 1, natural=True) == (-SCROLL_PX, 0)
    assert scroll_delta("right", 2, natural=True) == (0, 2 * SCROLL_PX)


def test_drag_sends_down_drags_up_with_the_button_and_modifiers(monkeypatch):
    from pupil.control import actions

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
    from pupil.control import actions

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
    from pupil.control import apps

    opened = []
    monkeypatch.setattr(apps.commands, "available_apps", lambda: {"Notes": "/N.app", "Blender": "/B.app"})
    monkeypatch.setattr(apps, "open_path", lambda name, path: opened.append(path) or f"Opened {name}.")
    assert apps.open_app("notes") == ("Opened Notes.", False) and opened == ["/N.app"]
    assert apps.open_app("blend")[1] is False and opened[-1] == "/B.app"
    assert apps.open_app("Photoshop")[1] is True


def test_blender_python_without_a_bridge_explains_what_to_do(monkeypatch, tmp_path):
    from pupil.control.apps import blender

    monkeypatch.setattr(blender, "TOKEN_FILE", tmp_path / "token")  # no token file: nothing to connect with
    monkeypatch.setattr(blender, "_running", lambda: True)
    assert "wasn't started by voice control" in blender.run("1")["error"]
    monkeypatch.setattr(blender, "_running", lambda: False)
    assert "open_app" in blender.run("1")["error"]


def test_short_text_is_typed_long_or_unicode_text_is_pasted(monkeypatch):
    from pupil.control import actions

    calls = []
    monkeypatch.setattr(actions.pyautogui, "write", lambda s, interval=0: calls.append(("write", s)))
    monkeypatch.setattr(actions.pyautogui, "hotkey", lambda *k: calls.append(("hotkey", k)))
    monkeypatch.setattr(actions, "paste", lambda text: calls.append(("paste", text)))
    actions.type_text("/btw\nok")
    assert calls == [("write", "/btw"), ("hotkey", ("shift", "enter")), ("write", "ok")]  # new line, not send
    calls.clear()
    long = "a whole paragraph of dictation that would take seconds to type key by key"
    actions.type_text(long)
    actions.type_text("café")
    assert calls == [("paste", long), ("paste", "café")]


def test_dictation_join_spacing():
    from pupil.control.dictation import join

    assert join("We test on Monday", "and then we ship it.") == " and then we ship it."
    assert join("Hello ", "world") == "world"
    assert join("Done", ". Next") == ". Next"
    assert join("", "/btw") == "/btw"


def test_menu_paths_match_titles_and_report_what_exists(monkeypatch):
    import pytest

    from pupil.control import uitree

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
    from pupil.control import uitree

    assert uitree.applescript("return 6 * 7") == ("42", False)
    out, error = uitree.applescript("this is not applescript")
    assert error and "error" in out.lower()


def test_adapter_tools_only_when_their_app_is_in_play(monkeypatch):
    from pupil.control import apps

    monkeypatch.setattr(apps, "_running", lambda app: False)
    assert apps.tools_for("make a slide about this") == []
    assert [t["name"] for t in apps.tools_for("build me a car in blender")] == ["blender_python"]
    monkeypatch.setattr(apps, "_running", lambda app: app == "Blender")
    assert [t["name"] for t in apps.tools_for("make the cube red")] == ["blender_python"]
    assert apps.call("not_an_adapter", {}) is None


def test_close_never_guesses_a_whole_window(monkeypatch):
    from pupil.control import controller as ctl
    from pupil.control.snapping import Element

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


def test_menu_commands_are_picked_by_jev_once_per_app(monkeypatch):
    from pupil.control import controller as ctl

    asked, pressed = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._menu_choice, c._busy_with, c._status, c._flash = {}, None, lambda text: None, None
    c.gate = _A(available=True, menu_command=lambda desc, combo, titles: asked.append(titles) or "Select Previous Tab (⌃⇧⇥)")
    monkeypatch.setattr(ctl.uitree, "menu_commands", lambda: ("Chrome", ["Select Previous Tab (⌃⇧⇥)", "Close Tab (⌘W)"]))
    monkeypatch.setattr(ctl.uitree, "press_menu_command", lambda command: pressed.append(command) or "Select Previous Tab")
    c._acted = lambda *a, **k: None
    c._shortcut("previous_tab")
    c._shortcut("previous_tab")
    assert len(asked) == 1 and pressed == ["Select Previous Tab (⌃⇧⇥)"] * 2  # cached for the app

def test_agent_hears_interruptions_and_redirects_when_paused(monkeypatch):
    from pupil.control import agent as agent_mod

    a = agent_mod.Agent.__new__(agent_mod.Agent)
    a._steer, a._steer_lock = [], agent_mod.threading.Lock()
    a._thread, a._paused = None, None
    a.steer("actually make it blue")
    assert a._take_steer() == [{"type": "text", "text": 'The user just said: "actually make it blue"'}]
    assert a._take_steer() == []  # delivered once

    spawned = []
    a._spawn = lambda messages, request, approved: spawned.append((messages, approved))
    messages = [{"role": "user", "content": "delete the old drafts"}]
    a._paused = (messages, [_A(id="t1"), _A(id="t2")], "delete the old drafts")
    assert a.redirect("no, just archive them")
    sent = spawned[0][0][-1]["content"]
    assert [b.get("tool_use_id") for b in sent[:2]] == ["t1", "t2"] and all(b["is_error"] for b in sent[:2])
    assert sent[2]["text"] == 'The user just said: "no, just archive them"' and spawned[0][1] is None
    assert a._paused is None


def test_answers_to_which_one(monkeypatch):
    from pupil.control import controller as ctl
    from pupil.control.heat import GazeCloud
    from pupil.control.snapping import Element

    a = Element("AXRadioButton", "YouTube", 0, 0, 100, 30)
    b = Element("AXRadioButton", "Gaze dashboard", 300, 0, 100, 30)
    done, said = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._ep, c._busy_with, c.hud, c._status = {}, None, _A(badges=[]), said.append
    c._act_on = lambda action, args, target: done.append(target)

    def ask(answer):
        c.choosing = {"options": [a, b], "action": "click", "args": {}}
        c.gate = _A(available=True, pick=lambda text, options: answer)
        c._resolve_choice("whatever was said", GazeCloud([(350.0, 15.0, 1.0)], 20))

    ask(b)
    ask("looked_at")  # "this one": where they're looking now
    ask("none")
    assert done == [b, b] and said[-1] == "Didn't catch which one: nothing done."

def test_relative_correction_redoes_the_action_on_the_neighbour(monkeypatch):
    from pupil.control import controller as ctl
    from pupil.control.snapping import Element

    tabs = [Element("AXRadioButton", f"t{i}", 100 * i, 0, 100, 30, parent_role="AXTabGroup") for i in range(3)]
    done, shortcuts = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._ep, c._status, c._candidates = {}, lambda text: None, []
    c._do = lambda kind, args, gaze, target: done.append((kind, target.label))
    c._close = lambda target: done.append(("close", target.label)) or True
    c._shortcut = lambda name, times=1: shortcuts.append(name)
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    c._last_target = ctl.TargetReference("click", tabs[1], tabs, {}, "", "", ctl.time.time())
    c._relative("left")
    assert done[-1] == ("click", "t0")
    c._last_target = ctl.TargetReference("close", tabs[1], tabs, {}, "", "", ctl.time.time())
    c._relative("right")  # closed the wrong tab: reopen it, then close the right one
    assert shortcuts == ["reopen_tab"] and done[-1] == ("close", "t2")


def test_gaze_indicator_shows_the_most_important_state(monkeypatch):
    from pupil.control import controller as ctl
    from pupil.control.indicator import Status

    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c.enabled, c._flash, c._busy_with, c.pending, c.choosing, c.clarifying, c.dictating = True, None, None, None, None, None, False
    c.maybe = None
    c.speaker = _A(speaking=False)
    c.agent = _A(busy=False, paused=False, phase=None, doing="", _steps=0, tier="standard")
    c.hud = _A(mic="idle")
    assert c._indicator_status() is None  # nothing happening: hidden
    c.hud.mic = "listening"
    assert c._indicator_status().state == "listening"
    c._busy_with = Status("thinking", "thinking")
    assert c._indicator_status().state == "thinking"  # Jev deciding beats listening
    c.agent.busy, c.agent._steps = True, 3
    assert c._indicator_status() == Status("agent", "Agent · step 3")
    c.agent.phase, c.agent.doing = "acting", "left click"
    assert c._indicator_status() == Status("acting", "Agent: left click")
    c.speaker.speaking = True
    assert c._indicator_status().state == "speaking"
    c._flash = (Status("done", "✓ clicked"), ctl.time.time() + 1)
    assert c._indicator_status().label == "✓ clicked"
    c.enabled = False
    assert c._indicator_status() is None


def test_websites_and_searches_go_through_the_address_bar(monkeypatch):
    from pupil.control import controller as ctl

    typed, pressed, opened = [], [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._ep, c._status, c._flash = {}, lambda text: None, None
    c._menu_for = lambda shortcut: "Open Location… (⌘L)"
    monkeypatch.setattr(ctl.actions, "type_text", typed.append)
    monkeypatch.setattr(ctl.actions, "press", lambda combo: pressed.append(combo))
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    monkeypatch.setattr(ctl.subprocess, "Popen", lambda cmd: opened.append(cmd))
    monkeypatch.setattr(ctl.uitree, "press_menu_command", lambda command: "Open Location…")  # a browser in front
    c._navigate(url="cornhub.com")
    assert typed == ["cornhub.com"] and pressed == ["return"] and not opened  # in the window you're in
    monkeypatch.setattr(ctl.uitree, "press_menu_command", lambda command: None)  # not a browser
    c._navigate(query="eye trackers")
    assert opened == [["open", "https://www.google.com/search?q=eye+trackers"]]


def test_quitting_targets_the_named_app_and_always_asks(monkeypatch):
    from pupil.control import commands, controller as ctl
    from pupil.control.snapping import Element

    monkeypatch.setattr(ctl.macos, "running_apps", lambda: [("Google Chrome", 0), ("Games", 1), ("WezTerm", 2)])
    pending, said = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._ep, c._status = {}, lambda text: None
    c.speaker = _A(say=said.append)
    c._set_pending = lambda p, text="": pending.append((p, text))
    c._quit_app("games", force=True)
    assert pending[-1][0][0] == "quit" and pending[-1][0][1]["pid"] == 1 and pending[-1][1] == "Force quit Games"
    dock_icon = Element("AXDockItem", "WezTerm", 0, 0, 50, 50)
    c._quit_app(None, force=False, target=dock_icon)  # "quit that app" looking at its Dock icon
    assert pending[-1][0][1]["name"] == "WezTerm" and pending[-1][1] == "Quit WezTerm"
    assert "quit_app" in commands.ALWAYS_CONFIRM  # Jev's Cmd+Q asks too, whatever its risk score


def test_settings_match_options_by_a_unique_part():
    from pupil.control import settings

    state = {"mic": "automatic"}
    settings.register("microphone", "which mic", lambda: state["mic"], lambda v: state.update(mic=v),
                      lambda: ["automatic", "😨😨😨 AirPods Pro", "MacBook Pro Microphone"])
    assert settings.change("microphone", "airpods") == ("microphone set to '😨😨😨 AirPods Pro'", False)
    assert state["mic"] == "😨😨😨 AirPods Pro"
    text, error = settings.change("microphone", "o")  # matches several: refuse, list the options
    assert error and "MacBook Pro Microphone" in text
    assert settings.change("mic", "automatic")[1] is False  # a setting by part of its name
    assert settings.change("volume", "11")[1] is True
    assert "microphone = 'automatic'" in settings.listing()


def test_chosen_voice_is_remembered(monkeypatch, tmp_path):
    from pupil.control import config, speech

    monkeypatch.setattr(config, "SPEAK", False)  # no network, no audio
    monkeypatch.setattr(speech.Speaker, "_voice_path", staticmethod(lambda: tmp_path / "voice.json"))
    s = speech.Speaker()
    assert s.voice == config.TTS_VOICE
    s.voices = [{"id": "v1", "name": "Daniel", "tagline": "Modern Assistant"}]
    s.set_voice("v1", preview=False)
    assert speech.Speaker().voice == "v1" and s.voice_name == "Daniel"  # a new session starts with it


def test_action_reference_revalidates_context_and_target(monkeypatch):
    from pupil.control import controller as ctl
    from pupil.control.snapping import Element

    old = Element("AXButton", "Run", 100, 100, 80, 30, parent="Toolbar")
    fresh = Element("AXButton", "Run", 102, 100, 80, 30, parent="Toolbar")
    done, status = [], []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c.screen = _A(w_pt=1000, h_pt=800)
    c._last_target = ctl.TargetReference("click", old, [old], {"clicks": 1, "button": "left"},
                                         "Editor", "Project", ctl.time.time())
    c._status = status.append
    c._act_on = lambda action, args, target: done.append((action, target))
    monkeypatch.setattr(ctl.uitree, "front_context", lambda: ("Editor", "Project"))
    monkeypatch.setattr(ctl.snapping, "candidates_near", lambda *args: [fresh])
    c._reference("again", None, None, [])
    assert done == [("click", fresh)]

    monkeypatch.setattr(ctl.uitree, "front_context", lambda: ("Browser", "Other"))
    c._reference("again", None, None, [])
    assert status[-1] == "That reference belongs to a different app or window." and c._last_target is None


def test_same_here_reuses_action_with_current_heat():
    from pupil.control import controller as ctl
    from pupil.control.heat import GazeCloud
    from pupil.control.snapping import Element

    old = Element("AXButton", "Old", 0, 0, 80, 30)
    left = Element("AXButton", "Left", 100, 100, 80, 30)
    right = Element("AXButton", "Right", 400, 100, 80, 30)
    done = []
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c._last_target = ctl.TargetReference("double_click", old, [old], {}, "", "", ctl.time.time())
    c._status = lambda text: None
    c._act_on = lambda action, args, target: done.append((action, target.label))
    c._ask_which = lambda *args: None
    c._reference("same_here", (440, 115), GazeCloud([(440.0, 115.0, 1.0)], 25), [left, right])
    assert done == [("double_click", "Right")]


def test_conversation_reaches_jev_and_the_agent():
    from pupil.control import controller as ctl
    c = ctl.VoiceController.__new__(ctl.VoiceController)
    c.gate = _A(recent=[], last_reply=None)
    c._ep = {"id": "e1", "text": "the quarterly report"}
    c._remember_command("the quarterly report")  # ignored by Jev, but it's what "open that" will mean
    c._ep = {"id": "e2", "text": "click this"}
    c._remember_command("click this")
    c._note_activity("Clicked Pull requests")
    c._converse("assistant", "Which one? One or two?")
    c._remember_command("open that")
    assert c.gate.recent == ["the quarterly report", "click this"]
    assert c.gate.last_reply == "Which one? One or two?"
    context = c._conversation_context("open that")
    assert '- user (just now): "the quarterly report"' in context and "- done: Clicked Pull requests" in context
    assert '- you said: "Which one? One or two?"' in context and "open that" not in context  # this request itself

    # "maybe" waits quietly for "do it": no prompt at the gaze
    c.enabled, c._flash, c._busy_with, c.pending, c.choosing, c.clarifying, c.dictating = True, None, None, None, None, None, False
    c.speaker, c.hud = _A(speaking=False), _A(mic="idle")
    c.agent = _A(busy=False, paused=False, phase=None, doing="", _steps=0, tier="standard")
    c.maybe = ("open the report", ctl.time.time() + 10, None, None, [], {})
    assert c._indicator_status() is None
