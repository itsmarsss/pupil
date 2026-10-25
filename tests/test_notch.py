from pupil.ui import notch_content as nc


def test_collapsed_ear_says_little():
    assert nc.ear_label(nc.NotchModel()) == ""  # idle: no words
    assert nc.ear_label(nc.NotchModel(state="listening", label="listening")) == "Listening"
    assert nc.ear_label(nc.NotchModel(state="agent", label="Agent · step 3 · deep")) == "Agent · 3"
    assert nc.ear_label(nc.NotchModel(state="done", label="✓ clicked")) == "Clicked"


def test_peek_shows_what_needs_reading_most_urgent_first():
    now = 100.0
    m = nc.NotchModel(caption=("Done. 3 results.", 99.0, None))
    assert nc.peek_line(m, now) == ("speech", "Done. 3 results.")
    m.pending = "quit Chrome"
    assert nc.peek_line(m, now) == ("confirm", "Quit Chrome?")
    m.pending, m.caption = None, ("old", 90.0, 95.0)  # said a while ago: gone
    assert nc.peek_line(m, now) is None


def test_headline_reads_like_a_sentence():
    assert nc.headline(nc.NotchModel(voice_on=False))[0] == "Voice control off"
    assert nc.headline(nc.NotchModel(voice_on=True, tracking="no face"))[0] == "Can’t see you"
    title, sub = nc.headline(nc.NotchModel(voice_on=True, state="agent", label="Agent · step 2", heard="book a table"))
    assert title == "Agent · step 2" and sub == "“book a table”"
    assert nc.headline(nc.NotchModel(voice_on=True, tracking="tracking"))[0] == "Ready"


def test_recent_commands_newest_first_and_only_recent():
    m = nc.NotchModel(activity=[{"said": "open notes", "did": "Opened Notes", "t": 0.0},
                                {"said": "click this", "did": "Clicked Pull requests", "t": 900.0},
                                {"said": "summarize this", "did": "3 key points", "t": 990.0, "agent": True}])
    assert nc.recent(m, now=1000.0) == [("summarize this", "3 key points", "now", True),
                                        ("click this", "Clicked Pull requests", "1m", False)]  # 16 min old: gone


def test_actions_are_described_for_people():
    from pupil.control.controller import _describe_action

    assert _describe_action("click", {"clicks": 1, "button": "left"}, "'Pull requests' link") == \
        "Clicked 'Pull requests' link"
    assert _describe_action("open_app", {"app": "Notes"}) == "Opened Notes"
    assert _describe_action("key", {"combo": "command+t", "shortcut": "new_tab"}) == "Open a new tab"
    assert _describe_action("scroll", {"direction": "down", "amount": 4}) == "Scrolled down"
