from gaze.ui import notch_content as nc


def _health(**kw):
    base = {"fatigue": 42, "level": "moderate", "blink_rate": 7.0, "minutes_on_screen": 12.0, "breaks": 2,
            "break_every_min": 20, "signals": {"blink_rate": 0.7, "screen_time": 0.0, "perclos": 0.1}}
    base.update(kw)
    return base


def test_collapsed_ear_says_little():
    assert nc.ear_label(nc.NotchModel()) == ""  # idle: no words
    assert nc.ear_label(nc.NotchModel(state="listening", label="listening")) == "Listening"
    assert nc.ear_label(nc.NotchModel(state="agent", label="Agent · step 3 · deep")) == "Agent · 3"
    assert nc.ear_label(nc.NotchModel(state="done", label="✓ clicked")) == "Clicked"


def test_peek_shows_what_needs_reading_most_urgent_first():
    now = 100.0
    m = nc.NotchModel(caption=("Done. 3 results.", 99.0, None), reminder=("Blink a few times.", 105.0))
    assert nc.peek_line(m, now) == ("speech", "Done. 3 results.")
    m.pending = "Quit Chrome"
    assert nc.peek_line(m, now) == ("confirm", "Quit Chrome?")
    m.pending, m.caption = None, ("old", 90.0, 95.0)  # said a while ago: gone
    assert nc.peek_line(m, now) == ("reminder", "Blink a few times.")
    assert nc.peek_line(m, 106.0) is None


def test_headline_reads_like_a_sentence():
    assert nc.headline(nc.NotchModel(voice_on=False))[0] == "Voice control off"
    assert nc.headline(nc.NotchModel(voice_on=True, tracking="no face"))[0] == "Can’t see you"
    title, sub = nc.headline(nc.NotchModel(voice_on=True, state="agent", label="Agent · step 2", heard="book a table"))
    assert title == "Agent · step 2" and sub == "“book a table”"
    assert nc.headline(nc.NotchModel(voice_on=True, tracking="tracking"))[0] == "Ready"


def test_health_tiles_explain_the_score():
    fatigue, blinks, brk = nc.tiles(nc.NotchModel(health=_health()))
    assert (fatigue.value, fatigue.unit, fatigue.detail, fatigue.tone) == ("42", "", "few blinks", "warn")
    assert (blinks.value, blinks.tone) == ("7", "bad")
    assert (brk.value, brk.unit, brk.tone) == ("8", "min", "ok") and abs(brk.progress - 0.6) < 1e-9
    _, _, due = nc.tiles(nc.NotchModel(health=_health(minutes_on_screen=21.0)))
    assert due.value == "Now" and due.tone == "warn"
    fatigue, _, _ = nc.tiles(nc.NotchModel(health={}))
    assert fatigue.value == "–" and fatigue.detail == "measuring…"


def test_sparkline_keeps_the_last_half_hour():
    pts = nc.sparkline([(0.0, 50), (1000.0, 20), (2000.0, 80)], now=2000.0)  # the first is 33 min old
    assert len(pts) == 2 and pts[-1] == (1.0, 0.8) and pts[0][1] == 0.2
