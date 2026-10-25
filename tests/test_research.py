import threading
import time

from pupil.control import research as R
from pupil.control.providers.apify import ProviderResult
from pupil.ui import notch_content as nc


class FakeProvider:
    def __init__(self, cards, wait=None):
        self.cards, self.wait, self.calls = cards, wait, []

    def call(self, name, inp, cancel, status, event):
        self.calls.append((name, inp))
        event({"phase": "running", "elapsed_s": 1.0, "cards": []})
        if self.wait is not None:
            self.wait.wait(2)
        if cancel.is_set():
            return ProviderResult("Apify run cancelled.", True)
        event({"phase": "complete", "elapsed_s": 2.0, "cards": self.cards})
        return ProviderResult("{}", False, {"charge_usd": 0.03}, {"cards": self.cards})


def _wait(job, timeout=3.0):
    end = time.time() + timeout
    while job.running or job.ended is None:
        if time.time() > end:
            raise AssertionError(f"job stuck in {job.phase}")
        time.sleep(0.01)


def test_research_runs_in_the_background_and_answers(monkeypatch):
    monkeypatch.setattr(R.extract, "research_question", lambda said, context: {
        "question": "Compare the Sony WH-1000XM6 with three alternatives", "constraints": "under $400"})
    monkeypatch.setattr(R.extract, "research_answer", lambda q, sources: "Bose QC Ultra edges it on comfort.")
    cards = [{"title": "Review", "url": "https://example.com/r", "recommendation": "Bose"}]
    done, provider = [], FakeProvider(cards)
    research = R.Research(on_done=done.append, provider=provider)
    job = research.start("compare this with three alternatives", "Page they're on: Sony WH-1000XM6 (https://x)")
    _wait(job)
    assert job.phase == "done" and job.answer == "Bose QC Ultra edges it on comfort." and done == [job]
    name, inp = provider.calls[0]
    assert name == "apify_web_research" and inp["query"].startswith("Compare the Sony") and "Sony" in inp["selected_text"]
    summary = research.snapshot()[0]
    assert summary["sources"] == [("Review", "https://example.com/r")] and summary["phase"] == "done"


def test_at_most_two_at_once_and_stop_aborts_them(monkeypatch):
    monkeypatch.setattr(R.extract, "research_question", lambda said, context: None)
    gate = threading.Event()
    research = R.Research(provider=FakeProvider([], wait=gate))
    a, b = research.start("one"), research.start("two")
    assert research.start("three") is None  # both slots busy
    assert research.cancel_all() == 2
    gate.set()
    _wait(a), _wait(b)
    assert a.phase == b.phase == "cancelled"


def test_notch_shows_research_status_and_answer():
    now = 100.0
    running = {"id": 2, "question": "Best eye trackers", "phase": "running", "elapsed_s": 12.4, "answer": "",
               "ended": None, "sources": []}
    done = {"id": 1, "question": "Cheapest XM6", "phase": "done", "elapsed_s": 30, "answer": "Amazon, $329.",
            "ended": 95.0, "sources": [("Deal", "https://example.com/d")]}
    m = nc.NotchModel(research=[running, done])
    assert nc.ear_state(m) == "researching" and nc.ear_label(m) == "Researching"
    assert nc.peek_line(m, now) == ("research", "Amazon, $329.")
    assert nc.peek_line(m, now + 20) is None  # shown for a while, then just in the open notch
    rows = nc.research_rows(m, now)
    assert [r["status"] for r in rows] == ["12s", "1 source"] and rows[1]["url"] == "https://example.com/d"
    m.state = "listening"  # voice control comes first in the ears
    assert nc.ear_state(m) == "listening"
