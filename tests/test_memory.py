import time
from collections import deque

from pupil.control import router
from pupil.control.controller import VoiceController
from pupil.control.gate import Gate
from pupil.control.memory import MAX_AGE_S, MAX_RECORDS, Memory


def test_recall_matches_an_earlier_decision_and_keeps_the_latest_research(tmp_path):
    memory = Memory(tmp_path / "memory.jsonl")
    memory.remember("user", "compare accessible headphones under $50", app="Safari", window="Shop")
    memory.remember("assistant", "The Sony fits the budget.")
    memory.remember("research", "accessible headphones under $50", cards=[{
        "title": "Sony WH-CH520", "recommendation": "Easy controls, under $50", "url": "https://example.com/sony",
    }])
    recalled = memory.recall("select the one we talked about")
    assert "Sony WH-CH520" in recalled
    assert "accessible headphones under $50" in recalled
    assert memory.latest_item().startswith("Sony WH-CH520")


def test_memory_drops_expired_records_and_keeps_the_newest_cap(tmp_path):
    memory = Memory(tmp_path / "memory.jsonl")
    stale = {"t": time.time() - MAX_AGE_S - 10, "kind": "user", "text": "ancient headphones"}
    memory.path.write_text(__import__("json").dumps(stale) + "\n")
    memory = Memory(tmp_path / "memory.jsonl")
    assert memory.recall("ancient headphones") == ""
    for i in range(MAX_RECORDS + 5):
        memory.remember("user", f"topic {i} keyboard switch")
    assert "topic 0 " not in memory.path.read_text()
    assert "topic 4 " not in memory.path.read_text()
    assert f"topic {MAX_RECORDS + 4}" in memory.recall("keyboard switch")


def test_clear_memory_forgets_and_removes_the_file(tmp_path):
    memory = Memory(tmp_path / "memory.jsonl")
    memory.remember("user", "compare accessible headphones under $50")
    memory.clear()
    assert memory.recall("select the one we talked about") == ""
    assert memory.latest_item() is None
    assert not memory.path.exists()


def test_forget_phrases_are_instant_commands():
    assert router.route("forget that").kind == "forget"
    assert router.route("clear memory").kind == "forget"
    assert router.route("clear your memory").kind == "forget"


def test_agent_context_includes_memory_and_jev_sees_one_line(tmp_path):
    memory = Memory(tmp_path / "memory.jsonl")
    memory.remember("research", "accessible headphones under $50", cards=[{
        "title": "Sony WH-CH520", "recommendation": "under $50", "url": "https://example.com/sony",
    }])
    controller = VoiceController.__new__(VoiceController)
    controller.conversation = deque()
    controller.memory = memory
    context = controller._conversation_context("select the one we talked about")
    assert "Sony WH-CH520" in context
    assert "not the gaze ring" in context

    gate = Gate.__new__(Gate)
    gate.recent, gate.last_reply, gate.remembered = [], None, memory.latest_item()
    state = gate._state("select that", "Safari", None, [])
    assert state["remembered_item"].startswith("Sony WH-CH520")

    memory.clear()
    assert controller._conversation_context("select the one we talked about") == ""


def test_finished_research_is_remembered(tmp_path):
    memory = Memory(tmp_path / "memory.jsonl")
    controller = VoiceController.__new__(VoiceController)
    controller.memory = memory
    controller.gate = Gate.__new__(Gate)
    controller.gate.remembered = None
    controller.gaze = type("G", (), {"report": lambda *a, **k: None})()
    controller._on_research({
        "query": "accessible headphones under $50", "phase": "running", "cards": [],
    })
    assert memory.latest_item() is None  # still running: nothing to act on yet
    controller._on_research({
        "query": "accessible headphones under $50", "phase": "complete",
        "cards": [{"title": "Sony WH-CH520", "recommendation": "under $50", "url": "https://example.com/sony",
                   "quote": "a long untrusted page that must not be stored"}],
    })
    stored = memory.path.read_text()
    assert "Sony WH-CH520" in stored and "untrusted page" not in stored
    assert controller.gate.remembered.startswith("Sony WH-CH520")
