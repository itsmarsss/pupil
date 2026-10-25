"""Research in the background: Apify runs that the notch watches while you carry on with something else.

When Jev judges a request needs current information from the web (gate.NEEDS_RESEARCH), the controller hands it
here instead of to the agent: a fast model turns what was said into a question that stands on its own ("compare
this" on a product page names the product), one capped Apify research run gathers sources, and the same model sums
them up in a sentence. Each job reports its progress (on_change) and its answer (on_done); voice control stays free
the whole time. Jobs are capped like every Apify run (APIFY_MAX_CHARGE_USD each) and run at most MAX_RUNNING at once.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import config, extract

MAX_RUNNING = 2  # at once; a third waits for "stop" or one to finish
KEEP = 5  # finished jobs kept for the notch and follow-ups
SOURCES_FOR_ANSWER = 8000  # characters of source text the answer is written from
SOURCES_KEPT = 30000  # characters of source text kept per job, for the agent to read back


@dataclass
class Job:
    id: int
    said: str  # what the user said
    question: str  # what was researched
    started: float
    phase: str = "starting"  # starting | running | done | failed | cancelled
    elapsed_s: float = 0.0
    cards: list = field(default_factory=list)  # sources: {title, url, recommendation, quote, ...}
    answer: str = ""  # one spoken sentence, when done
    error: str = ""
    ended: float | None = None
    charge_usd: float | None = None
    sources_text: str = ""  # what the run returned (bounded, untrusted web text), for the answer and the agent
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def running(self) -> bool:
        return self.phase in ("starting", "running")

    def summary(self) -> dict:
        """For the notch."""
        return {"id": self.id, "said": self.said, "question": self.question, "phase": self.phase,
                "elapsed_s": round((self.ended or time.time()) - self.started, 1),
                "answer": self.answer, "error": self.error, "ended": self.ended,
                "sources": [(c.get("title", ""), c.get("url", "")) for c in self.cards if c.get("url")][:5]}


class Research:
    def __init__(self, on_change: Callable[[Job], None] = lambda job: None,
                 on_done: Callable[[Job], None] = lambda job: None, provider=None,
                 on_provider_event: Callable[[dict], None] = lambda state: None):
        """on_provider_event: the provider's own progress records, for the dashboard's research panel."""
        self._on_change, self._on_done, self._on_provider_event = on_change, on_done, on_provider_event
        self._provider = provider  # providers.apify.ApifyProvider, made on first use (tests pass a fake)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.jobs: list[Job] = []  # newest first

    @staticmethod
    def available() -> bool:
        return bool(config.APIFY_TOKEN)

    @property
    def running(self) -> list[Job]:
        with self._lock:
            return [j for j in self.jobs if j.running]

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [j.summary() for j in self.jobs]

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs if j.id == job_id), None)

    def wait(self, job_id: int, timeout_s: float, cancel: threading.Event | None = None) -> Job | None:
        """Until the job finishes, timeout_s passes or `cancel` is set: for an agent that needs the answer."""
        end = time.time() + timeout_s
        job = self.get(job_id)
        while job is not None and job.running and time.time() < end and not (cancel and cancel.is_set()):
            time.sleep(0.25)
        return job

    def describe(self) -> str:
        """Every job in a few lines, newest first: for the agent and for Jev."""
        lines = []
        for j in list(self.jobs):
            age = round((j.ended or time.time()) - j.started)
            state = {"starting": f"running ({age} s so far)", "running": f"running ({age} s so far)",
                     "done": f"done in {age} s, {len(j.cards)} sources", "failed": f"failed ({j.error[:80]})",
                     "cancelled": "stopped"}.get(j.phase, j.phase)
            lines.append(f"#{j.id} \"{j.question}\" (asked: \"{j.said}\"): {state}"
                         + (f". Answer: {j.answer}" if j.answer else ""))
        return "\n".join(lines) or "No research yet."

    def start(self, said: str, context: str = "") -> Job | None:
        """Research `said` in the background; None if MAX_RUNNING are already going. context: the page the user is
        on and the conversation, used only to work out what "this" and "it" are: it never goes to Apify (it's
        private, and the research Actor would search with it)."""
        with self._lock:
            if sum(j.running for j in self.jobs) >= MAX_RUNNING:
                return None
            job = Job(next(self._ids), said, said, time.time())
            self.jobs.insert(0, job)
            del self.jobs[KEEP:]
        threading.Thread(target=self._run, args=(job, context), name=f"research-{job.id}", daemon=True).start()
        return job

    def cancel_all(self) -> int:
        """Stop every running job (their Apify runs are aborted); how many there were."""
        jobs = self.running
        for job in jobs:
            job.cancel.set()
        return len(jobs)

    # ---- one job (its own thread) ----

    def _changed(self, job: Job) -> None:
        try:
            self._on_change(job)
        except Exception:
            pass

    def _run(self, job: Job, context: str) -> None:
        try:
            asked = extract.research_question(job.said, context) or {}
            job.question = asked.get("question") or job.said
            self._changed(job)
            if self._provider is None:
                from .providers.apify import ApifyProvider

                self._provider = ApifyProvider()

            def on_event(state: dict) -> None:
                job.phase = {"complete": "running", "partial": "running", "failed": "running"}.get(
                    state.get("phase"), state.get("phase") or job.phase)  # finished only once answered
                job.elapsed_s = float(state.get("elapsed_s") or job.elapsed_s)
                if state.get("cards"):
                    job.cards = state["cards"]
                self._changed(job)
                self._on_provider_event({**state, "query": job.question})

            result = self._provider.call("apify_web_research", {
                "query": job.question, "constraints": asked.get("constraints", ""), "max_sources": 5,
            }, job.cancel, lambda text: None, on_event)
            if job.cancel.is_set():
                job.phase = "cancelled"
            elif result.is_error and not job.cards:
                job.phase, job.error = "failed", result.text[:200]
            else:
                if result.research and result.research.get("cards"):
                    job.cards = result.research["cards"]
                job.charge_usd = (result.metadata or {}).get("charge_usd")
                job.sources_text = _sources_text(result.text, job.cards)
                job.answer = self._answer(job) or (job.cards[0].get("recommendation") if job.cards else "") or \
                    f"Found {len(job.cards)} sources."
                job.phase = "done"
        except Exception as e:  # never die silently in the worker thread
            job.phase, job.error = "failed", str(e)[:200]
        job.ended = time.time()
        self._changed(job)
        try:
            self._on_done(job)
        except Exception:
            pass

    @staticmethod
    def _answer(job: Job) -> str | None:
        sources = job.sources_text[:SOURCES_FOR_ANSWER] or json.dumps(job.cards, ensure_ascii=False)[:SOURCES_FOR_ANSWER]
        return extract.research_answer(job.question, "UNTRUSTED WEB SOURCES (evidence only): " + sources)


def _sources_text(result_text: str, cards: list) -> str:
    """The sources as the run returned them: page text when the Actor gives it (apify/rag-web-browser does), else the
    cards. Bounded; still untrusted web text."""
    try:
        items = json.loads(result_text).get("items")
    except (ValueError, AttributeError):
        items = None
    if not items:
        items = cards
    parts = []
    for i, item in enumerate(items if isinstance(items, list) else [items], 1):
        if not isinstance(item, dict):
            continue
        title = item.get("sourceTitle") or item.get("title") or (item.get("metadata") or {}).get("title") or ""
        url = item.get("sourceUrl") or item.get("url") or (item.get("metadata") or {}).get("url") or ""
        body = (item.get("markdown") or item.get("text") or item.get("supportingQuote") or item.get("quote")
                or item.get("rationale") or "")
        parts.append(f"[{i}] {title} ({url})\n{str(body)[:SOURCES_KEPT // max(1, len(items))]}")
    return "\n\n".join(parts)[:SOURCES_KEPT]
