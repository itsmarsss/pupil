"""Durable facts from voice and research, so a later command can refer to what was decided.

The live transcript in the controller is only the last few minutes. This store keeps truncated requests,
replies, and research cards in `.logs/memory.jsonl` for about a week, and recalls the few that match a
new request. It never stores tokens, cookies, or raw page dumps.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

from . import config

MAX_RECORDS = 200
MAX_AGE_S = 7 * 24 * 3600
MAX_TEXT = 400
_SKIP = {
    "this", "that", "with", "from", "about", "select", "the", "and", "for", "one", "item",
    "talked", "page", "while", "what", "were", "your", "have", "just", "then", "them",
}


def _clip(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text[:MAX_TEXT]


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if len(word) > 3 and word not in _SKIP}


class Memory:
    def __init__(self, path: Path | None = None):
        self.path = path or config.LOG_DIR / "memory.jsonl"
        self._lock = threading.Lock()
        self._records = self._load()

    def remember(self, kind: str, text: str, *, app: str = "", window: str = "", page: str = "",
                 cards: list | None = None) -> None:
        text = _clip(text)
        if not text and not cards:
            return
        record = {"t": time.time(), "kind": kind, "text": text, "app": _clip(app)[:80],
                  "window": _clip(window)[:120], "page": _clip(page)[:200]}
        if cards:
            record["cards"] = [_card(card) for card in cards[:5] if _card(card)]
        with self._lock:
            self._records.append(record)
            self._prune()
            self._write()

    def recall(self, request: str, limit: int = 6) -> str:
        """A short block for the agent: overlapping facts, plus the latest research decision."""
        wanted = _tokens(request)
        refers = bool(re.search(r"\b(talked|remember|decided|discussed|earlier|before)\b", request.lower()))
        with self._lock:
            records = list(self._records)
        ranked = []
        for record in records:
            score = len(wanted & _tokens(_search_text(record)))
            if score:
                ranked.append((score, record["t"], record))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = [record for _, _, record in ranked[:limit]]
        if refers:  # "the one we talked about" has no content words; keep the latest exchange
            for record in reversed(records):
                if record.get("kind") in ("user", "assistant") and record not in chosen:
                    chosen.append(record)
                    break
        latest = next((record for record in reversed(records) if record.get("kind") == "research" and record.get("cards")), None)
        if latest and latest not in chosen:
            chosen.append(latest)
        return "\n".join(f"- {_line(record)}" for record in chosen[-limit:])

    def latest_item(self) -> str | None:
        """One line for Jev: the newest research pick, else the newest user request."""
        with self._lock:
            records = list(self._records)
        for record in reversed(records):
            cards = record.get("cards") or []
            if cards:
                title = cards[0].get("title") or cards[0].get("recommendation")
                topic = record.get("text")
                if title and topic:
                    return _clip(f"{title} ({topic})")[:180]
                return _clip(title or topic) or None
        for record in reversed(records):
            if record.get("kind") == "user" and record.get("text"):
                return record["text"][:180]
        return None

    def clear(self) -> None:
        with self._lock:
            self._records = []
            self.path.unlink(missing_ok=True)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("t"):
                records.append(record)
        cutoff = time.time() - MAX_AGE_S
        return [record for record in records if record["t"] >= cutoff][-MAX_RECORDS:]

    def _prune(self) -> None:
        cutoff = time.time() - MAX_AGE_S
        self._records = [record for record in self._records if record["t"] >= cutoff][-MAX_RECORDS:]

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = "".join(json.dumps(record, default=str) + "\n" for record in self._records)
        self.path.write_text(body)


def _card(card: dict) -> dict | None:
    if not isinstance(card, dict):
        return None
    kept = {"title": _clip(card.get("title") or "")[:180],
            "recommendation": _clip(card.get("recommendation") or "")[:240],
            "url": _clip(card.get("url") or "")[:300]}
    return kept if any(kept.values()) else None


def _search_text(record: dict) -> str:
    parts = [record.get("text", ""), record.get("page", ""), record.get("window", "")]
    for card in record.get("cards") or []:
        parts.extend(card.get(key, "") for key in ("title", "recommendation"))
    return " ".join(parts)


def _line(record: dict) -> str:
    where = ", ".join(part for part in (record.get("app"), record.get("window") or record.get("page")) if part)
    text = record.get("text") or ""
    cards = record.get("cards") or []
    if cards:
        picks = "; ".join(
            " — ".join(part for part in (card.get("title"), card.get("recommendation"), card.get("url")) if part)
            for card in cards[:3]
        )
        text = f"{text}: {picks}" if text else picks
    return f"{record.get('kind', 'note')}: {text}" + (f" [{where}]" if where else "")
