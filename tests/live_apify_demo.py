"""Opt-in capped Apify demo rehearsal; intentionally excluded from normal pytest.

Run from the repository root:
    GAZE_LIVE_APIFY=1 uv run python tests/live_apify_demo.py
"""
from __future__ import annotations

import os
import statistics
import threading
import time

from pupil.control.providers.apify import ApifyProvider


if os.environ.get("GAZE_LIVE_APIFY") != "1":
    raise SystemExit("Set GAZE_LIVE_APIFY=1 to spend Apify credits on this live rehearsal.")

runs = int(os.environ.get("GAZE_APIFY_DEMO_RUNS", "10"))
if runs < 1 or runs > 10:
    raise SystemExit("GAZE_APIFY_DEMO_RUNS must be between 1 and 10.")

provider = ApifyProvider()
if not provider.configured:
    raise SystemExit("APIFY_TOKEN is required.")

records = []
for trial in range(runs):
    started = time.monotonic()
    first_status = None
    first_source = None

    def event(state):
        global first_status, first_source
        elapsed = time.monotonic() - started
        if first_status is None:
            first_status = elapsed
        if first_source is None and state.get("cards"):
            first_source = elapsed

    result = provider.call(
        "apify_web_research",
        {
            "query": "Compare WCAG 2.2 keyboard accessibility guidance using primary public sources",
            "constraints": "Prefer W3C primary sources; return three independently sourced records",
            "max_sources": 3,
            "timeout_s": 20,
            "max_charge_usd": 0.25,
        },
        threading.Event(),
        lambda _text: None,
        event,
    )
    elapsed = time.monotonic() - started
    ok = not result.is_error and result.metadata.get("item_count", 0) >= 1
    records.append((ok, first_status or elapsed, first_source or elapsed, elapsed))
    print(f"{trial + 1}/{runs}: {'ok' if ok else 'failed'} in {elapsed:.1f}s")

success = sum(record[0] for record in records) / len(records)
print(
    f"success={success:.0%}; median first visual={statistics.median(r[1] for r in records):.1f}s; "
    f"first source={statistics.median(r[2] for r in records):.1f}s; "
    f"complete={statistics.median(r[3] for r in records):.1f}s"
)
if success < 0.9:
    raise SystemExit("Demo completion was below 90%.")
