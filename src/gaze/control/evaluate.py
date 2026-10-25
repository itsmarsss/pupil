"""`gaze episodes`: look back at logged voice commands, replay routing offline, or score the gate on labelled cases."""
from __future__ import annotations

import collections
import json
import statistics
import time
from pathlib import Path

from . import episodes

CASES = Path(__file__).resolve().parents[3] / "tests" / "agent_cases.jsonl"


def show(n: int = 20) -> int:
    eps = episodes.load()
    if not eps:
        print("no episodes yet: they're logged to .logs/episodes.jsonl as you use voice control")
        return 0
    for ep in eps[-n:]:
        route = ep.get("decision") or ep.get("route", "?")
        did = "; ".join(f"{a['kind']} {a.get('where') or a.get('app') or a.get('combo') or ''}".strip()
                        for a in ep.get("actions", []))
        if ep.get("agent_result"):
            did = f"agent ({ep.get('agent_steps')} steps, {ep.get('agent_s')} s): {ep['agent_result']}"
        when = time.strftime("%H:%M:%S", time.localtime(ep["t"]))
        print(f"{when}  {ep.get('label', '…'):>9}  {route:<12} {ep.get('ms', 0):>6} ms  \"{ep['text']}\"  {did}")
    labels = collections.Counter(ep.get("label", "…") for ep in eps)
    routes = collections.Counter(ep.get("decision") or ep.get("route", "?") for ep in eps)
    print(f"\n{len(eps)} episodes · labels {dict(labels)} · routes {dict(routes)}")
    for route in sorted(routes):
        ms = [ep["ms"] for ep in eps if (ep.get("decision") or ep.get("route")) == route and "ms" in ep]
        if len(ms) >= 3:
            print(f"  {route:<12} p50 {statistics.median(ms):.0f} ms  p95 {sorted(ms)[int(0.95 * (len(ms) - 1))]:.0f} ms")
    return 0


def replay() -> int:
    """Which logged decisions the current truth table would make differently (no API calls)."""
    rows = episodes.replay(episodes.load())
    changed = [(ep, was, now) for ep, was, now in rows if was != now]
    for ep, was, now in changed:
        print(f"{ep.get('label', '…'):>9}  {was} -> {now}  \"{ep['text']}\"")
    print(f"{len(rows)} gate decisions replayed, {len(changed)} would change")
    return 0


def evaluate(path: Path = CASES) -> int:
    """Ask Jev about each labelled case (live, ~0.4 s each) and compare the route with the expected one."""
    from .gate import Gate
    from .snapping import Element

    gate = Gate()
    if not gate.available:
        print(f"Jev unavailable: {gate.error}")
        return 1
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip() and not line.startswith("//")]
    wrong, ms = [], []
    for case in cases:
        elements = [Element(**e) for e in case.get("elements", [])]
        gaze = tuple(case["gaze"]) if case.get("gaze") else None
        d = gate.decide(case["text"], gaze, elements, case.get("app", "Safari"), **case.get("flags", {}))
        ms.append(d.ms)
        expect = case["expect"] if isinstance(case["expect"], list) else [case["expect"]]
        ok = d.route in expect
        if not ok:
            wrong.append((case, d))
        print(f"{'✓' if ok else '✗'} {d.route:<12} (want {'/'.join(expect)})  \"{case['text']}\"  {d.why}")
    print(f"\n{len(cases) - len(wrong)}/{len(cases)} routed as expected · Jev p50 {statistics.median(ms):.0f} ms")
    return 0 if not wrong else 2
