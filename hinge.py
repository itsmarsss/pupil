"""MacBook lid (hinge) angle from the built-in lid angle sensor (2019+ MacBook Pros, M-series).

HID device 05AC:8104, sensor usage page 0x20 / usage 0x8A; feature report 1 = [id, angle lo, angle hi] in degrees.

    uv run python hinge.py
"""
from __future__ import annotations

import hid

VID, PID, USAGE_PAGE, USAGE = 0x05AC, 0x8104, 0x20, 0x8A


class LidSensor:
    def __init__(self):
        self.dev = None
        info = next((d for d in hid.enumerate(VID, PID) if d["usage_page"] == USAGE_PAGE and d["usage"] == USAGE), None)
        if info is None:
            return
        try:
            self.dev = hid.device()
            self.dev.open_path(info["path"])
        except OSError:
            self.dev = None

    @property
    def available(self) -> bool:
        return self.dev is not None

    def read(self) -> float | None:
        if self.dev is None:
            return None
        try:
            r = self.dev.get_feature_report(1, 8)
        except OSError:
            return None
        return float(r[2] << 8 | r[1]) if len(r) >= 3 else None


if __name__ == "__main__":
    import time
    s = LidSensor()
    if not s.available:
        raise SystemExit("no lid angle sensor found")
    while True:
        print(f"\rlid angle {s.read():5.1f}°", end="", flush=True)
        time.sleep(0.1)
