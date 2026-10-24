"""Re-run the config comparison offline on a saved run (no camera needed).

    uv run python evaluate.py                     # latest run in runs/
    uv run python evaluate.py runs/20261023-213000.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import calib
import calibrate_gaze as cg
import geometry as g


class SavedScreen:
    def __init__(self, geo: g.Screen):
        self.geo = geo

    def px(self, f):
        return int(f[0] * self.geo.width_px), int(f[1] * self.geo.height_px)


def calib_file_for(run: Path) -> Path | None:
    """The calibration frames saved most recently before this test run."""
    cands = sorted(p for p in run.parent.glob("*-calib.npz") if p.name[:15] <= run.name[:15])
    return cands[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="a runs/<stamp>.json file (default: latest)")
    args = ap.parse_args()
    runs = sorted(p for p in cg.RUNS.glob("*.json") if (cg.RUNS / f"{p.stem}-test.npz").exists())
    run = Path(args.run) if args.run else (runs[-1] if runs else None)
    if run is None:
        raise SystemExit("no saved runs with frames yet")
    meta = json.loads(run.read_text())
    scr = SavedScreen(g.Screen(**meta["screen"]))
    test_frames, test_points, regions, _ = cg.load_frames(cg.RUNS / f"{run.stem}-test.npz")
    cf = calib_file_for(run)
    if cf is None:
        raise SystemExit("no calibration frames saved before this run")
    calib_frames, calib_points, _, calib_groups = cg.load_frames(cf)
    w, h = meta.get("capture", [1280, 720])
    intr = g.Intrinsics.from_hfov(w, h, meta["hfov_deg"])
    print(f"run {run.name}  calibration {cf.name}  hfov {meta['hfov_deg']:.1f}°  screen {scr.geo.width_px}x{scr.geo.height_px}")

    fix = cg.CalibData(calib_frames, calib_points, calib_groups).fixations(scr)
    raw = cg.summary(cg.score(test_frames, test_points, scr, intr, None, regions))
    print(f"  {'uncalibrated':40s} {raw['accuracy_cm']:.2f} cm  {raw['accuracy_deg']:.2f}°")
    for cfg in calib.ABLATION:
        uc = calib.fit(fix, intr, scr.geo, cfg)
        sm = cg.summary(cg.score(test_frames, test_points, scr, intr, uc, regions))
        regs = "  ".join(f"{k} {v['accuracy_cm']:.2f}" for k, v in sm.get("by_region", {}).items())
        print(f"  {cfg.label():40s} {sm['accuracy_cm']:.2f} cm  {sm['accuracy_deg']:.2f}°  precision {sm['precision_sd_deg']:.2f}°  {regs}"
              f"   gains {uc.mean('gain_yaw'):.2f}/{uc.mean('gain_pitch'):.2f}")


if __name__ == "__main__":
    main()
