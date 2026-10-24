"""Read real device geometry from macOS: display physical size / notch and camera field of view.

    uv run python device.py          # print and write device.json
"""
from __future__ import annotations

import json
from pathlib import Path

import AppKit
import AVFoundation
import CoreMedia
import Quartz

DEVICE_JSON = Path(__file__).with_name("device.json")
# Prior when the camera reports no FOV and none was measured: typical fixed-focus 1080p laptop webcam at 16:9.
# No third-party measurement of the MacBook notch camera exists (searched Oct 2026); plausible range 55-70°.
DEFAULT_HFOV_DEG = 62.0


def display_info() -> dict:
    did = Quartz.CGMainDisplayID()
    size_mm = Quartz.CGDisplayScreenSize(did)  # physical panel size
    mode = Quartz.CGDisplayCopyDisplayMode(did)
    screen = AppKit.NSScreen.mainScreen()
    frame = screen.frame()  # points (CSS-px equivalent), includes the notch strip
    safe_top = screen.safeAreaInsets().top if hasattr(screen, "safeAreaInsets") else 0.0
    mm_per_pt = size_mm.width / frame.size.width
    usable_h_pt = frame.size.height - safe_top  # fullscreen apps sit below the notch
    return {
        "builtin": bool(Quartz.CGDisplayIsBuiltin(did)),
        "panel_mm": [size_mm.width, size_mm.height],
        "frame_pt": [frame.size.width, frame.size.height],
        "pixels": [Quartz.CGDisplayModeGetPixelWidth(mode), Quartz.CGDisplayModeGetPixelHeight(mode)],
        "backing_scale": screen.backingScaleFactor(),
        "safe_area_top_pt": safe_top,
        "mm_per_pt": mm_per_pt,
        "px_per_cm_pt": 10 / mm_per_pt,
        # usable rectangle (what a fullscreen window covers), in points and cm
        "usable_pt": [frame.size.width, usable_h_pt],
        "usable_cm": [frame.size.width * mm_per_pt / 10, usable_h_pt * mm_per_pt / 10],
    }


def camera_info() -> list[dict]:
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    out = []
    for dev in devices:
        formats = []
        for f in dev.formats():
            dims = CoreMedia.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
            formats.append({"width": dims.width, "height": dims.height, "hfov_deg": float(f.videoFieldOfView())})
        # de-duplicate (formats repeat per pixel format / frame-rate range)
        uniq = {(x["width"], x["height"], round(x["hfov_deg"], 3)): x for x in formats}
        out.append({
            "name": str(dev.localizedName()),
            "model_id": str(dev.modelID()),
            "unique_id": str(dev.uniqueID()),
            "formats": sorted(uniq.values(), key=lambda x: (x["width"], x["height"])),
        })
    return out


def probe() -> dict:
    disp = display_info()
    cams = camera_info()
    # Built-in MacBook camera sits in the notch, centred horizontally. Assume its centre is halfway down
    # the notch strip, i.e. half the safe-area inset above the top of the usable area. Measure and
    # override in device.json if needed.
    cam_to_top_cm = disp["safe_area_top_pt"] / 2 * disp["mm_per_pt"] / 10 if disp["safe_area_top_pt"] else 0.5
    return {
        "display": disp,
        "cameras": cams,
        "camera_placement": {"cam_to_top_cm": round(cam_to_top_cm, 3), "cam_x_offset_cm": 0.0,
                             "note": "estimated from the notch; measure to refine"},
        # horizontal FOV of the camera at a 16:9 capture, set by calibrate_fov.py when the OS doesn't report it
        "camera_hfov_deg": None,
    }


def load() -> dict:
    if not DEVICE_JSON.exists():
        info = probe()
        DEVICE_JSON.write_text(json.dumps(info, indent=2))
        return info
    return json.loads(DEVICE_JSON.read_text())


def save(info: dict) -> None:
    DEVICE_JSON.write_text(json.dumps(info, indent=2))


def camera_index(cameras: list[dict], name: str = "FaceTime") -> int:
    """OpenCV's AVFoundation index for the first camera whose name contains `name` (same order as AVFoundation)."""
    for i, c in enumerate(cameras):
        if name.lower() in c["name"].lower():
            return i
    return 0


def hfov_for(cameras: list[dict], width: int, height: int, name: str | None = None) -> float | None:
    """FOV of the camera format matching the capture size (exact match, else same aspect ratio, widest)."""
    cams = [c for c in cameras if name is None or name in c["name"]] or cameras
    fmts = cams[0]["formats"] if cams else []
    fmts = [f for f in fmts if f["hfov_deg"] > 0]  # some cameras (e.g. FaceTime HD on M3) report 0
    exact = [f for f in fmts if f["width"] == width and f["height"] == height]
    if exact:
        return exact[0]["hfov_deg"]
    same_aspect = [f for f in fmts if abs(f["width"] / f["height"] - width / height) < 0.01]
    return max(same_aspect, key=lambda f: f["width"])["hfov_deg"] if same_aspect else None


if __name__ == "__main__":
    info = probe()
    DEVICE_JSON.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nwrote {DEVICE_JSON}")
