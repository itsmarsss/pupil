"""Turn a saved session into a dashboard replay: the app's own pipeline and calibration, frame by frame."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..core.device import load_device
from ..core.types import Frame
from ..eval.replay import _live_profile, extract, load
from ..eval.session import read_session
from ..face.normalize import NormalisedModel
from ..face.tracker import FaceTracker
from ..telemetry import calibration_info, frame_state, scene_info
from ..tracker import Tracker
from .feeds import ReplayFeed, ReplayFrame, crops_jpeg, jpeg

log = logging.getLogger(__name__)


def build_replay(root: Path, model, normalise: str = "roll") -> ReplayFeed:
    screen, timings = load(root)
    # 1. fit the app's calibration on the session's own dots and head-movement step
    data = extract(root, model, f"hd+{normalise}" if normalise != "none" else "hd", timings, screen)
    fit_on = [d for d in data if d.phase in ("calib", "motion") and len(d.features)]
    if not fit_on:
        raise ValueError(f"{root.name} has no calibration frames to fit on")
    profile = _live_profile(fit_on, screen)

    # 2. run every saved frame through the tracker with that calibration
    gaze_model = NormalisedModel(model, normalise) if normalise != "none" else model
    tracker = Tracker(source=None, face_tracker=FaceTracker(), model=gaze_model, screen=screen, profile=profile)
    frames, lid = [], None
    for sf in read_session(root):
        if sf.image is None:
            continue
        lid = sf.lid_deg if sf.lid_deg is not None else lid
        sample = tracker.process(Frame(sf.t_ns, sf.image))
        context = {"phase": sf.phase, "target": list(sf.target), "busy": True, "lid_deg": lid,
                   "smoothing": tracker.smoothing, "overlay": True}
        state = frame_state(sample, profile, tracker.plane, context, fps=30.0)
        state["latency_ms"] = None  # not meaningful in a replay
        crops = crops_jpeg(getattr(gaze_model, "last_inputs", None)) if sample.features is not None else b""
        frames.append(ReplayFrame(json.dumps(state), jpeg(sf.image), crops))
    tracker.face_tracker.close()
    log.info("replay of %s: %d frames", root.name, len(frames))
    scene = scene_info(screen, tracker.plane, load_device(), lid)
    return ReplayFeed(frames, scene, calibration_info(profile), root.name)
