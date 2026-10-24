"""Gaze model interface. Every model maps a frame + face to a feature vector."""
from __future__ import annotations

from typing import Protocol

from ..core.types import FaceObservation, Frame, GazeFeatures


class GazeModel(Protocol):
    # Identifies the feature space; a calibration is only valid for the model_id it was fit on.
    model_id: str

    def infer(self, frame: Frame, face: FaceObservation) -> GazeFeatures | None: ...

    def close(self) -> None: ...
