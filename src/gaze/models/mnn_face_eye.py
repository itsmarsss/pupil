"""MNN model with face + two eye crops + box geometry as input.

Inputs (NHWC float32 in [0, 1]): face 224², image-left eye 112², image-right eye 112²
(horizontally flipped), and a 12-d vector of face/eye boxes normalised by frame size.
Output: a feature vector whose first two values are the model's own gaze estimate.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from ..core.paths import GAZE_MODEL_FILE, model_path
from ..core.types import FaceObservation, Frame, GazeFeatures
from ..face.tracker import crop

FACE_SIZE = (224, 224)
EYE_SIZE = (112, 112)


def _file_id(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"{path.stem}@{digest}"


class MNNFaceEyeModel:
    def __init__(self, path: str | Path | None = None, threads: int = 4):
        import MNN

        self.path = Path(path) if path else model_path(GAZE_MODEL_FILE)
        if not self.path.exists():
            raise FileNotFoundError(f"Gaze model not found at {self.path}. Set GAZE_MODELS_DIR or copy it there.")
        self.model_id = _file_id(self.path)
        rt = MNN.nn.create_runtime_manager(({"precision": "low", "backend": 0, "numThread": threads},))
        self._module = MNN.nn.load_module_from_file(
            str(self.path), ["face", "left", "right", "rect"], ["output_0"], runtime_manager=rt
        )
        self._face = MNN.expr.placeholder((1, *FACE_SIZE, 3), MNN.expr.NHWC)
        self._left = MNN.expr.placeholder((1, *EYE_SIZE, 3), MNN.expr.NHWC)
        self._right = MNN.expr.placeholder((1, *EYE_SIZE, 3), MNN.expr.NHWC)
        self._rect = MNN.expr.placeholder((1, 12))
        # The newest crops fed to the network (face, image-left eye, image-right eye mirrored), RGB in [0, 1].
        # Kept for visualisation; replaced by reference each frame, so readers on other threads see a whole set.
        self.last_inputs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @staticmethod
    def prepare(frame: Frame, face: FaceObservation):
        """Crops and geometry vector, or None if any crop is empty."""
        img = frame.image
        patches = [crop(img, r) for r in (face.face_rect, face.left_eye_rect, face.right_eye_rect)]
        if any(p is None or p.size == 0 for p in patches):
            return None
        face_p, left_p, right_p = patches
        w, h = face.frame_size
        boxes = []
        for x, y, bw, bh in (face.face_rect, face.left_eye_rect, face.right_eye_rect):
            boxes += [bw, bh, x, y]
        geom = (np.array(boxes, dtype=np.float32) / np.array([w, h] * 6, dtype=np.float32)).reshape(1, 12)
        # INTER_AREA when shrinking high-res crops avoids aliasing; INTER_LINEAR when enlarging.
        def resize(p, size):
            interp = cv2.INTER_AREA if p.shape[1] > size[0] else cv2.INTER_LINEAR
            return cv2.resize(p, size, interpolation=interp).astype(np.float32) / 255.0

        face_in = resize(face_p, FACE_SIZE)
        left_in = resize(left_p, EYE_SIZE)
        right_in = cv2.flip(resize(right_p, EYE_SIZE), 1)
        return face_in, left_in, right_in, geom

    def infer(self, frame: Frame, face: FaceObservation) -> GazeFeatures | None:
        inputs = self.prepare(frame, face)
        if inputs is None:
            return None
        face_in, left_in, right_in, geom = inputs
        self.last_inputs = (face_in, left_in, right_in)
        self._face.write(face_in)
        self._left.write(left_in)
        self._right.write(right_in)
        self._rect.write(geom)
        out = self._module.onForward([self._face, self._left, self._right, self._rect])
        if not out:
            return None
        vec = np.array(out[0].read(), dtype=np.float32).reshape(-1)
        return GazeFeatures(face.t_ns, vec, (float(vec[0]), float(vec[1])), self.model_id)

    def close(self) -> None:
        pass
