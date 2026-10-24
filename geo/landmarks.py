"""MediaPipe Face Landmarker (478 face + iris landmarks and the facial transformation matrix)."""
import mediapipe as mp

MODEL = "models/face_landmarker.task"


def make_landmarker():
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)
