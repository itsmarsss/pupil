# geo — model-based gaze on macOS

Geometric gaze estimation: MediaPipe face + iris landmarks → 3D eyeball model → gaze ray → intersection with the
screen plane, using real device geometry read from macOS (no credit card, no FOV guess).

```
mkdir -p models && curl -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
uv sync
uv run python device.py                         # probe display + camera, writes device.json
uv run python calibrate_fov.py --paper a4       # once per camera: measure its FOV (or --paper letter)
uv run python live.py                           # live 3D view in Rerun + control window: C calibrate, T test, F free look
uv run python calibrate_gaze.py                 # the same calibration/test without the 3D view
uv run python calibrate_gaze.py --mouse         # mouse-follow calibration instead of dots
uv run python hinge.py                          # print the live lid angle
uv run python evaluate.py                       # re-score the latest saved run offline
uv run python test_geometry.py && uv run python test_calib.py
```

No tape measure. The FaceTime HD camera on M3 MacBook Pros reports a 0° FOV through AVFoundation and Apple
doesn't publish it, so `calibrate_fov.py` measures it from your iris size (~1.17 cm in adults) at a known
distance, using a sheet of paper from the screen to your brow as the spacer. Fitting the FOV from gaze instead
doesn't work: it can't be separated from the per-user gains, even if you lean in and out (see `test_calib.py`).

`calibrate_gaze.py` (or C in `live.py`) fits, per user and in the head frame, gaze angle = gain × measured angle +
kappa for yaw and pitch. On top of that physical model, each fix below can be switched on or off (`calib.GazeConfig`);
every accuracy test fits and scores all of them on the same frames and prints the comparison. Across 5 real
test sessions only the quadratic residual correction helped (dots calibration: ~1.1 cm / ~1°), so it is the
only one on by default:

| Fix | What it addresses |
|---|---|
| blink rejection | frames with eye openness < 60% of your calibration median are dropped |
| equator iris centre | vertical iris position from the iris's left/right edge points; eyelids drag MediaPipe's centre point (raise your lids and the reading moves) |
| per-eye | separate kappa/gains per eye, then the two screen hits are averaged |
| rigid head pose | head rotation from nose/forehead/cheekbone landmarks against a per-user reference, so eyelid motion can't leak into head pitch |
| residual correction | screen-space correction after the physical fit: `affine`, or `quad` (default; quadratic in x, y), which fixed the curved leftover error in real runs (2.02 → 1.48 cm on a 13-dot run) |
| head compensation | on by default when the calibration has head-sweep frames: corrects how the error changes when your head moves away from the calibration pose (dot calibration alone can't learn it; the head is still) |

Mouse-follow calibration: move the mouse slowly and keep your eyes on the dot under it; the background is an
8×5 grid of tiles that turn green as they collect enough frames. Frames are only used while the mouse moves slowly
(fast moves = eyes jumping), and are matched to where the mouse was 0.1 s earlier (eyes trail a moving target).
Tiles are the averaging groups for the fit.

Face pointing: a ray from the nose tip along the head's forward axis; where it meets the screen is drawn as a
yellow/orange marker (3D view, free look with H, on-screen overlay with H in `live.py`). It also guides the head
sweep that ends each dot calibration: eyes stay on a dot while you steer the face-pointing ring around a circle
(3 dots, ~7 s each; `--no-sweep` skips it).

The fit also drops glances/saccades per dot, averages each dot's frames before fitting (fitting single noisy
frames biases the gains low), and down-weights outlier dots. Display output (overlay, free look) uses a One Euro
filter. Raw frames of every calibration and test are saved in `runs/`; `uv run python evaluate.py [run.json]`
re-runs the comparison offline.

`live.py` keys (control window): **C** calibrate (25 dots) · **K** quick calibration (13 dots) · **M** mouse-follow calibration · **T** test + comparison · **F** free look · **D** drift
correction (one centre dot) · **V** cycle config · **O** gaze overlay · **H** face-pointing overlay · **S** smoothing · **Q** quit.

## Frame and conventions

Camera space, centimetres, OpenGL-style (Rerun `RUB`): +x = image right (the user's **left**; webcam frames are not
mirrored), +y up, +z out of the lens toward the user's back. The camera is the origin, the face is at negative z,
and the screen is the plane z = 0 hanging just below the camera. The hinge angle is drawn but doesn't affect the
maths: the camera and the screen are both in the lid.

## Where each number comes from

| Quantity | Source |
|---|---|
| Physical display size | `CGDisplayScreenSize` (mm) |
| Hinge angle (drawing only) | lid angle sensor, HID 05AC:8104 usage 0x20/0x8A, feature report 1 |
| Usable area (below the notch) | `NSScreen.frame` minus `safeAreaInsets.top` |
| Camera horizontal FOV | `AVCaptureDeviceFormat.videoFieldOfView`, or `calibrate_fov.py` (iris size + paper spacer) when that reports 0 |
| Camera position | assumed centred in the notch; override `camera_placement` in `device.json` after measuring |
| Fullscreen window extent | its pixel size; assumed full width and bottom-aligned on the panel |
| Head pose | MediaPipe facial transformation matrix, depth rescaled from MediaPipe's assumed 63° vertical FOV to the real one |
| Eyeball centre | eye-corner midpoint, pushed `--eye-depth` cm back along the head axis |
| Iris 3D | camera ray through the iris pixel ∩ eyeball sphere (`--eye-radius`) |

## Known limits

- The raw model uses the optical axis (eyeball centre → iris); kappa (~5°) and the landmark gains need the
  per-user calibration.
- The vertical gain (~3×) also amplifies landmark noise, so expect more vertical jitter than horizontal.
- Eyeball radius and depth are population averages.
- Landmark depth (z) from MediaPipe is relative and noisy, which affects the eyeball centre.
