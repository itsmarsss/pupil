# gaze

Webcam gaze tracking for macOS that works within a minute of launch and gets more
accurate the longer you use it.

The repo holds two approaches, measured the same way so they can be compared:

| | What it is | Accuracy so far (median error at held-out dots) |
|---|---|---|
| `src/gaze` | **The app.** Appearance-based: a neural gaze model on face and eye crops, calibrated per person. Menu-bar app with a gaze ring over every app. | ~1.4–1.7° head still, ~2–3° head moving (16–25 dots) |
| [`geo/`](geo/README.md) | **Geometric.** Iris landmarks + 3D eyeball model → gaze ray → screen plane, using real device geometry. Few per-person parameters. | ~1.4–1.7° (test runs) |

1° is about 0.8 cm on screen at a typical 45 cm viewing distance.

## Requirements

- macOS on Apple Silicon, built-in camera
- [uv](https://docs.astral.sh/uv/) (Python 3.11 for the app; `geo/` uses 3.12 and its own environment)
- Gaze model weights at `models/gaze_base.mnn` (or set `GAZE_MODELS_DIR`). Not included
  in the repo. The MediaPipe face landmarker downloads itself on first run.
- Camera permission for your terminal. Close other apps using the camera: macOS shares
  it at whatever resolution the first app picked, and the app warns if it gets less
  than 1080p.

## Usage

```sh
uv sync
uv run gaze doctor      # camera access, models, screen size, lid angle
uv run gaze             # calibrate (first run), then show the gaze ring
```

**First run:** a short intro checks that your face is visible and shows your distance,
then 16 dots (~21 s), then ~6 s looking at one dot while slowly turning your head, then
a 3-dot accuracy check. After that a ring follows your gaze over every app. It holds
still while you look at something, and turns grey when your head is outside the range
it was calibrated for or the lid has moved since calibrating.

**Later runs** reuse the saved calibration after a 3-dot check and only ask you to
recalibrate if it's off by more than 4°.

**While you work,** clicks are used as extra calibration samples, but only when you were
already looking near what you clicked.

The ◉ menu-bar item has Recalibrate, Check accuracy (9 dots), Measure (9 dots head still
+ 9 moving), and toggles for the ring, smoothing level, click learning, snapping to
buttons and recording. Esc leaves any fullscreen screen; Ctrl+C in the terminal (or
◉ → Quit) quits and saves.

### Options

| Option | |
|---|---|
| `--targets 5\|9\|13\|16\|25` | calibration dots (default 16; more dots = noticeably better) |
| `--recalibrate` | ignore the saved calibration |
| `--profile NAME` | separate calibration per person or setup |
| `--smoothing off\|light\|medium\|strong` | ring smoothing (default medium) |
| `--normalise roll\|frontal\|none` | straighten the face before the model (default roll: level the eyes) |
| `--mode hd\|vga` | 1440x1080 (default) or 640x480 input, same field of view |
| `--camera NAME_OR_INDEX` | which camera (default: the built-in FaceTime camera); see `gaze cameras` |
| `--no-clicks` | don't learn from clicks |
| `--snap` | snap the ring to nearby buttons and links (asks for Accessibility access) |
| `--measure` | run the head-still + head-moving measurement after launch |
| `--record FILE` | write every gaze sample (and clicks) to a CSV |
| `--results FILE` | append accuracy checks to a CSV |
| `--save-session [DIR]` | save calibration and check frames (face crops only) for replay |

### Measuring changes

```sh
uv run gaze --recalibrate --save-session --measure
uv run gaze replay ~/Library/Application\ Support/Gaze/recordings/session-…
uv run gaze replay SESSION --modes hd+roll,hd,hd+frontal,vga --all
```

`replay` re-runs a saved session through each pipeline variant and calibrator on the
same frames, and reports head-still and head-moving checks separately. Profiles,
recordings and the accuracy log live in `~/Library/Application Support/Gaze/`.

## How it works

```
camera 1080p → crop 4:3 → MediaPipe face landmarks → level the eyes → face + eye crops
  → gaze model (258 features) + head pose → per-person calibration → smoothing → ring
```

- **Calibration** fits ridge regressions from the model's features and head pose to
  screen points, and picks among them by leave-one-dot-out error, preferring the more
  regularised fit when scores are close.
- **The head-movement step** is what makes head pose usable: during the dots the head
  barely moves independently of gaze, so a fit can't learn to separate the two.
- **Smoothing** holds the ring at the running mean of the current fixation and moves
  only when several frames land clearly elsewhere, then lightly filters the result.
- **The lid angle** is read from the MacBook's hinge sensor. It isn't used to correct
  gaze (the camera and screen tilt together), only to flag a stale calibration.

## Tests

```sh
uv run pytest                     # the app
cd geo && uv run python test_geometry.py && uv run python test_calib.py && uv run python test_iris.py
```
