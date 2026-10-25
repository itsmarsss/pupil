# gaze

Webcam gaze tracking for macOS that works within a minute of launch and gets more
accurate the longer you use it.

It combines two estimators of where you're looking:

- **Appearance:** a neural gaze model on face and eye crops, calibrated per person.
- **Geometry:** iris landmarks and a 3D eyeball model, traced to the screen plane using
  the real display size, camera focal length and lid angle.

Their errors are only weakly correlated, so the calibration uses both: the geometric
measurements are features of the learned fit, and the final point blends in the eyeball
model's own estimate. On 138 replayed check dots this cut the mean error from 2.52° to
2.20°. Right after calibrating, typical accuracy is ~0.8–2° with the head still and
~1–3° moving (1° ≈ 0.8 cm at 45 cm).

## Requirements

- macOS on Apple Silicon, built-in camera
- [uv](https://docs.astral.sh/uv/), Python 3.11; Node 22 + [pnpm](https://pnpm.io) for the dashboard
- Gaze model weights at `models/gaze_base.mnn` (or set `GAZE_MODELS_DIR`). Not included;
  see Licensing. The MediaPipe face landmarker downloads itself on first run.
- Camera permission for your terminal. Close other apps using the camera: macOS shares
  it at the resolution the first app picked, and the app warns when it gets less than 1080p.

## Usage

```sh
uv sync
uv run gaze doctor          # camera access, models, screen, field of view, lid angle
uv run gaze fov --paper a4  # once per camera: measure its field of view (~1 min)
uv run gaze                 # calibrate (first run), then show the gaze ring
```

**First run:** a short intro checks that your face is visible, then 16 dots (~21 s), then
~6 s looking at one dot while slowly turning your head, then a 3-dot quick check. After
that a ring follows your gaze over every app. It holds still while you look at something,
and turns grey when your head is outside its calibrated range or the lid has moved.

**Later runs** reuse the saved calibration after a 3-dot check. If it has drifted, you
can re-centre on one dot (2 s), which brought stale calibrations from 3–9° back to ~2° in
replayed sessions, or recalibrate.

**While you work,** clicks are used as extra calibration samples, but only when you were
already looking near what you clicked.

The ◉ menu-bar item has Recalibrate, Re-centre (1 dot), Check accuracy (9 dots), Measure
(9 dots head still + 9 moving), and toggles for the ring, smoothing, click learning,
snapping to buttons and recording. Esc leaves any fullscreen screen; Ctrl+C (or ◉ → Quit)
quits and saves. Only one copy runs at a time.

### Dashboard

```sh
cd web && pnpm install && pnpm build && cd ..
uv run gaze --dashboard     # opens http://127.0.0.1:8766/
```

A local web view of everything the engine computes: a 3D scene (camera, lid at its real
angle, screen, face point cloud, head axes, eyeballs, iris, eye rays, face-pointing ray,
each model's hit), the camera image with landmarks and model crops, the screen with the
target and every estimate, live plots, the calibration summary, and controls. It only
listens on 127.0.0.1. For dashboard development, run `pnpm dev` in `web/` next to a
running `gaze --dashboard`.

`uv run gaze replay SESSION --dashboard` plays a saved session back in the same views, with
the app's own calibration fitted on that session, a timeline marking each dot, and
play / pause / step / speed controls.

### Options

| Option | |
|---|---|
| `--targets 5\|9\|13\|16\|25` | calibration dots (default 16; more dots = noticeably better) |
| `--recalibrate` | ignore the saved calibration |
| `--profile NAME` | separate calibration per person or setup |
| `--smoothing off\|light\|medium\|strong` | ring smoothing (default medium) |
| `--normalise roll\|frontal\|none` | straighten the face before the model (default roll: level the eyes) |
| `--appearance network\|patches` | eye features from the pretrained network (default) or per-person eye patches: no pretrained weights, ~45 % more error |
| `--mode hd\|vga` | 1440x1080 (default) or 640x480 input, same field of view |
| `--camera NAME_OR_INDEX` | which camera (default: the built-in FaceTime camera); see `gaze cameras` |
| `--dashboard [--dashboard-port N]` | serve and open the web dashboard |
| `--no-clicks` | don't learn from clicks |
| `--snap` | snap the ring to nearby buttons and links (asks for Accessibility access) |
| `--measure` | run the head-still + head-moving measurement after launch |
| `--record FILE` | write every gaze sample (and clicks) to a CSV |
| `--results FILE` | append accuracy checks to a CSV |
| `--save-session [DIR]` | save calibration and check frames (face crops only) for replay |
| `--voice [--typed]` | voice control (see below) |
| `--publish` | also stream gaze over local UDP for other programs |

### Measuring changes

```sh
uv run gaze --save-session --measure
uv run gaze replay ~/Library/Application\ Support/Gaze/recordings/session-…
uv run gaze replay SESSION --modes hd+roll,hd,hd+frontal,vga --all
```

`replay` re-runs a saved session through each pipeline variant on the same frames, including
`live` (the app's own calibration), and reports head-still and head-moving checks
separately. Profiles, recordings, the device file and the accuracy log live in
`~/Library/Application Support/Gaze/`.

## Voice control

Look at something and say what to do. Voice control runs inside the same app as the tracker:

```sh
uv sync --extra control
cp .env.example .env                     # add TYPESAFE_API_KEY (fast tier) and/or ANTHROPIC_API_KEY (Claude agent)
uv run gaze --voice --dashboard          # or: uv run gaze control --dashboard
```

`--typed` types commands in the terminal instead of speaking; `gaze control --gaze mouse` uses the
mouse cursor as gaze (no camera). Needs Microphone, Accessibility and Screen Recording for your terminal
as well as Camera.

| Tier | | Examples |
|---|---|---|
| 0 · `router.py` | exact phrases, matched by rules | "click", "double click this", "scroll down a lot", "go back", "type hello", "stop", "calibrate", "re-centre", "check accuracy" |
| 1 · `gate.py` | Jev answers small yes/no questions about the command and the elements near your gaze; a truth table picks the action | "open Safari", "search repositories", "close this tab" (asks to confirm risky ones) |
| 2 · `agent.py` | Claude computer-use loop, told where you were looking while you spoke | "open this in a new tab and summarize it" |

"This" means where you looked *while speaking*, snapped to the nearest Accessibility element
(`snapping.py`). If the tracker wasn't confident then (head turned past what calibration covered),
"click" and "type" ask you to confirm instead of acting. A pill at the bottom of the screen shows the mic,
what was heard and what happened; the dashboard's "Voice & agent" panel logs it and outlines the target.
Say "stop" to cancel, "confirm" to approve. All settings are in `.env.example`. This drives your real
Mac: use a test browser profile.

## How it works

```
camera 1080p → crop 4:3 → MediaPipe landmarks + head pose ─┬─ level the eyes → face/eye crops → gaze network ─┐
                                                            └─ 3D eyeball model (geometry/) ──────────────────┤
                       per-person calibration (network features | geometric features | head pose) ◄──────────┘
                       → blend with the eyeball model → drift offset → smoothing → ring / dashboard / voice control
```

- **Calibration** fits ridge regressions from those features to screen points and picks
  among them by leave-one-dot-out error, preferring the more regularised fit when close.
- **The head-movement step** is what makes head pose usable: during the dots the head
  barely moves independently of gaze, so a fit can't learn to separate the two.
- **The eyeball model** fits a per-person gain and offset (kappa) for eye yaw and pitch,
  then a small quadratic screen correction and head compensation.
- **Smoothing** holds the ring at the running mean of the current fixation and moves only
  when several frames land clearly elsewhere, then lightly filters the result.
- **The lid angle** comes from the MacBook's hinge sensor. It isn't a gaze correction (the
  camera and screen tilt together); it flags a stale calibration.
- `src/gaze/telemetry.py` defines the per-frame state every display reads (versioned).

## Licensing

The default gaze network's weights are non-commercial (CC BY-NC-SA), as are the public gaze
datasets that would be needed to retrain one. `--appearance patches` runs without any
pretrained gaze weights, at ~45 % more error. MediaPipe is Apache-2.0.

## Tests

```sh
uv run pytest
cd web && pnpm typecheck
```
