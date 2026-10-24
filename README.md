# gaze

Webcam gaze tracking for macOS that works within a minute of launch and gets more
accurate the longer you use it.

## Requirements

- macOS on Apple Silicon, built-in camera
- Python 3.11, [uv](https://docs.astral.sh/uv/)
- Gaze model weights at `models/gaze_base.mnn` (or set `GAZE_MODELS_DIR`). The
  MediaPipe face landmarker downloads itself on first run.

## Usage

```sh
uv sync
uv run gaze doctor      # check camera access, models, screen
uv run gaze             # calibrate (first run), then show the gaze ring
```

First run: a short intro checks that your face is visible, then 9 dots (~12 s), then a
3-dot accuracy check. After that a ring follows your gaze over every app. Clicks are
used as extra calibration samples while you work.

Later runs reuse the saved calibration after a 3-dot check, and only ask you to
recalibrate if it has drifted.

The ◉ menu-bar item has Recalibrate, Check accuracy (9 dots), and toggles for the
ring, click learning, snapping to buttons and recording. Esc leaves any fullscreen
screen; Ctrl+C in the terminal quits and saves.

Useful options:

| Option | |
|---|---|
| `--mode vga` | 640x480 input instead of 1440x1080 |
| `--targets 5\|9\|13` | calibration dots |
| `--recalibrate` | ignore the saved calibration |
| `--no-clicks` | don't learn from clicks |
| `--profile NAME` | separate calibration per person/setup |
| `--results FILE` | append accuracy checks to a CSV |
| `--record FILE` | write every gaze sample (and clicks) to a CSV |
| `--snap` | snap the ring to nearby buttons and links (asks for Accessibility access) |
| `--save-session [DIR]` | save calibration/validation face crops for replay |

`uv run gaze replay SESSION_DIR` re-runs a saved session through each capture mode and
calibrator and prints the accuracy of each, so changes can be compared on identical data.

Calibration profiles and accuracy logs live in `~/Library/Application Support/Gaze/`.

## Tests

```sh
uv run pytest
```
