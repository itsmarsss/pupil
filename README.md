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

The ◉ menu-bar item is grouped into Calibration (Recalibrate, Re-centre, Check accuracy,
Measure), View (gaze ring, smoothing, snap to buttons, open the dashboard), Input (camera,
voice control on/off, microphone) and Learning & data (click learning, recording). Cameras
and mics switch live; each camera keeps its own calibration, so a new one asks for a
calibration the first time. Esc leaves any fullscreen screen; Ctrl+C (or ◉ → Quit) quits
and saves. Only one copy runs at a time.

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

## The notch

What the app is doing lives in the notch (or hangs from the top centre on screens without one), in the spirit of
[Notchy](https://notchy.dev), and only about the happy path: small ears show whether it sees you and is
listening, and while a command runs, what it's doing ("Listening", "Agent · 3"); it grows a line for things to
read (the agent's reply, "say confirm") and opens when you hover it or look at it: what it's doing, your last few
commands and what they did (or what to try, at first), gaze accuracy, and a mic toggle. Never in screenshots, so the agent can't see it; ◉ → View →
Notch or `GAZE_NOTCH=0` turns it off. The web dashboard is for debugging.

## Eye health

While it tracks your eyes, gaze also watches out for them (`health/`): blink rate (relaxed is ~15–20 a
minute; staring at screens often drops it to 5–7, which dries the eyes), blink length, PERCLOS (share of the
last minute with the eyes mostly closed, the standard drowsiness measure), lid droop against the start of
the session, and screen time since your last break (20 s with your face out of view or looking away counts:
the 20-20-20 rule), and alertness from how your eyes move, against your first minutes: how fast they jump
between things, whether the jumps drag (take longer per degree), how steadily they hold, and how often you
look around. They combine into a fatigue score (low / moderate / high), sensitive on
purpose: one strong sign is enough. Blinks are measured against
your own open-eye baseline, so there's no fixed threshold to tune. Gentle, rate-limited reminders arrive
as notifications (and at your gaze with voice control on): blink when the rate stays low, look away after
20 minutes, take a break when fatigue is high. ◉ → Health shows the numbers and turns reminders off; the
dashboard has a Health panel; a line a minute goes to `~/Library/Application Support/Gaze/health/`.
These are heuristics for nudges, not a medical measurement.

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
| 0 · `router.py` | only what must work instantly and offline, matched by rules | "stop", "confirm" / "do it", "start/stop dictation", "calibrate", "re-centre", "check accuracy" |
| 1 · `gate.py` | Jev answers small yes/no and pick-one questions about the command and the elements near your gaze; a truth table picks the action. No keyword lists: clicks, scrolling, shortcuts (via the app's own menu command, which Jev picks from its real menus), websites and searches, typing, quitting apps, "again" / "no, the left one" and answers to "which one?" are all its reading. A fast model (`extract.py`) pulls out the text Jev can't carry: what to type, the URL or search, the steps of a multi-step request. It asks a question only when something the screen can't supply is missing, offers commands it isn't sure were meant for it ("do it" runs them), and runs a few actions in a row itself, handing the rest to the agent if a step needs it | "click this", "scroll down a little", "go to github.com", "type hello world", "quit Games", "open Safari", "close this tab" (asks to confirm risky ones), "select all and copy", "send it to…" → "Who to?" |
| 2 · `agent.py` | Claude computer-use loop, told where you were looking while you spoke, with app-agnostic tools beside the screenshot/mouse/keyboard ones (`uitree.py`): any app's menu bar (`menu`), its controls by id (`ui_elements`, `press_element`, `set_value`), AppleScript/JXA, `drag` with any button and modifiers, `open_app`; plus app adapters where an app has its own scripting (`apps/`: Blender via `blender_python`) | "open this in a new tab and summarize it", "export this as PDF", "build me a car in Blender" |

Listening: the default mic (AirPods included), raw; Apple's voice processing (noise suppression, echo
cancellation) is optional, ◉ → Voice isolation. The lip gate (`lips.py`) only lets speech through while
you face the screen and your lips move, keeping 1.5 s of audio so a late open loses nothing, and drops
transcripts from a visible face whose lips never moved (someone else, a video); the dashboard's Lips
panel shows it live. Speech goes to Cartesia Ink-2 when `CARTESIA_API_KEY` is set (streamed, so the turn
ends ~1 s after you stop; only while the gate is open, since it bills per second), otherwise to on-
device Whisper `base.en` behind a Silero speech detector. `GAZE_LIP_GATE=shadow` logs what the gate
would block; `GAZE_VOICE_PROCESSING=1` turns voice isolation on at start. Misheard words are left for
Jev to read in context, with no fuzzy matching to a phrase list.

Which model: Jev routes each agent task to a tier (fast: Sonnet at low effort, for quick questions and
real-time games; standard; deep: Opus, for writing, building and multi-app work) and a step budget (quick,
normal, or ongoing until you say stop, so it can play Tetris). The agent can also change this app's own
settings when asked: camera, microphone, voice isolation, spoken replies, gaze display, smoothing.

Talking back: the agent's replies, its clarifying questions and "say confirm" prompts are read aloud with
Cartesia Sonic, telegraphic ("Done. 3 results."), and run as faint captions under a small status pill that
follows your gaze (colour-coded: listening, thinking, agent working, waiting on you); `GAZE_SPEAK=0` turns
speech off. Start talking and it stops: barge-in is
detected from your lips moving, not sound, so the Mac's own speaker doesn't cut itself off. Speaking while
Claude is working reaches it at its next step ("actually, make it blue"); saying something other than
"confirm" when it's waiting to confirm skips that step and follows the new instruction; "stop" cancels.

Typing: "type …" and dictation mode ("start dictation" … "stop dictation") type what you meant, not the
raw transcript: a fast model (Claude Haiku) uses the app and the text before the cursor to turn "slash btw"
into `/btw` in a chat box, "dash dash help" into `--help`, spoken punctuation into punctuation, and to drop
fillers and self-corrections ("at 5, no wait, 6" → "at 6"). Password fields are never read.

"This" means where you looked *while speaking*, snapped to the nearest Accessibility element
(`snapping.py`). If the tracker wasn't confident then (head turned past what calibration covered),
"click" and "type" ask you to confirm instead of acting. A pill at the bottom of the screen shows the mic,
what was heard and what happened; the dashboard's "Voice & agent" panel logs it, outlines the target and
lists which model each part calls (speech in, Jev, extraction, dictation, agent tier, speech out), lit
while a call is open.
Say "stop" to cancel, "confirm" to approve. All settings are in `.env.example`. This drives your real
Mac: use a test browser profile.

The heat map gives extra weight to your final stable fixation. For a recent clicked target, say
"again" to repeat the action, "do the same here" while looking somewhere else, or
"close the one I just clicked". References expire after `RELATIVE_WINDOW_S` (20 s) and never cross apps or windows.

Every command is logged to `.logs/episodes.jsonl` with what was seen, decided and done, and labelled by
your reaction ("stop" right after, or what Jev reads as "no, not that" = corrected; saying it again =
repeated). `gaze episodes` lists
them with latency per route, `--replay` re-runs the routing rules on the logged answers offline, and
`--eval` scores the gate live on the labelled cases in `tests/agent_cases*.jsonl`.

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
