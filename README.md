# Pupil

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
- Pretrained gaze-network weights at `models/gaze_base.mnn` (or set `PUPIL_MODELS_DIR`). Not included;
  see Licensing. The MediaPipe face landmarker downloads itself on first run.
- Camera permission for your terminal. Close other apps using the camera: macOS shares
  it at the resolution the first app picked, and the app warns when it gets less than 1080p.

## Usage

```sh
uv sync
uv run pupil doctor          # camera access, models, screen, field of view, lid angle
uv run pupil fov --paper a4  # once per camera: measure its field of view (~1 min)
uv run pupil                 # calibrate (first run), then show the gaze ring
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
uv run pupil --dashboard     # opens http://127.0.0.1:8766/
```

A local web view of everything the engine computes: a 3D scene (camera, lid at its real
angle, screen, face point cloud, head axes, eyeballs, iris, eye rays, face-pointing ray,
each model's hit), the camera image with landmarks and model crops, the screen with the
target and every estimate, live plots, the calibration summary, and controls. It only
listens on 127.0.0.1. For dashboard development, run `pnpm dev` in `web/` next to a
running `pupil --dashboard`.

`uv run pupil replay SESSION --dashboard` plays a saved session back in the same views, with
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
| `--camera NAME_OR_INDEX` | which camera (default: the built-in FaceTime camera); see `pupil cameras` |
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
uv run pupil --save-session --measure
uv run pupil replay ~/Library/Application\ Support/Pupil/recordings/session-…
uv run pupil replay SESSION --modes hd+roll,hd,hd+frontal,vga --all
```

`replay` re-runs a saved session through each pipeline variant on the same frames, including
`live` (the app's own calibration), and reports head-still and head-moving checks
separately. Profiles, recordings, the device file and the accuracy log live in
`~/Library/Application Support/Pupil/`.

## The notch

What the app is doing lives in the notch (or hangs from the top centre on screens without one), in the spirit of
[Notchy](https://notchy.dev), and only about the happy path: small ears show whether it sees you and is
listening, and while a command runs, what it's doing ("Listening", "Agent · 3"); it grows a line for things to
read (the agent's reply, "say confirm") and opens when you hover it or look at it: what it's doing, your last few
commands and what they did (or what to try, at first), gaze accuracy, and a mic toggle. Never in screenshots, so the agent can't see it; ◉ → View →
Notch or `PUPIL_NOTCH=0` turns it off. The web dashboard is for debugging.

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
dashboard has a Health panel; a line a minute goes to `~/Library/Application Support/Pupil/health/`.
These are heuristics for nudges, not a medical measurement.

## Voice control

Look at something and say what to do. Voice control runs inside the same app as the tracker:

```sh
uv sync --extra control
cp .env.example .env                     # add TYPESAFE_API_KEY (fast tier) and/or ANTHROPIC_API_KEY (Claude agent)
uv run pupil --voice --dashboard          # or: uv run pupil control --dashboard
```

`pupil control --gaze face` follows where your head points instead of your eyes, with ~0.6° of emulated
inaccuracy: steady and calibration-free for demos (one centre dot lines it up; `--face-noise DEG`, `--face-gain X`,
default 2.5, since heads turn less than eyes). `--typed` types commands in the terminal instead of speaking; `pupil control --gaze mouse --dashboard`
uses the **cursor for gaze** but keeps the **webcam for the lip gate**, so voice, barge-in, and TTS behave like
eye mode (barge-in when your lips move after speech starts). Pass `--no-camera` for cursor-only dev mode (mic ducks
during TTS). If **your** speech gets cut mid-sentence, lower `PUPIL_CARTESIA_END_THRESHOLD` (e.g. `0.2`). Needs
Microphone, **Camera**, Accessibility and Screen Recording for your terminal
as well as Camera.

| Tier | | Examples |
|---|---|---|
| 0 · `router.py` | only what must work instantly and offline, matched by rules | "stop", "confirm" / "do it", "start/stop dictation", "calibrate", "re-centre", "check accuracy" |
| 1 · `gate.py` | Jev answers small yes/no and pick-one questions about the command and the elements near your gaze; a truth table picks the action. No keyword lists: clicks, scrolling, shortcuts (via the app's own menu command, which Jev picks from its real menus), websites and searches, typing, quitting apps, "again" / "no, the left one" and answers to "which one?" are all its reading. A fast model (`extract.py`) pulls out the text Jev can't carry: what to type, the URL or search, the steps of a multi-step request. It asks a question only when something the screen can't supply is missing, holds commands it isn't sure were meant for it without interrupting ("do it" runs them), and runs a few actions in a row itself, handing the rest to the agent if a step needs it | "click this", "scroll down a little", "go to github.com", "type hello world", "quit Games", "open Safari", "close this tab" (asks to confirm risky ones), "select all and copy", "send it to…" → "Who to?" |
| 2 · `agent.py` | Claude computer-use loop, told where you were looking while you spoke, with app-agnostic tools beside the screenshot/mouse/keyboard ones (`uitree.py`): any app's menu bar (`menu`), its controls by id (`ui_elements`, `press_element`, `set_value`), AppleScript/JXA, `drag` with any button and modifiers, `open_app`; plus app adapters where an app has its own scripting (`apps/`: Blender via `blender_python`) | "open this in a new tab and summarize it", "export this as PDF", "build me a car in Blender" |

Beyond the screen: the agent fetches what a request needs the way a person would, instead of answering only from
the screen: your location from macOS Location Services (the `location` tool, which always waits for your "confirm";
macOS also asks once), the Maps app if that fails, Contacts / Calendar / Reminders through their apps, a web search
or the research tool. "What's my postal code?" works. A task that's stopped, fails or runs out of steps is kept for
15 minutes with its whole history: "keep going", "continue", "where were we" picks it up where it was (Jev reads
those against the interrupted task; it checks the screen first and doesn't redo what's done).

Attention check: it only listens while you're paying attention: your face visible, your head turned toward the
screen (within 35°) and, once calibrated, your gaze on or near it. Turn to talk to someone or look down at your
phone for more than 1.5 s and nothing is sent or transcribed, a turn in progress ends, and the notch says "Not
listening". Glances don't count.

Listening: the default mic (AirPods included), raw; Apple's voice processing (noise suppression, echo
cancellation) is optional, ◉ → Voice isolation. The lip gate (`lips.py`) only lets speech through while
you face the screen and your lips move, keeping 1.5 s of audio so a late open loses nothing, and drops
transcripts from a visible face whose lips never moved (someone else, a video); the dashboard's Lips
panel shows it live. Speech goes to Cartesia Ink-2 when `CARTESIA_API_KEY` is set (streamed, so the turn
ends ~1 s after you stop; only while the gate is open, since it bills per second), otherwise to on-
device Whisper `base.en` behind a Silero speech detector. `PUPIL_LIP_GATE=shadow` logs what the gate
would block; `PUPIL_VOICE_PROCESSING=1` turns voice isolation on at start. Misheard words are left for
Jev to read in context, with no fuzzy matching to a phrase list.

Web research (by Anthony Lu): with `APIFY_TOKEN` set, the agent can read public pages and research current
questions ("compare this with three alternatives", "is this the cheapest?") through Apify's hosted MCP server,
citing its sources. When Jev judges a request needs the web, it runs in the background (`research.py`): a fast
model names what "this" is from the page you're on, one capped Apify run gathers sources, and the answer comes back
as a sentence, while you carry on with something else. The notch shows it ("Researching" in its ears, then the
answer, and in the open notch each job's progress, answer and a click-through to its top source); it's spoken
only if nothing else is going on, two can run at once, and "stop" with nothing else running drops them. The
agent's `research` tool reads it back (status, results with sources and page text, start, wait) and `open_urls`
shows the sources, so "is it done?", "what did it find?", "open the sources" or "put that in a note" work without
starting new research. Your conversation and window title only shape the question: they never go to Apify. Inside
agent tasks the agent can also look up and run a specialised Apify Actor, always after you say "confirm". Scraped text is untrusted data; every run is capped (45 s, 15 items,
$0.25 by default) and so is each task's total (`PUPIL_APIFY_TASK_BUDGET_USD`, $1). Results show in the
dashboard's "Decision research" tab; `actors/accessible-decision-researcher` is a custom Actor for it.

Which model: Jev routes each agent task to a tier (fast: Sonnet at low effort, for quick questions and
real-time games; standard; deep: Opus, for writing, building and multi-app work) and a step budget (quick,
normal, or ongoing until you say stop, so it can play Tetris). The agent can also change this app's own
settings when asked: camera, microphone, voice isolation, spoken replies, gaze display, smoothing.

Talking back: the agent's replies, its clarifying questions and "say confirm" prompts are read aloud with
Cartesia Sonic, telegraphic ("Done. 3 results."), and run as faint captions under a small status pill that
follows your gaze (colour-coded: listening, thinking, agent working, waiting on you); `PUPIL_SPEAK=0` turns
speech off. Start talking and it stops: barge-in is
detected from your lips moving, not sound, so the Mac's own speaker doesn't cut itself off. Speaking while
Claude is working reaches it at its next step ("actually, make it blue"); saying something other than
"confirm" when it's waiting to confirm skips that step and follows the new instruction; "stop" cancels. Each
new task starts with the last few minutes of the conversation (what you said, even what Jev let pass, what was
done and what it answered), and Jev reads your last few sentences and its last question, so "that one", "the
other one" and answers to its questions make sense.

Typing: "type …" and dictation mode ("start dictation" … "stop dictation") type what you meant, not the
raw transcript: a fast model (Claude Haiku) uses the app and the text before the cursor to turn "slash btw"
into `/btw` in a chat box, "dash dash help" into `--help`, spoken punctuation into punctuation, and to drop
fillers and self-corrections ("at 5, no wait, 6" → "at 6"). Password fields are never read.

"This" means where you looked *while speaking*, snapped to the nearest Accessibility element
(`snapping.py`). If the tracker wasn't confident then (head turned past what calibration covered),
"click" and "type" ask you to confirm instead of acting. The notch shows what was heard and what happened
(click a recent command in the open notch to see all of it: the whole result, how it was handled, when)
(`PUPIL_HUD_BAR=1` brings back the old status bar at the bottom); the dashboard's "Voice & agent" panel logs it, outlines the target and
lists which model each part calls (speech in, Jev, extraction, dictation, agent tier, speech out), lit
while a call is open.
Say "stop" to cancel, "confirm" to approve. All settings are in `.env.example`. This drives your real
Mac: use a test browser profile.

Optional live decision research uses Apify. Add `APIFY_TOKEN` to `.env`, look at a public product,
listing, venue, or article, and say “compare this with three accessible alternatives under $50.”
The Decision research dashboard panel progresses from starting to sourced cards; “why?”, “open source
two,” and follow-up comparisons work through the normal gaze/voice agent. Curated research is capped at
15 items, 45 seconds, and $0.25 by default. Store Actor runs always pause for confirmation. Cookies,
credentials, private/local pages, authenticated sources, and bulk personal-data collection are blocked.
Source text is untrusted and may be incomplete; the assistant presents evidence rather than certifying
truth. Use **Delete history** in the research panel to clear retained cards.

### Custom Apify Actor setup

The stock research Actor works with only `APIFY_TOKEN`. To deploy this repository's typed decision
researcher and use it instead:

```sh
brew install apify-cli                 # or: npm install -g apify-cli
apify login                            # browser login is recommended
cd actors/accessible-decision-researcher
pnpm install
pnpm check && pnpm build
apify push --open --wait-for-finish=300
```

Copy the deployed Actor's full name from its Apify Console API/integration section, then add it to the
repository root `.env`:

```env
APIFY_TOKEN=your-token
PUPIL_APIFY_RESEARCH_ACTOR=your-username/accessible-decision-researcher
```

Restart Pupil after changing `.env`; configuration is loaded when the process starts.

The heat map gives extra weight to your final stable fixation. For a recent clicked target, say
"again" to repeat the action, "do the same here" while looking somewhere else, or
"close the one I just clicked". References expire after `RELATIVE_WINDOW_S` (20 s) and never cross apps or windows.

The terminal shows one short block per command (what you said, what happened, Jev's time and the total) and a
line when an agent task or research finishes; HTTP and library chatter is silenced (`PUPIL_VERBOSE=1` shows every
status line). Every command is logged to `.logs/episodes.jsonl` with what was seen, decided and done, and labelled by
your reaction ("stop" right after, or what Jev reads as "no, not that" = corrected; saying it again =
repeated). `pupil episodes` lists
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
- `src/pupil/telemetry.py` defines the per-frame state every display reads (versioned).

## Licensing

The default gaze network's weights are non-commercial (CC BY-NC-SA), as are the public gaze
datasets that would be needed to retrain one. `--appearance patches` runs without any
pretrained gaze weights, at ~45 % more error. MediaPipe is Apache-2.0.

## Tests

```sh
uv run pytest
cd web && pnpm typecheck
```
