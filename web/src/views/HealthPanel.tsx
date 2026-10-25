import { useEngine } from "../engine";
import type { HealthState } from "../protocol";

/** How strongly a signal suggests tiredness (0 fine .. 1 strong), as the dashboard's status colours. */
const tone = (v: number | null | undefined) => (v == null ? undefined : v < 0.35 ? "ok" : v < 0.65 ? "warn" : "bad");
const fmt = (v: number | null | undefined, unit = "", nd = 0) => (v == null ? "–" : `${v.toFixed(nd)}${unit}`);
const vsStart = (ratio: number | null | undefined) =>
  ratio == null ? "–" : `${ratio >= 1 ? "+" : ""}${Math.round((ratio - 1) * 100)}% vs start`;
const NAMES: Record<string, string> = {
  blink_rate: "low blink rate", perclos: "heavy eyelids", blink_duration: "slow blinks", droop: "drooping lids",
  screen_time: "long screen time", gaze_drift: "unsteady gaze", saccade_speed: "slow eye movements",
  eye_drag: "eye movements dragging", saccade_rate: "little looking around",
};

/** Eye health (health/): fatigue and what drives it, blinking, alertness from eye movement, breaks. The fatigue
 *  graph is in the plot row. Heuristic nudges, not a medical measurement. */
export function HealthPanel() {
  const { frame } = useEngine();
  const h: HealthState | undefined = frame?.health;
  if (!h || h.minutes_on_screen === undefined) return <div className="pane muted">no health data yet</div>;
  const s = h.signals ?? {};
  const top = Object.entries(s).filter(([, v]) => v != null && v >= 0.35).sort((a, b) => (b[1] as number) - (a[1] as number))[0];
  const every = h.break_every_min ?? 20;
  const minutes = h.minutes_on_screen ?? 0;
  const eyes = h.eyes;
  const levelTone = h.level === "high" ? "bad" : h.level === "moderate" ? "warn" : h.level ? "ok" : undefined;
  return (
    <div className="pane scroll">
      <dl>
        <dt>fatigue</dt><dd className={levelTone}>{h.fatigue == null ? "measuring…" : `${h.fatigue} · ${h.level}`}</dd>
        <dt>mostly</dt><dd>{h.fatigue == null ? "needs ~30 s of your face" : top ? NAMES[top[0]] : "rested"}</dd>
      </dl>
      <h4>Blinking</h4>
      <dl>
        <dt>rate</dt><dd className={tone(s.blink_rate)}>{fmt(h.blink_rate, " / min")}</dd>
        <dt>length</dt><dd className={tone(s.blink_duration)}>{fmt(h.blink_duration_ms, " ms")}</dd>
        <dt>eyes closed</dt><dd className={tone(s.perclos)}>{h.perclos == null ? "–" : `${(h.perclos * 100).toFixed(1)}%`}</dd>
        <dt>lids</dt><dd className={tone(s.droop)}>{vsStart(h.openness_vs_start)}</dd>
      </dl>
      <h4>Eye movement</h4>
      <dl>
        <dt>speed</dt><dd className={tone(s.saccade_speed)}>{vsStart(eyes?.saccade_speed)}</dd>
        <dt>drag</dt><dd className={tone(s.eye_drag)}>{vsStart(eyes?.saccade_drag)}</dd>
        <dt>drift</dt><dd className={tone(s.gaze_drift)}>{eyes?.drift_deg == null ? "–" : `${eyes.drift_deg.toFixed(1)}°`}{eyes?.gaze_drift != null ? ` · ${vsStart(eyes.gaze_drift)}` : ""}</dd>
        <dt>looking around</dt><dd className={tone(s.saccade_rate)}>{fmt(eyes?.saccades_per_min, " / min")}</dd>
      </dl>
      <h4>Breaks</h4>
      <dl>
        <dt>on screen</dt><dd className={tone(s.screen_time)}>{fmt(minutes, " min")}</dd>
        <dt>next</dt><dd className={minutes >= every ? "warn" : undefined}>{minutes >= every ? "now: look away 20 s" : `in ${Math.ceil(every - minutes)} min`}</dd>
        <dt>taken</dt><dd>{h.breaks ?? 0}</dd>
      </dl>
      <div className="caption">Eye movement compares with your first 5 minutes. Nudges, not a medical measurement.</div>
    </div>
  );
}
