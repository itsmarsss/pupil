import { engine, useEngine } from "../engine";
import type { HealthState } from "../protocol";
import { Plot, type PlotSpec } from "./Plots";

// Colours by how strongly a value suggests tiredness (0 fine .. 1 strong)
const GOOD = "#3ddc84";
const MID = "#ffc43d";
const BAD = "#ff6b6b";
const MUTED = "#8b93a1";
const severity = (v: number) => (v < 0.35 ? GOOD : v < 0.65 ? MID : BAD);
const LEVEL_COLOR = { low: GOOD, moderate: MID, high: BAD } as const;
const pct = (ratio: number | null | undefined, invert = false) =>
  ratio == null ? "–" : `${(invert ? 1 - ratio : ratio - 1) >= 0 ? "+" : ""}${Math.round(((invert ? 1 - ratio : ratio - 1)) * 100)}%`;

const FATIGUE_PLOT: PlotSpec = { id: "fatigue", title: "Fatigue", series: [{ key: "fatigue", label: "fatigue", color: BAD }],
  range: [0, 100] };

type SignalKey = keyof NonNullable<HealthState["signals"]>;
const SIGNALS: { key: SignalKey; label: string; value: (h: HealthState) => string }[] = [
  { key: "blink_rate", label: "Blink rate", value: (h) => (h.blink_rate == null ? "–" : `${h.blink_rate.toFixed(0)} / min`) },
  { key: "perclos", label: "Eyes closed", value: (h) => (h.perclos == null ? "–" : `${(h.perclos * 100).toFixed(1)}%`) },
  { key: "blink_duration", label: "Blink length", value: (h) => (h.blink_duration_ms == null ? "–" : `${h.blink_duration_ms} ms`) },
  { key: "droop", label: "Lids vs start", value: (h) => (h.openness_vs_start == null ? "–" : `${Math.round(h.openness_vs_start * 100)}%`) },
  { key: "gaze_drift", label: "Gaze drift", value: (h) => pct(h.eyes?.gaze_drift) },
  { key: "saccade_speed", label: "Eye speed", value: (h) => pct(h.eyes?.saccade_speed) },
  { key: "saccade_rate", label: "Looking around", value: (h) => (h.eyes?.saccades_per_min == null ? "–" : `${h.eyes.saccades_per_min.toFixed(0)} / min`) },
  { key: "reaction", label: "Reaction", value: (h) => (h.reaction_ms == null ? "check" : `${h.reaction_ms} ms`) },
  { key: "screen_time", label: "Screen time", value: (h) => `${(h.minutes_on_screen ?? 0).toFixed(0)} min` },
];
const NAMES: Record<string, string> = {
  blink_rate: "low blink rate", perclos: "heavy eyelids", blink_duration: "slow blinks", droop: "drooping lids",
  screen_time: "long screen time", gaze_drift: "unsteady gaze", saccade_speed: "slower eye movements",
  saccade_rate: "less looking around", reaction: "slower reactions",
};

/** Eye health: fatigue over time (a graph, like the rest of the dashboard) with what drives it, blinking, breaks,
 *  eye movements and reaction time (health/). Heuristic nudges, not a medical measurement. */
export function HealthPanel() {
  const { frame } = useEngine();
  const h = frame?.health;
  if (!h || h.minutes_on_screen === undefined) return <div className="pane placeholder">no health data yet</div>;
  const signals = h.signals ?? {};
  const top = Object.entries(signals).filter(([, v]) => v != null && v >= 0.35).sort((a, b) => (b[1] as number) - (a[1] as number))[0];
  const every = h.break_every_min ?? 20;
  const minutes = h.minutes_on_screen ?? 0;
  const due = minutes >= every;
  const level = h.level;
  return (
    <div className="pane scroll health">
      <section className="h-top">
        <div className="h-score">
          <span className="h-num" style={{ color: level ? LEVEL_COLOR[level] : MUTED }}>{h.fatigue ?? "–"}</span>
          <span className="h-level" style={{ color: level ? LEVEL_COLOR[level] : MUTED }}>{level ?? "measuring"}</span>
          <span className="h-why">{h.fatigue == null ? "needs ~30 s of your face" : top ? <>mostly <b>{NAMES[top[0]]}</b></> : "rested"}</span>
        </div>
        <div className="h-plot"><Plot spec={FATIGUE_PLOT} source={() => engine.healthSeries} unit="min" /></div>
      </section>
      <section className="h-row">
        <span className={`h-break${due ? " due" : ""}`}>
          {due ? "look away 20 s" : <>break in <b>{Math.ceil(every - minutes)}m</b></>}
          <span className="h-sub"> · {h.breaks ?? 0} taken</span>
        </span>
        <button className="h-check" onClick={() => engine.send("alertness")} title="8 dots, ~20 s: how fast your eyes react">
          Check alertness
        </button>
      </section>
      <section className="h-signals">
        {SIGNALS.map((s) => {
          const v = signals[s.key];
          return (
            <div key={s.key} className="h-signal">
              <span className="h-label">{s.label}</span>
              <span className="h-bar">
                <span style={{ width: `${v == null ? 0 : Math.max(v * 100, 3)}%`, background: v == null ? MUTED : severity(v) }} />
              </span>
              <span className="h-value">{s.value(h)}</span>
            </div>
          );
        })}
      </section>
      <div className="h-note">eye movements compare with your first minutes · nudges, not a medical measurement</div>
    </div>
  );
}
