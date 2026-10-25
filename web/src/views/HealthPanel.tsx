import { engine, useEngine } from "../engine";
import type { HealthState } from "../protocol";

// Colours by how strongly a value suggests tiredness (0 fine .. 1 strong)
const GOOD = "#3ddc84";
const MID = "#ffc43d";
const BAD = "#ff6b6b";
const MUTED = "#8b93a1";
const severity = (v: number) => (v < 0.35 ? GOOD : v < 0.65 ? MID : BAD);
const LEVEL_COLOR = { low: GOOD, moderate: MID, high: BAD } as const;

const SIGNALS: { key: keyof NonNullable<HealthState["signals"]>; label: string; value: (h: HealthState) => string }[] = [
  { key: "blink_rate", label: "Blink rate", value: (h) => (h.blink_rate == null ? "–" : `${h.blink_rate.toFixed(0)} / min`) },
  { key: "perclos", label: "Eyes closed", value: (h) => (h.perclos == null ? "–" : `${(h.perclos * 100).toFixed(1)}%`) },
  { key: "blink_duration", label: "Blink length", value: (h) => (h.blink_duration_ms == null ? "–" : `${h.blink_duration_ms} ms`) },
  { key: "droop", label: "Lids vs start", value: (h) => (h.openness_vs_start == null ? "–" : `${Math.round(h.openness_vs_start * 100)}%`) },
  { key: "screen_time", label: "Screen time", value: (h) => `${(h.minutes_on_screen ?? 0).toFixed(0)} min` },
];
const NAMES: Record<string, string> = {
  blink_rate: "low blink rate", perclos: "heavy eyelids", blink_duration: "slow blinks", droop: "drooping lids",
  screen_time: "long screen time",
};

/** Fatigue as an arc from green through amber to red, with the score in the middle. */
function Gauge({ score, level }: { score: number | null | undefined; level: HealthState["level"] }) {
  const r = 64, cx = 80, cy = 78, len = Math.PI * r;
  const shown = score == null ? 0 : score / 100;
  return (
    <svg className="gauge" viewBox="0 0 160 92" role="img" aria-label={`fatigue ${score ?? "not measured"}`}>
      <defs>
        <linearGradient id="gauge-fill" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0" stopColor={GOOD} /><stop offset="0.5" stopColor={MID} /><stop offset="1" stopColor={BAD} />
        </linearGradient>
      </defs>
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="#232833" strokeWidth={12} strokeLinecap="round" />
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} fill="none" stroke="url(#gauge-fill)" strokeWidth={12}
            strokeLinecap="round" strokeDasharray={`${len * shown} ${len}`} className="gauge-arc" />
      <text x={cx} y={cy - 14} textAnchor="middle" className="gauge-score">{score ?? "–"}</text>
      <text x={cx} y={cy + 6} textAnchor="middle" className="gauge-level" fill={level ? LEVEL_COLOR[level] : MUTED}>
        {level ?? "measuring"}
      </text>
    </svg>
  );
}

/** Blink rate over the last couple of minutes, with the relaxed 15-20 band shaded. */
function Sparkline({ values }: { values: number[] }) {
  const w = 70, h = 18, max = 30;
  const y = (v: number) => h - (Math.min(v, max) / max) * h;
  const pts = values.map((v, i) => `${(i / Math.max(values.length - 1, 1)) * w},${y(v)}`).join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <rect x={0} y={y(20)} width={w} height={y(15) - y(20)} fill={GOOD} opacity={0.12} />
      {values.length > 1 && <polyline points={pts} fill="none" stroke="#4dabf7" strokeWidth={2} vectorEffect="non-scaling-stroke" />}
    </svg>
  );
}

/** Progress toward the next 20-20-20 break. */
function BreakRing({ minutes, every }: { minutes: number; every: number }) {
  const r = 24, c = 2 * Math.PI * r, due = minutes >= every;
  const frac = Math.min(minutes / every, 1);
  return (
    <svg className={`ring${due ? " due" : ""}`} viewBox="0 0 64 64">
      <circle cx={32} cy={32} r={r} fill="none" stroke="#232833" strokeWidth={8} />
      <circle cx={32} cy={32} r={r} fill="none" stroke={due ? MID : "#4dabf7"} strokeWidth={8} strokeLinecap="round"
              strokeDasharray={`${c * frac} ${c}`} transform="rotate(-90 32 32)" />
      <text x={32} y={39} textAnchor="middle" className="ring-text">{Math.floor(minutes)}</text>
    </svg>
  );
}

/** Eye health: fatigue and what drives it, blinking, and breaks (health/tracker.py). Compact: a small gauge beside
 *  the two numbers that matter, then a bar per signal. Heuristic nudges, not a medical measurement. */
export function HealthPanel() {
  const { frame } = useEngine();
  const h = frame?.health;
  if (!h || h.minutes_on_screen === undefined) return <div className="pane placeholder">no health data yet</div>;
  const signals = h.signals ?? {};
  const top = Object.entries(signals).filter(([, v]) => v != null && v >= 0.35).sort((a, b) => (b[1] as number) - (a[1] as number))[0];
  const rate = h.blink_rate;
  const every = h.break_every_min ?? 20;
  const minutes = h.minutes_on_screen ?? 0;
  const due = minutes >= every;
  const blinkSeries = engine.series.blink_rate.filter((v): v is number => v != null);
  return (
    <div className="pane scroll health">
      <section className="h-top">
        <Gauge score={h.fatigue} level={h.level} />
        <div className="h-side">
          <div className="h-stat">
            <span className="h-num" style={{ color: rate == null ? MUTED : rate < 8 ? BAD : rate < 12 ? MID : GOOD }}>
              {rate == null ? "–" : rate.toFixed(0)}
            </span>
            <span className="h-unit">blinks/min</span>
            <Sparkline values={blinkSeries} />
          </div>
          <div className="h-stat">
            <BreakRing minutes={minutes} every={every} />
            <span className={`h-break${due ? " due" : ""}`}>
              {due ? "look away 20 s" : <>break in <b>{Math.ceil(every - minutes)}m</b></>}
              <span className="h-sub"> · {h.breaks ?? 0} taken</span>
            </span>
          </div>
          <div className="h-why">
            {h.fatigue == null ? "needs ~30 s of your face" : top ? <>mostly <b>{NAMES[top[0]]}</b></> : "eyes look rested"}
          </div>
        </div>
      </section>
      <section className="h-signals">
        {SIGNALS.map((s) => {
          const v = signals[s.key];
          return (
            <div key={s.key} className="h-signal" title={s.label}>
              <span className="h-label">{s.label}</span>
              <span className="h-bar">
                <span style={{ width: `${v == null ? 0 : Math.max(v * 100, 3)}%`, background: v == null ? MUTED : severity(v) }} />
              </span>
              <span className="h-value">{s.value(h)}</span>
            </div>
          );
        })}
      </section>
      <div className="h-note">nudges, not a medical measurement</div>
    </div>
  );
}
