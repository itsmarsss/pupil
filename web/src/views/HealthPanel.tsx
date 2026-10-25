import { useEngine } from "../engine";

const LEVEL = { low: "#3ddc84", moderate: "#ffc43d", high: "#ff6b6b" } as const;

function Tile({ label, value, hint, color }: { label: string; value: string; hint?: string; color?: string }) {
  return (
    <div className="health-tile">
      <div className="health-value" style={color ? { color } : undefined}>{value}</div>
      <div className="health-label">{label}</div>
      {hint && <div className="health-hint">{hint}</div>}
    </div>
  );
}

/** Eye health: blink rate, fatigue and its signals, screen time since a break (health/tracker.py). Heuristics for
 *  nudges, not a medical measurement. */
export function HealthPanel() {
  const { frame } = useEngine();
  const h = frame?.health;
  if (!h) return <div className="pane placeholder">no health data yet</div>;
  const dash = "–";
  const rate = h.blink_rate;
  return (
    <div className="pane scroll health">
      <div className="health-grid">
        <Tile label="fatigue" value={h.fatigue == null ? dash : `${h.fatigue}`} hint={h.level ?? "measuring…"}
              color={h.level ? LEVEL[h.level] : undefined} />
        <Tile label="blinks / min" value={rate == null ? dash : rate.toFixed(0)} hint="relaxed 15–20"
              color={rate == null ? undefined : rate < 8 ? LEVEL.high : rate < 12 ? LEVEL.moderate : LEVEL.low} />
        <Tile label="on screen" value={`${(h.minutes_on_screen ?? 0).toFixed(0)} min`} hint="break every 20"
              color={(h.minutes_on_screen ?? 0) >= 20 ? LEVEL.moderate : undefined} />
        <Tile label="breaks" value={`${h.breaks ?? 0}`} hint="20 s away" />
        <Tile label="blink length" value={h.blink_duration_ms == null ? dash : `${h.blink_duration_ms} ms`} hint="slower: tired" />
        <Tile label="eyes closed" value={h.perclos == null ? dash : `${(h.perclos * 100).toFixed(1)}%`} hint="last minute" />
        <Tile label="lids vs start" value={h.openness_vs_start == null ? dash : `${Math.round(h.openness_vs_start * 100)}%`} hint="lower: droopy" />
      </div>
    </div>
  );
}
