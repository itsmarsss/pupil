import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { engine, useEngine } from "../engine";
import type { LipVerdict } from "../protocol";

const MOUTH = "#e599f7";
const RANGE = "#ffd43b";
const LIMIT = "#ff6b6b";
const DEFAULT_THRESHOLD = 0.04;

function verdictText(v: LipVerdict): string {
  if (v.moving === null) return "let through: face seen too little";
  if (v.why === "no face") return "ignored: no face";
  return `${v.moving ? "accepted" : "ignored"}: range ${v.range?.toFixed(3)}`;
}

/** Mouth openness, its rolling movement range and the lip gate's threshold; each utterance is shaded by the
 *  gate's verdict (green accepted, red ignored, grey let through because the face wasn't seen enough). */
export function LipsPlot() {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const verdicts = useRef<LipVerdict[]>([]);
  const origin = useRef(0); // frame time at the plot's x = 0 (the newest sample)
  const { frame, version } = useEngine();
  const gate = frame?.voice?.lip_gate;
  const threshold = gate?.threshold ?? DEFAULT_THRESHOLD;
  const last = frame?.voice?.lips?.at(-1);
  verdicts.current = frame?.voice?.lips ?? [];

  useEffect(() => {
    const el = box.current!;
    const shade = (u: uPlot) => {
      const ctx = u.ctx;
      const { left, top, width, height } = u.bbox;
      ctx.save();
      ctx.font = `${11 * devicePixelRatio}px system-ui`;
      for (const v of verdicts.current) {
        const x0 = Math.max(u.valToPos(v.t0 - origin.current, "x", true), left);
        const x1 = Math.min(u.valToPos(v.t1 - origin.current, "x", true), left + width);
        if (x1 <= x0) continue;
        ctx.fillStyle = v.moving === false ? "rgba(255,107,107,0.16)" : v.moving ? "rgba(61,220,132,0.16)" : "rgba(139,147,161,0.14)";
        ctx.fillRect(x0, top, x1 - x0, height);
        ctx.fillStyle = "#8b93a1";
        ctx.fillText(v.why === "no face" ? "no face" : v.range?.toFixed(3) ?? "?", x0 + 4, top + 14 * devicePixelRatio);
      }
      ctx.restore();
    };
    const opts: uPlot.Options = {
      width: el.clientWidth, height: Math.max(el.clientHeight, 60), legend: { show: false },
      axes: [
        { stroke: "#8b93a1", grid: { stroke: "#232833" }, values: (_u, vals) => vals.map((v) => `${v}s`) },
        { stroke: "#8b93a1", grid: { stroke: "#232833" }, size: 46, values: (_u, vals) => vals.map((v) => v.toFixed(2)) },
      ],
      series: [
        {},
        { label: "openness", stroke: MOUTH, width: 1.5 },
        { label: "movement (1 s range)", stroke: RANGE, width: 1.5 },
        { label: "threshold", stroke: LIMIT, width: 1, dash: [6, 4] },
      ],
      scales: { x: { time: false }, y: { range: (_u, _min, max) => [0, Math.max(max ?? 0, 0.12)] } },
      hooks: { drawClear: [shade] },
    };
    plot.current = new uPlot(opts, [[], [], [], []], el);
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height: Math.max(el.clientHeight, 60) }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
    };
  }, []);

  useEffect(() => {
    const s = engine.series;
    if (!s.t.length) return;
    origin.current = s.t[s.t.length - 1];
    plot.current?.setData([s.t.map((t) => t - origin.current), s.mouth, s.mouth_range, s.t.map(() => threshold)] as uPlot.AlignedData);
  }, [version, threshold]);

  return (
    <div className="pane flush lips">
      <div className="caption lips-head">
        <span><i style={{ background: MOUTH }} />openness</span>
        <span><i style={{ background: RANGE }} />movement, last 1 s</span>
        <span><i style={{ background: LIMIT }} />threshold {threshold.toFixed(3)}</span>
        <span className="muted">
          {gate ? (gate.enabled ? "gate on" : "gate off") : "voice control off"}
          {last ? ` · last: ${verdictText(last)}` : ""}
        </span>
      </div>
      <div className="plot" ref={box} />
    </div>
  );
}
