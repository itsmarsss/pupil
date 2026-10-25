import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { engine, useEngine } from "../engine";
import type { LipGateState, LipVerdict } from "../protocol";

const MOUTH = "#e599f7";
const MOTION = "#ffd43b";
const LIMIT = "#ff6b6b";
const OPEN_LINE = "#b197fc";

function gateText(g: LipGateState | undefined): string {
  if (!g) return "lip gate off";
  const state = g.open ? "open" : `closed (${g.reason})`;
  return g.mode === "shadow" ? `shadow mode: would be ${state}` : `gate ${state}`;
}

/** The live lip gate (control/lips.py): mouth openness, lip movement over the gate's 0.4 s window, the movement
 *  threshold (adapts to your resting jitter) and the held-open line. The background is tinted while the gate is
 *  open; each transcript is marked with the verdict (green kept, red dropped because your lips didn't move). */
export function LipsPlot() {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const verdicts = useRef<LipVerdict[]>([]);
  const origin = useRef(0); // frame time at the plot's x = 0 (the newest sample)
  const { frame, version } = useEngine();
  const gate = frame?.voice?.lip_gate;
  const last = frame?.voice?.lips?.at(-1);
  verdicts.current = frame?.voice?.lips ?? [];

  useEffect(() => {
    const el = box.current!;
    const shade = (u: uPlot) => {
      const ctx = u.ctx;
      const { left, top, width, height } = u.bbox;
      const xs = u.data[0] as number[];
      const open = engine.series.lip_open;
      ctx.save();
      ctx.fillStyle = "rgba(77,171,247,0.10)"; // gate open
      for (let i = 0; i < xs.length - 1; i++) {
        if (open[open.length - xs.length + i] !== 1) continue;
        const x0 = u.valToPos(xs[i], "x", true), x1 = u.valToPos(xs[i + 1], "x", true);
        ctx.fillRect(x0, top, x1 - x0 + 0.5, height);
      }
      ctx.font = `${11 * devicePixelRatio}px system-ui`;
      for (const v of verdicts.current) {
        const x0 = Math.max(u.valToPos(v.t0 - origin.current, "x", true), left);
        const x1 = Math.min(u.valToPos(v.t1 - origin.current, "x", true), left + width);
        if (x1 <= x0) continue;
        ctx.fillStyle = v.moving ? "rgba(61,220,132,0.22)" : "rgba(255,107,107,0.22)";
        ctx.fillRect(x0, top + height - 6 * devicePixelRatio, x1 - x0, 6 * devicePixelRatio);
        ctx.fillStyle = v.moving ? "#3ddc84" : "#ff6b6b";
        ctx.fillText(v.moving ? "kept" : "dropped", x0 + 2, top + 13 * devicePixelRatio);
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
        { label: "movement (0.4 s)", stroke: MOTION, width: 1.5 },
        { label: "movement threshold", stroke: LIMIT, width: 1, dash: [6, 4] },
        { label: "held open above", stroke: OPEN_LINE, width: 1, dash: [2, 4] },
      ],
      scales: { x: { time: false }, y: { range: (_u, _min, max) => [0, Math.max(max ?? 0, 0.12)] } },
      hooks: { drawClear: [shade] },
    };
    plot.current = new uPlot(opts, [[], [], [], [], []], el);
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
    const line = (v: number | undefined) => s.t.map(() => v ?? null);
    plot.current?.setData([s.t.map((t) => t - origin.current), s.mouth, s.mouth_motion,
      line(gate?.threshold), line(gate?.open_above)] as uPlot.AlignedData);
  }, [version, gate?.threshold, gate?.open_above]);

  return (
    <div className="pane flush lips">
      <div className="caption lips-head">
        <span><i style={{ background: MOUTH }} />openness</span>
        <span><i style={{ background: MOTION }} />movement</span>
        <span><i style={{ background: LIMIT }} />threshold {gate ? gate.threshold.toFixed(3) : "–"}</span>
        <span><i style={{ background: OPEN_LINE }} />held open {gate ? gate.open_above.toFixed(3) : "–"}</span>
        <span className={gate?.open ? "ok" : "muted"}>
          {gateText(gate)}
          {last ? ` · last: ${last.moving ? "kept" : "dropped"}` : ""}
        </span>
      </div>
      <div className="plot" ref={box} />
    </div>
  );
}
