import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { engine, useEngine, type Series } from "../engine";

export interface PlotSpec {
  id: string;
  title: string;
  series: { key: string; label: string; color: string }[];
  range?: [number, number]; // fixed y range (default: auto)
}

export const PLOTS: PlotSpec[] = [
  { id: "head", title: "Head (°)", series: [
    { key: "yaw", label: "yaw", color: "#ff6b6b" }, { key: "pitch", label: "pitch", color: "#51cf66" },
    { key: "roll", label: "roll", color: "#4dabf7" }] },
  { id: "distance", title: "Distance (cm)", series: [{ key: "distance", label: "distance", color: "#ffd43b" }] },
  { id: "lid", title: "Lid (°)", series: [{ key: "lid", label: "hinge", color: "#9aa4b2" }] },
  { id: "eyes", title: "Eye openness", series: [
    { key: "open_l", label: "left", color: "#e599f7" }, { key: "open_r", label: "right", color: "#91a7ff" }] },
  { id: "disagreement", title: "ML vs eyeball (pt)", series: [{ key: "disagreement", label: "disagreement", color: "#ff9f43" }] },
  { id: "health", title: "Blinks & fatigue", series: [
    { key: "blink_rate", label: "blinks/min", color: "#4dabf7" }, { key: "fatigue", label: "fatigue", color: "#ff6b6b" }] },
  { id: "engine", title: "Engine", series: [
    { key: "fps", label: "fps", color: "#3ddc84" }, { key: "latency", label: "latency ms", color: "#adb5bd" }] },
];

/** Compact tick labels: 1234 → 1.2k, so wide numbers never get clipped. */
function compact(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return `${+(v / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${+(v / 1e3).toFixed(1)}k`;
  return `${+v.toFixed(a < 1 && a > 0 ? 2 : 1)}`;
}

/** `source` picks the data (default: the live 20 s series; the Health panel passes its 30-minute one). */
export function Plot({ spec, source = () => engine.series, unit = "s" }: { spec: PlotSpec; source?: () => Series; unit?: "s" | "min" }) {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const { version } = useEngine();

  useEffect(() => {
    const el = box.current!;
    const opts: uPlot.Options = {
      width: el.clientWidth, height: Math.max(el.clientHeight, 60), legend: { show: false },
      axes: [
        { stroke: "#8b93a1", grid: { stroke: "#232833" }, values: (_u, vals) => vals.map((v) => (unit === "min" ? `${+(v / 60).toFixed(1)}m` : `${v}s`)) },
        { stroke: "#8b93a1", grid: { stroke: "#232833" }, size: 46, values: (_u, vals) => vals.map(compact) },
      ],
      series: [{}, ...spec.series.map((s) => ({ label: s.label, stroke: s.color, width: 1.5 }))],
      scales: { x: { time: false }, ...(spec.range ? { y: { range: spec.range } } : {}) },
    };
    plot.current = new uPlot(opts, [[], ...spec.series.map(() => [])], el);
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height: Math.max(el.clientHeight, 60) }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
    };
  }, [spec]);

  useEffect(() => {
    const s = source();
    if (!s.t.length) return;
    const t0 = s.t[s.t.length - 1];
    plot.current?.setData([s.t.map((t) => (t as number) - t0), ...spec.series.map((x) => s[x.key])] as uPlot.AlignedData);
  }, [version, spec]);

  return <div className="plot" ref={box} />;
}
