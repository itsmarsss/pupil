import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import { engine, useEngine } from "../engine";

interface PlotSpec {
  title: string;
  series: { key: string; label: string; color: string }[];
}

const PLOTS: PlotSpec[] = [
  { title: "Head (°)", series: [
    { key: "yaw", label: "yaw", color: "#ff6b6b" }, { key: "pitch", label: "pitch", color: "#51cf66" },
    { key: "roll", label: "roll", color: "#4dabf7" }] },
  { title: "Distance (cm)", series: [{ key: "distance", label: "distance", color: "#ffd43b" }] },
  { title: "Eye openness", series: [
    { key: "open_l", label: "left", color: "#e599f7" }, { key: "open_r", label: "right", color: "#91a7ff" }] },
  { title: "ML vs eyeball (pt)", series: [{ key: "disagreement", label: "disagreement", color: "#ff9f43" }] },
  { title: "Engine", series: [
    { key: "fps", label: "fps", color: "#3ddc84" }, { key: "latency", label: "latency ms", color: "#adb5bd" }] },
];

function Plot({ spec }: { spec: PlotSpec }) {
  const box = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const { version } = useEngine();

  useEffect(() => {
    const el = box.current!;
    const opts: uPlot.Options = {
      width: el.clientWidth, height: el.clientHeight - 28, title: spec.title,
      legend: { show: false },
      axes: [{ stroke: "#8b93a1", grid: { stroke: "#232833" } }, { stroke: "#8b93a1", grid: { stroke: "#232833" }, size: 40 }],
      series: [{}, ...spec.series.map((s) => ({ label: s.label, stroke: s.color, width: 1.5 }))],
      scales: { x: { time: false } },
    };
    plot.current = new uPlot(opts, [[], ...spec.series.map(() => [])], el);
    const ro = new ResizeObserver(() => plot.current?.setSize({ width: el.clientWidth, height: el.clientHeight - 28 }));
    ro.observe(el);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
    };
  }, [spec]);

  useEffect(() => {
    const s = engine.series;
    if (!s.t.length) return;
    const t0 = s.t[s.t.length - 1];
    plot.current?.setData([s.t.map((t) => (t as number) - t0), ...spec.series.map((x) => s[x.key])] as uPlot.AlignedData);
  }, [version, spec]);

  return <div className="plot" ref={box} />;
}

export function Plots() {
  return <div className="plots">{PLOTS.map((p) => <Plot key={p.title} spec={p} />)}</div>;
}
