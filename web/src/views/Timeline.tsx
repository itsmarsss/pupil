import { engine, useEngine } from "../engine";

const PHASE_COLOR: Record<string, string> = {
  calib: "#5aa0ff", motion: "#ffb300", valid: "#3ddc84", valid_still: "#3ddc84", valid_moving: "#ff9f43",
  recentre: "#b388ff",
};

/** Playback controls for a replayed session; hidden for the live tracker. */
export function Timeline() {
  const { scene, frame } = useEngine();
  const r = scene?.replay;
  if (!r) return null;
  const state = frame?.replay;
  const index = state?.index ?? 0;
  const pct = (i: number) => `${(100 * i) / Math.max(r.frames - 1, 1)}%`;
  return (
    <div className="timeline">
      <span className="session">replay · {r.session}</span>
      <button onClick={() => engine.send(state?.playing ? "pause" : "play")}>{state?.playing ? "Pause" : "Play"}</button>
      <button onClick={() => engine.send("step", -1)}>◀</button>
      <button onClick={() => engine.send("step", 1)}>▶</button>
      <select value={state?.speed ?? 1} onChange={(e) => engine.send("speed", Number(e.target.value))}>
        {[0.25, 0.5, 1, 2, 4].map((v) => <option key={v} value={v}>{v}×</option>)}
      </select>
      <div className="track">
        {r.markers.map((m) => (
          <span key={m.index} className="marker" style={{ left: pct(m.index), background: PHASE_COLOR[m.phase ?? ""] ?? "#666" }}
                title={m.phase ?? ""} />
        ))}
        <input type="range" min={0} max={r.frames - 1} value={index}
               onChange={(e) => engine.send("seek", Number(e.target.value))} />
      </div>
      <span className="pos">{index + 1}/{r.frames} · {frame?.phase ?? ""}</span>
    </div>
  );
}
