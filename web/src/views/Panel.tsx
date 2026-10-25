import { C } from "../colors";
import { engine, useEngine } from "../engine";

const fmt = (v: number | null | undefined, unit = "", nd = 1) =>
  v == null || Number.isNaN(v) ? "–" : `${v.toFixed(nd)}${unit}`;

/** Connection, tracking state and the colour legend. */
export function StatusPanel() {
  const { connected, schemaMismatch, frame, scene } = useEngine();
  return (
    <div className="pane scroll">
      <dl>
        <dt>connection</dt><dd className={connected ? "ok" : "bad"}>{connected ? "live" : "reconnecting…"}</dd>
        {schemaMismatch && <><dt>protocol</dt><dd className="bad">version mismatch: rebuild the dashboard</dd></>}
        <dt>state</dt><dd>{frame?.state ?? "–"}{frame?.phase && frame.phase !== "idle" ? ` · ${frame.phase}` : ""}</dd>
        <dt>fps · latency</dt><dd>{fmt(frame?.fps, "", 0)} · {fmt(frame?.latency_ms, " ms", 0)}</dd>
        <dt>distance</dt><dd>{fmt(frame?.distance_cm, " cm")}</dd>
        <dt>confidence</dt><dd>{fmt(frame?.confidence, "", 2)}</dd>
        <dt>head yaw · pitch</dt><dd>{fmt(frame?.head_ypr?.[0], "°", 0)} · {fmt(frame?.head_ypr?.[1], "°", 0)}</dd>
        <dt>lid</dt><dd>{fmt(frame?.lid_deg ?? scene?.lid_deg, "°", 0)}</dd>
        <dt>camera FOV</dt>
        <dd>{fmt(scene?.camera.hfov_deg, "°")}{scene && !scene.camera.measured && <span className="bad"> (guess: run gaze fov)</span>}</dd>
      </dl>
      <h4>Legend</h4>
      <ul className="legend">
        {([["target", "target"], ["fused", "app (combined)"], ["ml", "ML calibration"], ["eyeball", "eyeball model"],
           ["smoothed", "ring (smoothed)"], ["pointing", "face pointing"], ["rays", "eye rays"]] as const).map(([k, label]) => (
          <li key={k}><span style={{ background: C[k] }} />{label}</li>
        ))}
      </ul>
    </div>
  );
}

/** The active calibration: what was fitted and how well it held out. */
export function CalibrationPanel() {
  const { calibration: cal } = useEngine();
  if (!cal?.calibrated) return <div className="pane muted">not calibrated</div>;
  return (
    <div className="pane scroll">
      <dl>
        <dt>profile</dt><dd>{cal.profile}</dd>
        <dt>dots · clicks</dt><dd>{cal.dots} · {cal.clicks}{cal.head_motion ? " · head sweep" : ""}</dd>
        <dt>calibrator</dt><dd>{cal.chosen}</dd>
        <dt>drift offset</dt>
        <dd>{cal.offset_pt ? `${Math.round(cal.offset_pt[0])}, ${Math.round(cal.offset_pt[1])} pt` : "–"}</dd>
        {cal.eyeball && <>
          <dt>eyeball gain</dt><dd>{cal.eyeball.gain.map((g) => g.toFixed(2)).join(" / ")}</dd>
          <dt>eyeball kappa</dt><dd>{cal.eyeball.kappa_deg.map((k) => `${k.toFixed(1)}°`).join(" / ")}</dd>
        </>}
      </dl>
      <h4>Held-out error</h4>
      <dl className="small">
        {Object.entries(cal.cv_error_pt ?? {}).map(([k, v]) => (
          <div key={k} className="row"><dt>{k}</dt><dd className={k === cal.chosen ? "ok" : undefined}>{fmt(v, " pt", 0)}</dd></div>
        ))}
      </dl>
    </div>
  );
}

/** Commands for the live app (hidden while replaying). */
export function ControlsPanel() {
  const { connected, frame, scene } = useEngine();
  if (scene?.replay) return <div className="pane muted">replay: use the timeline above</div>;
  const busy = !connected || !!frame?.busy;
  return (
    <div className="pane controls">
      <button disabled={busy} onClick={() => engine.send("recalibrate")}>Recalibrate</button>
      <button disabled={busy} onClick={() => engine.send("recentre")}>Re-centre (1 dot)</button>
      <button disabled={busy} onClick={() => engine.send("check")}>Check accuracy</button>
      <button disabled={busy} onClick={() => engine.send("measure")}>Measure (still + moving)</button>
      <button disabled={!connected} onClick={() => engine.send("cycle_smoothing")}>Smoothing: {frame?.smoothing ?? "–"}</button>
      <button disabled={!connected} onClick={() => engine.send("toggle_overlay")}>Ring: {frame?.overlay ? "on" : "off"}</button>
    </div>
  );
}
