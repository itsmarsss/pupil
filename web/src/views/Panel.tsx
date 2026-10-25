import { C } from "../colors";
import { engine, useEngine } from "../engine";

const fmt = (v: number | null | undefined, unit = "", nd = 1) => (v == null || Number.isNaN(v) ? "–" : `${v.toFixed(nd)}${unit}`);

/** Status, calibration summary, legend and controls. */
export function Panel() {
  const { connected, schemaMismatch, frame, scene, calibration: cal } = useEngine();
  const busy = frame?.busy;
  return (
    <aside className="panel">
      <section>
        <h3>Engine</h3>
        <dl>
          <dt>connection</dt><dd className={connected ? "ok" : "bad"}>{connected ? "live" : "reconnecting…"}</dd>
          {schemaMismatch && <><dt>protocol</dt><dd className="bad">version mismatch: rebuild the dashboard</dd></>}
          <dt>state</dt><dd>{frame?.state ?? "–"}{frame?.phase && frame.phase !== "idle" ? ` · ${frame.phase}` : ""}</dd>
          <dt>fps · latency</dt><dd>{fmt(frame?.fps, "", 0)} · {fmt(frame?.latency_ms, " ms", 0)}</dd>
          <dt>distance</dt><dd>{fmt(frame?.distance_cm, " cm")}</dd>
          <dt>confidence</dt><dd>{fmt(frame?.confidence, "", 2)}</dd>
          <dt>lid</dt><dd>{fmt(frame?.lid_deg ?? scene?.lid_deg, "°", 0)}</dd>
          <dt>camera FOV</dt><dd>{fmt(scene?.camera.hfov_deg, "°")} {scene && !scene.camera.measured && <span className="bad">(guess: run gaze fov)</span>}</dd>
        </dl>
      </section>
      <section>
        <h3>Calibration</h3>
        {cal?.calibrated ? (
          <dl>
            <dt>profile</dt><dd>{cal.profile}</dd>
            <dt>dots · clicks</dt><dd>{cal.dots} · {cal.clicks}{cal.head_motion ? " · head sweep" : ""}</dd>
            <dt>calibrator</dt><dd>{cal.chosen}</dd>
            {cal.eyeball && <>
              <dt>eyeball gain</dt><dd>{cal.eyeball.gain.map((g) => g.toFixed(2)).join(" / ")}</dd>
              <dt>eyeball kappa</dt><dd>{cal.eyeball.kappa_deg.map((k) => `${k.toFixed(1)}°`).join(" / ")}</dd>
            </>}
            <dt>held-out error</dt>
            <dd className="small">
              {Object.entries(cal.cv_error_pt ?? {}).map(([k, v]) => <div key={k}>{k}: {fmt(v, " pt", 0)}</div>)}
            </dd>
          </dl>
        ) : <p className="muted">not calibrated</p>}
      </section>
      <section>
        <h3>Legend</h3>
        <ul className="legend">
          {([["target", "target"], ["fused", "app (combined)"], ["ml", "ML calibration"], ["eyeball", "eyeball model"],
             ["smoothed", "ring (smoothed)"], ["pointing", "face pointing"], ["rays", "eye rays"]] as const).map(([k, label]) => (
            <li key={k}><span style={{ background: C[k] }} />{label}</li>
          ))}
        </ul>
      </section>
      <section className="controls">
        <h3>Controls</h3>
        <button disabled={!connected || busy} onClick={() => engine.send("recalibrate")}>Recalibrate</button>
        <button disabled={!connected || busy} onClick={() => engine.send("check")}>Check accuracy</button>
        <button disabled={!connected || busy} onClick={() => engine.send("measure")}>Measure (still + moving)</button>
        <button disabled={!connected} onClick={() => engine.send("cycle_smoothing")}>Smoothing: {frame?.smoothing ?? "–"}</button>
        <button disabled={!connected} onClick={() => engine.send("toggle_overlay")}>Ring: {frame?.overlay ? "on" : "off"}</button>
      </section>
    </aside>
  );
}
