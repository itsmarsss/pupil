import { useEngine } from "../engine";
import type { ModelCall } from "../protocol";

const ROLE: Record<string, string> = {
  stt: "hearing", jev: "Jev", extract: "extracting", dictation: "dictation", agent: "agent", tts: "speaking",
};

/** One model row: its name, and either "calling…" (live) or how long its last call took and how long ago. */
function ModelRow({ m }: { m: ModelCall }) {
  const secs = m.t ? Math.max(0, Math.round(Date.now() / 1000 - m.t)) : null;
  const ago = secs == null ? "" : `${secs < 60 ? `${secs}s` : `${Math.round(secs / 60)}m`} ago`;
  const last = m.active ? `${m.what || "calling"}…`
    : m.ms != null ? `${m.ms} ms${ago ? ` · ${ago}` : ""}`
    : ago || "not called yet"; // a stream (hearing) that has ended has no single duration
  return (
    <>
      <dt title={m.does}>{ROLE[m.role] ?? m.role}</dt>
      <dd className={m.active ? "model live" : "model"} title={m.does}>
        <span className="model-dot" />{m.model} <span className="muted">{last}</span>
      </dd>
    </>
  );
}

const time = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** The voice controller: mic state, what it heard, what's waiting for "confirm", and its recent log. */
export function VoicePanel() {
  const { frame } = useEngine();
  const v = frame?.voice;
  if (!v) {
    return (
      <div className="pane muted">
        Voice control isn't running. Start it with <code>uv run pupil control --dashboard</code>.
      </div>
    );
  }
  return (
    <div className="pane scroll">
      <dl>
        <dt>controller</dt><dd className={v.connected ? "ok" : "bad"}>{v.connected ? "connected" : "quiet"}</dd>
        <dt>mic</dt><dd>{v.mic ?? "–"}</dd>
        <dt>heard</dt><dd className="heard">{v.heard ? `“${v.heard}”` : "–"}</dd>
        <dt>target</dt><dd>{v.target?.label ?? "–"}</dd>
      </dl>
      {v.pending && <div className="pending">Waiting for “confirm”: {v.pending}</div>}
      {v.models && v.models.length > 0 && (
        <>
          <h4>Models</h4>
          <dl>{v.models.map((m) => <ModelRow key={m.role} m={m} />)}</dl>
        </>
      )}
      <h4>Log</h4>
      <ul className="voice-log">
        {[...v.log].reverse().map((e, i) => (
          <li key={i} className={e.kind}>
            <span className="when">{time(e.t)}</span>
            {e.kind === "heard" ? <q>{e.text}</q> : e.text}
          </li>
        ))}
      </ul>
    </div>
  );
}
