import { useEngine } from "../engine";

const time = (t: number) => new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

/** The voice controller: mic state, what it heard, what's waiting for "confirm", and its recent log. */
export function VoicePanel() {
  const { frame } = useEngine();
  const v = frame?.voice;
  if (!v) {
    return (
      <div className="pane muted">
        Voice control isn't running. Start it with <code>uv run gaze control --dashboard</code>.
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
