import { useEngine } from "../engine";
import type { ResearchState } from "../protocol";

const retrieved = (value?: string) => {
  if (!value) return "retrieval time unavailable";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
};

export function ResearchContent({ state, onClear }: { state?: ResearchState; onClear: () => void }) {
  if (!state) {
    return (
      <div className="pane research-empty">
        <strong>Decision research</strong>
        <span>Look at a page and say “compare this with three alternatives.”</span>
        <small>Public sources only. Results are evidence, not verified facts.</small>
      </div>
    );
  }

  return (
    <div className="pane scroll research-panel" aria-live="polite">
      <div className="research-head">
        <div>
          <strong>{state.query}</strong>
          <div className={`research-phase ${state.phase}`}>
            {state.phase} · {state.elapsed_s.toFixed(1)}s
            {state.partial ? " · incomplete" : ""}
          </div>
        </div>
        <button
          className="research-clear"
          onClick={onClear}
          aria-label="Delete research history"
        >
          Delete history
        </button>
      </div>
      {state.error && <div className="research-error">{state.error}</div>}
      {!state.cards.length && state.phase !== "failed" && (
        <div className="research-progress">
          <span className="research-spinner" /> Gathering current sources…
        </div>
      )}
      <ol className="research-cards">
        {state.cards.map((card, index) => (
          <li
            key={`${state.run_id ?? "pending"}-${card.id}`}
            className={`research-card${card.contradiction ? " contradiction" : ""}`}
            tabIndex={0}
            aria-label={`Research result ${index + 1}: ${card.title}`}
          >
            <div className="research-card-top">
              <span className="source-number">{index + 1}</span>
              <h3>{card.title}</h3>
              {(card.contradiction || card.uncertain) && (
                <span className="evidence-flag">
                  {card.contradiction ? "conflicting" : "uncertain"}
                </span>
              )}
            </div>
            {card.recommendation && <p className="recommendation">{card.recommendation}</p>}
            {card.rationale && <p>{card.rationale}</p>}
            <div className="research-meta">Retrieved {retrieved(card.retrieved_at)}</div>
            <details>
              <summary>Why? Show evidence and source</summary>
              {card.quote && <blockquote>{card.quote}</blockquote>}
              {card.url && (
                <a href={card.url} target="_blank" rel="noreferrer">
                  Open source {index + 1}: {card.url}
                </a>
              )}
            </details>
          </li>
        ))}
      </ol>
      <p className="research-disclaimer">Web content is untrusted and may be incomplete. Check primary sources.</p>
    </div>
  );
}

export function ResearchPanel() {
  const engine = useEngine();
  return <ResearchContent state={engine.frame?.voice?.research} onClear={() => engine.send("clear_research")} />;
}
