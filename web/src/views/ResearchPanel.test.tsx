import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ResearchState } from "../protocol";
import { ResearchContent } from "./ResearchPanel";

const state = (phase: ResearchState["phase"], extra: Partial<ResearchState> = {}): ResearchState => ({
  query: "Compare accessible headphones",
  phase,
  elapsed_s: 1.2,
  cards: [],
  citations: [],
  ...extra,
});

describe("ResearchContent", () => {
  it("renders progressive status before sources arrive", () => {
    const html = renderToStaticMarkup(<ResearchContent state={state("running")} onClear={() => {}} />);
    expect(html).toContain("Gathering current sources");
    expect(html).toContain("running");
  });

  it("renders complete cards with evidence and citation links", () => {
    const html = renderToStaticMarkup(<ResearchContent state={state("complete", {
      cards: [{
        id: "1",
        title: "Primary source",
        url: "https://example.com/source",
        recommendation: "Best fit",
        rationale: "Matches the budget",
        quote: "The current price is $40.",
        contradiction: true,
      }],
      citations: ["https://example.com/source"],
    })} onClear={() => {}} />);
    expect(html).toContain("Primary source");
    expect(html).toContain("Open source 1");
    expect(html).toContain("conflicting");
  });

  it.each(["partial", "failed"] as const)("renders %s and its error", (phase) => {
    const html = renderToStaticMarkup(
      <ResearchContent state={state(phase, { partial: phase === "partial", error: "Sources incomplete" })} onClear={() => {}} />,
    );
    expect(html).toContain(phase);
    expect(html).toContain("Sources incomplete");
  });
});
