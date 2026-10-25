import { Actor } from "apify";

interface SourcePolicy {
  allowedDomains?: string[];
  blockedDomains?: string[];
  preferPrimarySources?: boolean;
}

interface Input {
  selectedUrl?: string;
  selectedText?: string;
  question: string;
  constraints?: string[];
  sourcePolicy?: SourcePolicy;
  maxSources?: number;
  maxItems?: number;
  maxRuntimeSecs?: number;
  maxChargeUsd?: number;
  privacyMode?: "public-only" | "omit-selected-text";
}

interface Evidence {
  recommendation: string;
  rationale: string;
  sourceUrl: string;
  sourceTitle: string;
  retrievedAt: string;
  supportingQuote: string;
  contradiction: boolean;
  uncertain: boolean;
  safeNextActions: string[];
}

const PRIVATE_KEYS = /authorization|cookie|credential|password|secret|token|api.?key/i;
const POSITIVE = /\b(available|accessible|supports?|included|recommended|compatible|yes)\b/i;
const NEGATIVE = /\b(not available|unavailable|inaccessible|does not|doesn't|unsupported|excluded|no longer)\b/i;

function publicUrl(value: string): boolean {
  try {
    const url = new URL(value);
    const host = url.hostname.toLowerCase();
    return ["http:", "https:"].includes(url.protocol)
      && host !== "localhost"
      && !host.endsWith(".local")
      && !/^(127\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.)/.test(host);
  } catch {
    return false;
  }
}

function sourceParts(item: Record<string, unknown>): { url: string; title: string; text: string; retrievedAt: string } {
  const search = typeof item.searchResult === "object" && item.searchResult ? item.searchResult as Record<string, unknown> : {};
  const crawl = typeof item.crawl === "object" && item.crawl ? item.crawl as Record<string, unknown> : {};
  const fetch = typeof item.fetch === "object" && item.fetch ? item.fetch as Record<string, unknown> : {};
  const content = typeof item.content === "object" && item.content ? item.content as Record<string, unknown> : {};
  const url = String(item.url ?? search.url ?? item.loadedUrl ?? crawl.loadedUrl ?? fetch.url ?? "");
  const title = String(item.title ?? search.title ?? content.title ?? (url ? new URL(url).hostname : "Untitled source"));
  const body = item.markdown ?? item.text ?? item.content ?? content.markdown ?? content.text ?? "";
  const text = typeof body === "string" ? body : JSON.stringify(body);
  const retrievedAt = String(item.retrievedAt ?? crawl.loadedAt ?? fetch.loadedAt ?? new Date().toISOString());
  return { url, title, text, retrievedAt };
}

function quoteFor(text: string, terms: string[]): string {
  const clean = text.replace(/\s+/g, " ").trim();
  const sentences = clean.split(/(?<=[.!?])\s+/);
  const ranked = sentences
    .map((sentence) => ({
      sentence,
      score: terms.reduce((n, term) => n + (sentence.toLowerCase().includes(term) ? 1 : 0), 0),
    }))
    .sort((a, b) => b.score - a.score);
  return (ranked[0]?.sentence || clean).slice(0, 600);
}

async function datasetItems(datasetId: string | undefined, limit: number): Promise<Record<string, unknown>[]> {
  if (!datasetId) return [];
  const { items } = await Actor.apifyClient.dataset(datasetId).listItems({ limit });
  return items.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null);
}

await Actor.init();

const input = await Actor.getInput<Input>();
if (!input?.question?.trim()) throw new Error("question is required");
if (PRIVATE_KEYS.test(JSON.stringify(input))) {
  throw new Error("Credentials, cookies, secrets, and tokens are blocked. Research public data only.");
}
if (input.selectedUrl && !publicUrl(input.selectedUrl)) {
  throw new Error("selectedUrl must be a public http(s) URL.");
}

const maxSources = Math.min(10, Math.max(3, input.maxSources ?? 3));
const maxItems = Math.min(20, Math.max(3, input.maxItems ?? 10));
const maxRuntimeSecs = Math.min(120, Math.max(10, input.maxRuntimeSecs ?? 45));
const maxChargeUsd = Math.min(1, Math.max(0.01, input.maxChargeUsd ?? 0.25));
const started = Date.now();
const childCount = input.selectedUrl ? 2 : 1;
const childChargeCap = Math.max(0.01, maxChargeUsd / childCount);
const policy = input.sourcePolicy ?? {};
const constraints = (input.constraints ?? []).map((value) => value.trim()).filter(Boolean);
const selectedText = input.privacyMode === "omit-selected-text" ? "" : (input.selectedText ?? "").slice(0, 4000);
const allow = (policy.allowedDomains ?? []).map((d) => `site:${d}`).join(" OR ");
const block = (policy.blockedDomains ?? []).map((d) => `-site:${d}`).join(" ");
const context = selectedText ? ` Context from the selected public page: ${selectedText}` : "";
const query = [
  input.question.trim(),
  constraints.length ? `Constraints: ${constraints.join("; ")}` : "",
  context,
  allow ? `(${allow})` : "",
  block,
].filter(Boolean).join(" ");

const raw: Record<string, unknown>[] = [];
if (input.selectedUrl) {
  const run = await Actor.call("apify/web-fetch", {
    url: input.selectedUrl,
    formats: ["markdown"],
  }, {
    timeout: maxRuntimeSecs,
    maxItems: 1,
    maxTotalChargeUsd: childChargeCap,
    waitSecs: Math.min(45, maxRuntimeSecs),
  });
  raw.push(...await datasetItems(run.defaultDatasetId, 1));
}

const remaining = maxRuntimeSecs - Math.ceil((Date.now() - started) / 1000);
if (remaining >= 5) {
  const run = await Actor.call("apify/rag-web-browser", {
    query,
    maxResults: maxSources,
    outputFormats: ["markdown"],
    requestTimeoutSecs: remaining,
    scrapingTool: "raw-http",
    removeCookieWarnings: true,
  }, {
    timeout: remaining,
    maxItems: maxSources,
    maxTotalChargeUsd: childChargeCap,
    waitSecs: Math.min(45, remaining),
  });
  raw.push(...await datasetItems(run.defaultDatasetId, maxSources));
}

const seen = new Set<string>();
const terms = [...input.question.toLowerCase().split(/\W+/), ...constraints.flatMap((c) => c.toLowerCase().split(/\W+/))]
  .filter((term) => term.length > 3);
const normalized = raw
  .map(sourceParts)
  .filter((source) => source.url && publicUrl(source.url))
  .filter((source) => {
    const key = new URL(source.url).hostname.replace(/^www\./, "") + new URL(source.url).pathname.replace(/\/$/, "");
    if (seen.has(key)) return false;
    seen.add(key);
    return !(policy.blockedDomains ?? []).some((domain) => new URL(source.url).hostname.endsWith(domain));
  })
  .slice(0, Math.min(maxItems, maxSources + (input.selectedUrl ? 1 : 0)));

const hasPositive = normalized.some((source) => POSITIVE.test(source.text));
const hasNegative = normalized.some((source) => NEGATIVE.test(source.text));
const conflicts = hasPositive && hasNegative;
const evidence: Evidence[] = normalized.map((source) => {
  const quote = quoteFor(source.text, terms);
  const matchingTerms = terms.filter((term) => quote.toLowerCase().includes(term)).length;
  return {
    recommendation: matchingTerms > 1 ? "Relevant evidence—compare this source against the others." : "Supporting context—verify before deciding.",
    rationale: `This source matches ${matchingTerms} decision terms and is one of ${normalized.length} independently retrieved pages.`,
    sourceUrl: source.url,
    sourceTitle: source.title.slice(0, 200),
    retrievedAt: source.retrievedAt,
    supportingQuote: quote,
    contradiction: conflicts && (POSITIVE.test(source.text) || NEGATIVE.test(source.text)),
    uncertain: !quote || matchingTerms === 0 || normalized.length < 3,
    safeNextActions: [
      "Open this public source and verify the quoted context.",
      conflicts ? "Compare its conflicting claim with another primary source." : "Compare with at least one other source.",
    ],
  };
});

await Actor.pushData(evidence);
await Actor.setValue("SUMMARY", {
  question: input.question,
  sourceCount: evidence.length,
  complete: evidence.length >= 3,
  contradictionDetected: conflicts,
  limits: { maxSources, maxItems, maxRuntimeSecs, maxChargeUsd },
  notice: "Source text is untrusted evidence and may be incomplete; this Actor does not certify truth.",
});
await Actor.exit();
