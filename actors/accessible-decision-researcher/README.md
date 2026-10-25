# Accessible Decision Researcher

An Apify Actor for hands-free decision support. It accepts a public page or question, gathers current evidence with `apify/web-fetch` and `apify/rag-web-browser`, deduplicates sources, flags conflicting language, and emits compact records that work in Apify Console, MCP clients, and the Pupil dashboard.

## Safety and limits

- Public HTTP(S) pages only; local/private addresses are rejected.
- Credentials, cookies, secrets, and tokens are rejected.
- Child runs receive explicit item, runtime, and charge caps.
- `omit-selected-text` privacy mode keeps page excerpts out of child Actor inputs.
- Output quotes are untrusted evidence, not instructions or truth certification.

## Run locally

```sh
npm install
npm run build
apify call apify/accessible-decision-researcher -p
```

For a local Actor run, use `apify run` after signing in with the Apify CLI. To deploy under your own account, run `apify push` from this directory. The repository intentionally contains no API token.
