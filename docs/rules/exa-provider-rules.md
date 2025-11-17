# Exa Provider Rules (Primary Search/Crawl)

Last reviewed: 2025-08-11

Exa is our always-on search and content provider. Google Search Grounding augments results for citations.

## Env & Flags
- `EXA_API_KEY` (required)
- `ENABLE_EXA_PRIMARY` (optional gate for staged rollout)
- Caching flags: reuse provider LRU; add specific Exa caches where high churn occurs

## Usage Patterns (exa-js)
```ts
import Exa from 'exa-js';
const exa = new Exa(process.env.EXA_API_KEY!);

// 1) Search
const results = await exa.search(query, {
  // optional filters
  startPublishedDate: '2023-01-01',
  numResults: 10,
  includeDomains: ['example.com'],
});

// 2) Contents (clean HTML → parsed content)
// exa-js SDK: pass URLs or Result[] from search
const contents = await exa.getContents(results.results);
// For advanced retrieval (text/highlights/summary), see Exa API docs; options may vary by SDK version.
```

## Rules
- **Deterministic mapping**: Always keep `(query → results → contents)` mapping stable for reproducible tests.
- **Retries**: Exponential backoff on 5xx/429; per-call timeout; skip after N failures.
- **Normalization**: Store normalized `{ url, title, text, highlights, summary }` for downstream steps.
- **Batching**: Prefer batched `getContents()` calls; respect concurrency limits.
- **Caching**: Key by `query` and `url/result-id`; define TTLs to balance freshness with cost.
- **Fallbacks**: If content fetch fails, keep URL reference and rely on Google grounding for citation.

## References
- Getting Started: https://docs.exa.ai/reference/getting-started
- Contents Retrieval: https://docs.exa.ai/reference/contents-retrieval-with-exa-api
- exa-js repo: https://github.com/exa-labs/exa-js
