# Google Search Grounding Rules

Last reviewed: 2025-08-11

Ground model outputs with Google Search to improve factuality and provide citations.

## Rules
- **Enable** the `google_search_retrieval` tool via env flag and provider wiring.
- **Inline citations**: Parse grounding metadata (chunk indices → attributions) and generate inline references and a References section.
- **Combine contexts**: Use URL context (provided URLs/snippets) alongside Google Search results when available.
- **Display suggestions**: Surface Google Search suggestions when present; helps validation and UX.
- **Respect quotas**: Batch enablement; avoid redundant calls when Exa already provides authoritative sources.

## Minimal Flow
1) Compose prompt; if grounding is enabled, include `tools: ['google_search_retrieval']` in provider call.
2) On response, inspect `groundingMetadata` to extract `attributedSources`.
3) Render inline citations and add a consolidated References list.

## Note for this Project
- Grounding augments Exa results. Exa remains the primary search/crawl provider.
- All parsing/citation logic must be deterministic and testable.

## References
- Grounding with Google Search: https://ai.google.dev/gemini-api/docs/grounding
- ADK grounding overview: https://google.github.io/adk-docs/grounding/google_search_grounding/
