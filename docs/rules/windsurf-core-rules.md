# Windsurf Core Rules for This Project

Last reviewed: 2025-08-11

These rules optimize how we work with Windsurf + Cascade in this repository. Keep them short, actionable, and enforceable.

## Operating Principles
- **Single source of truth**: Core orchestration lives in `src/deep-research.ts`; AI wrapper in `src/ai/providers.ts`.
- **Search strategy**: Exa is primary search/crawl; Google Search Grounding augments/cites. Firecrawl slated for removal (only with explicit approval).
- **Do not auto-run commands**: Ask before executing any terminal/external actions.
- **No fake usage**: Never log or reference variables just to silence warnings.
- **Env sync**: Always update `.env.example` when adding flags/keys.
- **Grounded output**: Prefer structured JSON via `responseSchema` and keep citations.

## Workflow Rules (Cascade usage)
- **Research tooling**: Use internal web search (not MCP tools) when external reading is required.
- **Planning**: Call the plan updater before large work or when scope changes.
- **Memory**: Persist key decisions, env flags, and acceptance criteria immediately.
- **Citations**: When using Google Search Grounding, surface inline citations and references.
- **Batched AI calls**: Use `generateBatch()`/`generateBatchWithTools()` for parallelism with concurrency caps.

## Coding Rules
- **Schema-first prompts**: Prefer `responseMimeType: "application/json"` + `responseSchema` (see `providers.ts`).
- **Chunking**: Use `createResearchSplitter()`; default to recursive; use semantic splitter only when embeddings are available and budgeted.
- **Error handling**: Implement retry/backoff on Exa calls and content fetches; skip or fallback on persistent failures.
- **Caching**: Respect provider LRU; add cache keys for Exa search/contents with TTL where appropriate.
- **Tests**: Use Jest (ESM + ts-jest) with provider/network mocks; no external calls in unit tests.

## Acceptance Criteria (Research Paper)
- **Sections**: Abstract, Table of Contents, Introduction, Body, Methodology, Limitations, Key Learnings, References.
- **Structure**: JSON schemas for outline/sections/summary enforced at generation time.
- **Citations**: At least 1+ citation per major section when claims rely on external facts; links are HTTPS and deduplicated.

## References
- Gemini Structured Output: https://ai.google.dev/gemini-api/docs/structured-output
- Grounding with Google Search: https://ai.google.dev/gemini-api/docs/grounding
- Exa API: https://docs.exa.ai/reference/getting-started
- Chunking strategies: https://www.pinecone.io/learn/chunking-strategies/
- Jest ESM + ts-jest: https://jestjs.io/docs/ecmascript-modules, https://kulshekhar.github.io/ts-jest/docs/guides/esm-support
