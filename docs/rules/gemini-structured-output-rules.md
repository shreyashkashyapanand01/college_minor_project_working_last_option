# Gemini Structured Output Rules

Last reviewed: 2025-08-11

Use schema-constrained JSON with Gemini for reliable parsing and evaluation.

## Rules
- **Always set** `responseMimeType: "application/json"` and provide a **minimal** `responseSchema`.
- **Keep schemas small**: Schema tokens count against input budget; avoid overly deep nesting and huge enums.
- **Validate** responses with `zod` and handle parse failures with a bounded retry.
- **Batch** where feasible using `generateBatch()` with identical `generationConfig` + caching.
- **Token hygiene**: Use `countTokens()` to budget prompts and consider pre-trimming with `adaptivePrompt()`.

## Example (TypeScript)
```ts
const response = await model.generateContent({
  contents: [{ role: 'user', parts: [{ text: prompt }] }],
  generationConfig: {
    responseMimeType: 'application/json',
    responseSchema: {
      type: 'object',
      properties: {
        summary: { type: 'string' },
        sources: { type: 'array', items: { type: 'string' } }
      },
      required: ['summary', 'sources']
    },
    // optional tuning
    temperature: 0.4,
    topP: 0.9,
    topK: 40,
    candidateCount: 1,
    maxOutputTokens: Number(process.env.GEMINI_MAX_OUTPUT_TOKENS || 65536)
  }
});
```

## Project Conventions
- **Schemas**:
  - `outline`: `{ outline: string[] }`
  - `sections`: `{ sections: string[], citations?: string[] }`
  - `summary`: `{ summary: string }`
- **Env flags**: `ENABLE_GEMINI_GOOGLE_SEARCH`, `ENABLE_GEMINI_CODE_EXECUTION`, `ENABLE_GEMINI_FUNCTIONS`, `ENABLE_PROVIDER_CACHE`.

## References
- Structured Output: https://ai.google.dev/gemini-api/docs/structured-output
- Vertex AI notes on `responseMimeType`/schema: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/control-generated-output
