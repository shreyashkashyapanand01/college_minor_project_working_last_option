import { GoogleGenAI } from "@google/genai";
import { logger } from "../logger.js";
import { LRUCache } from "lru-cache";
import { createHash } from "node:crypto";
import { RecursiveCharacterTextSplitter } from "./text-splitter.js";
import { getEncoding, type Tiktoken, type TiktokenEncoding } from "js-tiktoken";
import { cosineSimilarity } from "ai";

function clampNumber(n: number, min: number, max: number): number {
  if (Number.isNaN(n)) return min;
  return Math.max(min, Math.min(max, n));
}

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) throw new Error("Missing GEMINI_API_KEY");

const client = new GoogleGenAI({
  apiKey: API_KEY,
  ...(process.env.GEMINI_API_ENDPOINT
    ? { apiEndpoint: process.env.GEMINI_API_ENDPOINT }
    : {})
});

const MODEL = process.env.GEMINI_MODEL || "gemini-2.0-flash";

const MAX_TOKENS = clampNumber(
  parseInt(process.env.GEMINI_MAX_OUTPUT_TOKENS || "65536", 10),
  1024,
  65000
);

const TEMPERATURE = parseFloat(process.env.GEMINI_TEMPERATURE || "0.4");
const TOP_P = parseFloat(process.env.GEMINI_TOP_P || "0.9");
const TOP_K = clampNumber(parseInt(process.env.GEMINI_TOP_K || "40", 10), 1, 1000);
const CANDIDATE_COUNT = clampNumber(
  parseInt(process.env.GEMINI_CANDIDATE_COUNT || "2", 10),
  1,
  8
);

const ENABLE_URL_CONTEXT =
  (process.env.ENABLE_URL_CONTEXT || "true").toLowerCase() === "true";
const ENABLE_GEMINI_GOOGLE_SEARCH =
  (process.env.ENABLE_GEMINI_GOOGLE_SEARCH || "true").toLowerCase() === "true";
const ENABLE_GEMINI_CODE_EXECUTION =
  (process.env.ENABLE_GEMINI_CODE_EXECUTION || "false").toLowerCase() === "true";
const ENABLE_GEMINI_FUNCTIONS =
  (process.env.ENABLE_GEMINI_FUNCTIONS || "false").toLowerCase() === "true";

const ENABLE_PROVIDER_CACHE =
  (process.env.ENABLE_PROVIDER_CACHE || "true").toLowerCase() === "true";

const PROVIDER_CACHE_MAX = clampNumber(
  parseInt(process.env.PROVIDER_CACHE_MAX || "100", 10),
  10,
  5000
);
const PROVIDER_CACHE_TTL_MS = clampNumber(
  parseInt(process.env.PROVIDER_CACHE_TTL_MS || "600000", 10),
  1000,
  86400000
);

const ai = client;

const EMBEDDING_MODEL =
  process.env.GEMINI_EMBEDDING_MODEL || "text-embedding-004";

type EmbedVector = number[];
type EmbedResponse = { embeddings?: Array<{ values?: EmbedVector }> };

export async function generateTextEmbedding(text: string): Promise<number[]> {
  const res: EmbedResponse = (await ai.models.embedContent({
    model: EMBEDDING_MODEL,
    contents: [{ role: "user", parts: [{ text }] }]
  })) as unknown as EmbedResponse;
  const v = res?.embeddings?.[0]?.values;
  return Array.isArray(v) ? v : [];
}

type Empty = Record<string, never>;
type Tool = { googleSearch: Empty } | { codeExecution: Empty };
type GenExtra = Partial<{
  responseMimeType: string;
  responseSchema: object;
  tools: Tool[];
}>;

function baseConfig(extra?: GenExtra) {
  return {
    temperature: TEMPERATURE,
    maxOutputTokens: MAX_TOKENS,
    candidateCount: CANDIDATE_COUNT,
    topP: TOP_P,
    topK: TOP_K,
    ...(extra?.responseMimeType
      ? { responseMimeType: extra.responseMimeType }
      : {}),
    ...(extra?.responseSchema ? { responseSchema: extra.responseSchema } : {})
  };
}

type Part = { text?: string };
type ContentMsg = { role: string; parts: Part[] };
type ContentArg = string | { contents: ContentMsg[] };

type GenerateRaw = {
  candidates?: Array<{ content?: { parts?: Array<Part> } }>;
};
type GenerateWrapped = GenerateRaw & {
  response: { text: () => Promise<string> };
};

const providerCache = ENABLE_PROVIDER_CACHE
  ? new LRUCache<string, GenerateWrapped>({
      max: PROVIDER_CACHE_MAX,
      ttl: PROVIDER_CACHE_TTL_MS
    })
  : null;

function hashKey(v: unknown): string {
  try {
    return createHash("sha256").update(JSON.stringify(v)).digest("hex");
  } catch {
    return String(v);
  }
}

function defaultTools(): Tool[] {
  const tools: Tool[] = [];
  if (ENABLE_GEMINI_GOOGLE_SEARCH) tools.push({ googleSearch: {} });
  if (ENABLE_GEMINI_CODE_EXECUTION) tools.push({ codeExecution: {} });
  return tools;
}

function extractTextFromRaw(r: GenerateRaw): string {
  const parts = r.candidates?.[0]?.content?.parts;
  const t = parts?.find((p) => typeof p.text === "string")?.text;
  return t ?? "";
}

async function generateContentInternal(
  prompt: ContentArg,
  extra?: GenExtra
): Promise<GenerateWrapped> {
  const contents: ContentMsg[] =
    typeof prompt === "string"
      ? [{ role: "user", parts: [{ text: prompt }] }]
      : prompt.contents;

  const toolsCombined = [...defaultTools(), ...(extra?.tools || [])];
  const configObj = baseConfig(extra);

  const cacheKey = ENABLE_PROVIDER_CACHE
    ? hashKey({ MODEL, contents, configObj, toolsCombined })
    : "";

  if (providerCache && cacheKey) {
    const hit = providerCache.get(cacheKey);
    if (hit) {
      logger.info({ key: cacheKey.slice(0, 8) }, "[provider-cache] HIT");
      return hit;
    }
    logger.info({ key: cacheKey.slice(0, 8) }, "[provider-cache] MISS");
  }

  const raw = await ai.models.generateContent({
    model: MODEL,
    contents,
    config: configObj,
    tools: toolsCombined.length ? toolsCombined : undefined
  });

  const text = extractTextFromRaw(raw as GenerateRaw);

  const wrapped: GenerateWrapped = Object.assign({}, raw, {
    response: { text: async () => text }
  });

  if (providerCache) providerCache.set(cacheKey, wrapped);

  return wrapped;
}

export const researchModel = {
  generateContent: (p: ContentArg) => generateContentInternal(p)
};

export async function countTokens(
  contents: Array<{ role: string; parts: Array<{ text?: string }> }>
) {
  const r = await ai.models.countTokens({ model: MODEL, contents });
  return (r.totalTokens as number) ?? 0;
}

export async function trimPrompt(prompt: string, max: number) {
  if (!prompt) return "";
  const contents = [{ role: "user", parts: [{ text: prompt }] }];
  const tlen = await countTokens(contents);
  if (tlen <= max) return prompt;
  const overflow = tlen - max;
  const approx = Math.max(2, Math.floor(prompt.length / Math.max(1, tlen)));
  const sliceLen = Math.max(140, prompt.length - overflow * approx);
  return prompt.slice(0, sliceLen);
}

export async function callGeminiProConfigurable(
  prompt: string,
  opts?: { json?: boolean; schema?: object; tools?: Tool[] }
): Promise<string> {
  const extra: GenExtra | undefined = opts?.json
    ? {
        responseMimeType: "application/json",
        ...(opts?.schema ? { responseSchema: opts.schema } : {})
      }
    : undefined;

  const wrapped = await generateContentInternal(
    {
      contents: [{ role: "user", parts: [{ text: prompt }] }]
    },
    {
      ...(extra || {}),
      ...(opts?.tools ? { tools: opts.tools } : {})
    }
  );

  return await wrapped.response.text();
}

export async function generateBatch(
  prompts: ContentArg[],
  extra?: GenExtra,
  concurrency = 5
) {
  const results: GenerateWrapped[] = [];
  const running: Promise<void>[] = [];
  let index = 0;

  async function worker() {
    while (index < prompts.length) {
      const i = index++;
      const p = prompts[i];
      const w = await generateContentInternal(p, extra);
      results[i] = w;
    }
  }

  for (let i = 0; i < concurrency; i++) running.push(worker());
  await Promise.all(running);
  return results;
}

export async function generateBatchWithTools(
  prompts: ContentArg[],
  tools: Tool[],
  extra?: Omit<GenExtra, "tools">,
  concurrency = 5
) {
  const results: GenerateWrapped[] = [];
  const running: Promise<void>[] = [];
  let index = 0;

  async function worker() {
    while (index < prompts.length) {
      const i = index++;
      const p = prompts[i];
      const w = await generateContentInternal(p, {
        ...(extra || {}),
        tools
      });
      results[i] = w;
    }
  }

  for (let i = 0; i < concurrency; i++) running.push(worker());
  await Promise.all(running);
  return results;
}
