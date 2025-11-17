export interface ResearchResult {
  learnings: string[];
  visitedUrls: string[];
}

// ===========================
// JSON Schema (plain objects) for Gemini responses
// ===========================
export const SerpQueriesJsonSchema = {
  type: 'array',
  items: {
    type: 'object',
    properties: {
      query: { type: 'string' },
      researchGoal: { type: 'string' }
    },
    required: ['query'],
    additionalProperties: false,
  },
  minItems: 1
} as const;

export const OutlineJsonSchema = {
  type: 'object',
  properties: {
    outline: { type: 'array', items: { type: 'string' } }
  },
  required: ['outline'],
  additionalProperties: false,
} as const;

export const SectionsJsonSchema = {
  type: 'object',
  properties: {
    sections: { type: 'array', items: { type: 'string' } },
    citations: { type: 'array', items: { type: 'string' } }
  },
  required: ['sections'],
  additionalProperties: false,
} as const;

export const SummaryJsonSchema = {
  type: 'object',
  properties: { summary: { type: 'string' } },
  required: ['summary'],
  additionalProperties: false,
} as const;

export const TitleJsonSchema = {
  type: 'object',
  properties: { title: { type: 'string' } },
  required: ['title'],
  additionalProperties: false,
} as const;

export const FeedbackResponseJsonSchema = {
  type: 'object',
  properties: {
    followUpQuestions: { type: 'array', items: { type: 'string' } },
    analysis: { type: 'string' },
    confidenceScore: { type: 'number' }
  },
  required: ['followUpQuestions', 'analysis'],
  additionalProperties: false,
} as const;

export interface MCPResearchResult {
  content: { type: "text"; text: string }[];
  metadata: {
    learnings: string[];
    visitedUrls: string[];
    stats: {
      totalLearnings: number;
      totalSources: number;
    };
  };
}

// ==========================================
// Centralized types and schemas (no runtime bloat with type-only imports)
// ==========================================

// Zod is used for JSON schema enforcement across analyzer/synthesizer/supervisor
import { z } from 'zod';

// Core primitives
export type EmbeddingVector = number[];

export interface Chunk {
  id: string;
  text: string;
  tokens: number;
  embedding?: EmbeddingVector;
  source?: string;
}

export interface SerpQuery {
  query: string;
  researchGoal?: string;
}

export type Citation = string;

// Outline / Sections / Summary
export const OutlineSchema = z.object({
  title: z.string(),
  sections: z.array(z.string()),
}).strict();
export type Outline = z.infer<typeof OutlineSchema>;

export const SectionsSchema = z.object({
  sections: z.array(z.string()),
  citations: z.array(z.string()).optional(),
}).strict();
export type SectionsJSON = z.infer<typeof SectionsSchema>;

export const SummarySchema = z.object({
  summary: z.string(),
}).strict();
export type SummaryJSON = z.infer<typeof SummarySchema>;

export interface Report {
  abstract: string;
  toc: string[];
  introduction: string;
  body: string[];
  methodology: string;
  limitations: string;
  keyLearnings: string[];
  references: string[];
}

// Agent IO contracts
export interface GatherOutput {
  queries: string[];
  chunks: Chunk[];
  sources: string[];
}

export interface AnalyzeOutput {
  findings: string[];
  citations: Citation[];
}

export interface SynthesizeOutput {
  outline: Outline;
  sections: string[];
  citations?: Citation[];
}

export interface SuperviseVerdict {
  ok: boolean;
  reasons?: string[];
  repaired?: SynthesizeOutput;
}

export interface FullResearchResult {
  report: Report;
  meta: {
    timings?: Record<string, number>;
    tokens?: Record<string, number>;
  };
}

// Providers interfaces
export interface EmbeddingsProvider {
  embed(text: string): Promise<EmbeddingVector>;
  countTokens(text: string): Promise<number>;
}

export interface LlmJsonClient<TConfig = unknown> {
  generate<T>(args: { prompt: string; schema: z.ZodType<T>; config?: TConfig }): Promise<T>;
}