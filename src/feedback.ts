import { LRUCache } from 'lru-cache';
import { z } from 'zod';
import { callGeminiProConfigurable } from './ai/providers.js';
import { OutputManager } from './output-manager.js';
import { feedbackPromptTemplate, systemPrompt } from './prompt.js';
import { validateAcademicOutput, conductResearch } from './deep-research.js';
import {
  generateProgressBar as terminalProgressBar,
  TERMINAL_CONTROLS,
  getTerminalDimensions
} from './terminal-utils.js';

interface FeedbackOptions {
  query: string;
  numQuestions?: number;
  researchGoal?: string;
  depth?: number;
  breadth?: number;
  existingLearnings?: string[];
}

const output = new OutputManager();

const FEEDBACK_CACHE_CONFIG = {
  max: 100,
  ttl: 300000,
  dispose: (value: FeedbackResponse) => {
    OutputManager.logCacheEviction(value);
  }
};

const feedbackCache = new LRUCache<string, FeedbackResponse>(FEEDBACK_CACHE_CONFIG);

const FeedbackResponseSchema = z.object({
  followUpQuestions: z.array(z.string().min(10)).max(5),
  analysis: z.string().min(100),
  confidenceScore: z.number().min(0).max(1).optional()
}).passthrough();

export type FeedbackResponse = z.infer<typeof FeedbackResponseSchema>;

interface FeedbackAnalysis {
  readonly [key: string]: unknown;
  readonly score: number;
  readonly keywords: readonly string[];
  readonly suggestions: string[];
  readonly analysisDate: Date;
}

const DEFAULT_KEYWORDS = ['example', 'test', 'dummy'] as const;

async function analyzePastFeedback(query: string): Promise<FeedbackAnalysis> {
  const { width } = getTerminalDimensions();
  const analysisProgress = terminalProgressBar({
    current: 0,
    total: 100,
    width: width - 20,
    label: 'Analyzing feedback'
  });

  try {
    process.stdout.write(`${TERMINAL_CONTROLS.cursorHide}${analysisProgress}`);

    if (typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Invalid query: Must be a non-empty string');
    }

    const startTime = performance.now();
    const matchedKeywords: string[] = [];
    const suggestions: string[] = [];

    const foundKeyword = DEFAULT_KEYWORDS.find(keyword =>
      query.toLowerCase().includes(keyword.toLowerCase())
    );

    if (foundKeyword) {
      matchedKeywords.push(foundKeyword);
      suggestions.push(`Consider refining the query to be more specific than "${foundKeyword}"`);
    }

    const analysis: FeedbackAnalysis = {
      score: calculateFeedbackScore(matchedKeywords),
      keywords: [...matchedKeywords],
      suggestions,
      analysisDate: new Date()
    };

    const analysisTime = performance.now() - startTime;

    output.log(`Analyzed feedback for "${query}" in ${analysisTime.toFixed(2)}ms`, {
      score: analysis.score,
      keywords: analysis.keywords
    });

    return analysis;
  } finally {
    process.stdout.write(TERMINAL_CONTROLS.cursorShow);
  }
}

function calculateFeedbackScore(keywords: readonly string[]): number {
  const baseScore = 100;
  const penalty = keywords.length * 10;
  return Math.max(baseScore - penalty, 0);
}

function generateProgressBar({
  current,
  total,
  width = 30,
  label
}: {
  current: number;
  total: number;
  width?: number;
  label: string;
}): string {
  const filled = Math.round((width * current) / total);
  return `${label} [${'█'.repeat(filled)}${' '.repeat(width - filled)}]`;
}

export async function generateFeedback({
  query,
  numQuestions = 3,
  researchGoal = "Understand the user's query",
  depth = 1,
  breadth = 1,
  existingLearnings = []
}: FeedbackOptions): Promise<FeedbackResponse> {
  if (typeof query !== 'string' || query.trim().length < 2) {
    query = 'general research topic';
  } else if (query.trim().split(' ').length < 3) {
    query = `effects of ${query.trim()}`;
  }

  let cacheKey: string;
  try {
    const keyObject: Record<string, unknown> = {
      query,
      numQuestions,
      researchGoal,
      depth,
      breadth,
      existingLearnings
    };

    if (numQuestions === 3) delete keyObject.numQuestions;
    if (researchGoal === "Understand the user's query") delete keyObject.researchGoal;
    if (depth === 1) delete keyObject.depth;
    if (breadth === 1) delete keyObject.breadth;

    const learningsHash =
      existingLearnings.length > 0
        ? String(existingLearnings.reduce((acc, val) => acc + val.charCodeAt(0), 0))
        : '';

    keyObject.learningsHash = learningsHash;
    cacheKey = JSON.stringify(keyObject);
  } catch {
    cacheKey = 'default-feedback-key';
  }

  try {
    const cachedFeedback = feedbackCache.get(cacheKey);
    if (cachedFeedback) {
      output.log('Cache hit:', { key: cacheKey });
      return cachedFeedback;
    }
  } catch {}

  const context = `
Research Goal: ${researchGoal}
Current Depth: ${depth}
Current Breadth: ${breadth}
Existing Learnings: ${existingLearnings.join('\n')}
`;

  const researchResult = await conductResearch(query, depth);
  const analysisText = researchResult?.analysis?.trim() || query;
  const pastFeedback = await analyzePastFeedback(analysisText);

  const geminiPrompt = `${systemPrompt()}\n\n${context}\n\n${pastFeedback}\n\n${feedbackPromptTemplate
    .replace('{{query}}', query)
    .replace('{{numQuestions}}', String(numQuestions))}`;

  let feedbackResponse: FeedbackResponse = {
    analysis: 'Initial feedback response.',
    followUpQuestions: []
  };

  const { width } = getTerminalDimensions();
  let localProgress = 0;
  const totalSteps = 3;

  try {
    output.log(`Generating feedback for query: "${query}"...`);

    try {
      process.stdout.write(
        `${TERMINAL_CONTROLS.cursorHide}${generateProgressBar({
          current: localProgress,
          total: totalSteps,
          width: Math.max(10, width - 20),
          label: 'Feedback'
        })}`
      );
    } catch {}

    const { FeedbackResponseJsonSchema } = await import('./types.js');
    localProgress = 1;

    try {
      process.stdout.write(
        `\r${generateProgressBar({
          current: localProgress,
          total: totalSteps,
          width: Math.max(10, width - 20),
          label: 'Feedback'
        })}`
      );
    } catch {}

    let jsonText = await callGeminiProConfigurable(
      `${systemPrompt()}\n\nReturn ONLY JSON matching the schema.\n${geminiPrompt}`,
      { json: true, schema: FeedbackResponseJsonSchema, tools: [{ googleSearch: {} }] }
    );

    let parsed: unknown;

    try {
      parsed = JSON.parse(jsonText);
      feedbackResponse = FeedbackResponseSchema.parse(parsed);
    } catch {
      jsonText = await callGeminiProConfigurable(
        `${systemPrompt()}\n\nYour previous output did not match the schema. Repair it to valid JSON.\nSchema: ${JSON.stringify(
          FeedbackResponseJsonSchema
        )}\nPrevious:\n${jsonText}`,
        { json: true, schema: FeedbackResponseJsonSchema, tools: [{ googleSearch: {} }] }
      );

      try {
        parsed = JSON.parse(jsonText);
        feedbackResponse = FeedbackResponseSchema.parse(parsed);
      } catch {
        feedbackResponse = {
          analysis: 'Failed to parse feedback response after repair.',
          followUpQuestions: []
        };
      }
    }

    localProgress = 2;

    try {
      process.stdout.write(
        `\r${generateProgressBar({
          current: localProgress,
          total: totalSteps,
          width: Math.max(10, width - 20),
          label: 'Feedback'
        })}`
      );
    } catch {}

    const isValidOutput = validateAcademicOutput(feedbackResponse.analysis || '');
    if (!isValidOutput) {
      return {
        analysis: 'Unable to generate valid feedback',
        followUpQuestions: []
      };
    }

    try {
      feedbackCache.set(cacheKey, feedbackResponse);
    } catch {}
  } catch {
    feedbackResponse = {
      analysis: 'Error generating feedback. Please check API key and logs.',
      followUpQuestions: []
    };
  } finally {
    try {
      process.stdout.write(
        `\r${generateProgressBar({
          current: totalSteps,
          total: totalSteps,
          width: Math.max(10, width - 20),
          label: 'Feedback'
        })}${TERMINAL_CONTROLS.cursorShow}\n`
      );
    } catch {
      process.stdout.write(TERMINAL_CONTROLS.cursorShow);
    }
  }

  return feedbackResponse;
}
