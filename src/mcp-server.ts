import { config } from 'dotenv';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { research, writeFinalReport, type ResearchProgress, type ResearchOptions } from "./deep-research.js";
import { LRUCache } from 'lru-cache';
import { logger } from './logger.js';


// Get the directory name of the current module
const __dirname = fileURLToPath(new URL('.', import.meta.url));

// Load environment variables from .env.local
config({ path: resolve(__dirname, '../.env.local') });

// Log environment variables for debugging (excluding sensitive values)
logger.info({ env: {
  hasGeminiKey: !!process.env.GEMINI_API_KEY,
}}, 'Environment check');

// Change the interface name in mcp-server.ts to avoid conflict
interface MCPResearchResult {
    content: { type: "text"; text: string; }[];
    metadata: {
        learnings: string[];
        visitedUrls: string[];
        stats: {
            totalLearnings: number;
            totalSources: number;
        };
    };
    [key: string]: unknown;
}

// Update cache definition with TTL aligned to provider
const MCP_CACHE_TTL_MS = Math.max(1000, Math.min(86_400_000, parseInt(process.env.PROVIDER_CACHE_TTL_MS || '600000', 10)));
const deepResearchCache = new LRUCache<string, MCPResearchResult>({
  max: 50,
  ttl: MCP_CACHE_TTL_MS,
});

function hashKey(obj: unknown): string {
  try {
    return createHash('sha256').update(JSON.stringify(obj)).digest('hex');
  } catch {
    return String(obj);
  }
}

const server = new McpServer({
  name: "deep-research",
  version: "1.0.0"
});

// Define the deep research tool (modern API)
server.registerTool(
  "deepResearch.run",
  {
    title: "Deep Research",
    description: "Gemini-only deep research pipeline (Google Search grounding + URL context).",
    inputSchema: {
      query: z.string().min(1).describe("The research query to investigate"),
      depth: z.number().min(1).max(5).optional().describe("How deep to go in the research tree (1-5)"),
      breadth: z.number().min(1).max(5).optional().describe("How broad to make each research level (1-5)"),
      existingLearnings: z.array(z.string()).optional().describe("Optional learnings to build upon"),
      goal: z.string().optional().describe("Optional goal/brief to steer synthesis"),
      flags: z.object({ grounding: z.boolean().optional(), urlContext: z.boolean().optional() }).optional(),
    }
  },
  async ({ query, depth, breadth, existingLearnings = [] }): Promise<MCPResearchResult> => {
    // 1. Create cache key
    const cacheKey = hashKey({ query, depth, breadth, existingLearnings });

    // 2. Check cache
    const cachedResult = deepResearchCache.get(cacheKey);
    if (cachedResult) {
      logger.info({ key: cacheKey.slice(0,8), query }, '[mcp-cache] HIT');
      return cachedResult;
    } else {
      logger.info({ key: cacheKey.slice(0,8), query }, '[mcp-cache] MISS');
    }

    try {
      logger.info({ query }, 'Starting research');
      const result = await research({
        query,
        depth: depth ?? 3,
        breadth: breadth ?? 3,
        existingLearnings: existingLearnings,
        onProgress: (progress: ResearchProgress) => {
          // Basic progress logging; MCP notifications can be added when client expects them
          const depthPct = progress.totalDepth > 0 ? (progress.totalDepth - progress.currentDepth) / progress.totalDepth : 0;
          const breadthPct = progress.totalBreadth > 0 ? (progress.totalBreadth - progress.currentBreadth) / progress.totalBreadth : 0;
          const queriesPct = progress.totalQueries > 0 ? progress.completedQueries / progress.totalQueries : 0;
          const overall = Math.round(((depthPct + breadthPct + queriesPct) / 3) * 100);
          logger.info({ overall, progress }, 'Progress update');
        }
      } as ResearchOptions);

      logger.info({ query }, 'Research completed, generating report...');

      // Generate final report
      const report = await writeFinalReport({
        prompt: query,
        learnings: result.learnings,
        visitedUrls: result.visitedUrls
      });

      logger.info({ query }, 'Report generated successfully');

      const finalResult: MCPResearchResult = {
        content: [
          {
            type: "text",
            text: report
          }
        ],
        metadata: {
          learnings: result.learnings,
          visitedUrls: result.visitedUrls,
          stats: {
            totalLearnings: result.learnings.length,
            totalSources: result.visitedUrls.length
          }
        }
      };

      // 4. Store result in cache
      deepResearchCache.set(cacheKey, finalResult);
      logger.info({ query }, 'Stored result in cache');

      return finalResult;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      logger.error({ err: errorMessage }, 'Error during deep research');
      return {
        content: [{ type: "text", text: `Error during deep research: ${errorMessage}` }],
        metadata: { learnings: [], visitedUrls: [], stats: { totalLearnings: 0, totalSources: 0 } }
      } as MCPResearchResult;
    }
  }
);

// Expose capabilities as a simple resource (Gemini-only flags)
server.registerResource(
  "capabilities",
  "mcp://capabilities",
  {
    title: "Server Capabilities",
    description: "Feature flags and environment info",
    mimeType: "application/json"
  },
  async (uri) => ({
    contents: [{
      uri: uri.href,
      text: JSON.stringify({
        name: "deep-research",
        version: "1.0.0",
        geminiModel: process.env.GEMINI_MODEL || "gemini-2.5-flash",
        googleSearchEnabled: (process.env.ENABLE_GEMINI_GOOGLE_SEARCH || 'true').toLowerCase() === 'true',
        urlContextEnabled: (process.env.ENABLE_URL_CONTEXT || 'true').toLowerCase() === 'true',
        functionsEnabled: (process.env.ENABLE_GEMINI_FUNCTIONS || 'false').toLowerCase() === 'true',
        codeExecEnabled: (process.env.ENABLE_GEMINI_CODE_EXECUTION || 'false').toLowerCase() === 'true',
        providerCacheTtlMs: MCP_CACHE_TTL_MS,
      })
    }]
  })
);

// Start the MCP server
const transport = new StdioServerTransport();
server.connect(transport)
  .then(() => { logger.info('MCP server running'); })
  .catch((err: Error) => { logger.error({ err }, 'MCP server error'); });
