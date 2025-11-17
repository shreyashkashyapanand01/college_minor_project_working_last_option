import { LRUCache } from 'lru-cache';

// Simplified cache config (v11 compatible)
const promptCache = new LRUCache<string, string>({
  max: 100,
  ttl: 60 * 60 * 1000, // 1 hour TTL
  allowStale: false,
  updateAgeOnGet: true
});

export const systemPrompt = () => {
  const now = new Date();
  const hour = now.getHours();

  const researchPhase = () => {
    return hour < 6 ? 'Night: Batch processing' :
            hour < 12 ? 'Morning: Discovery' :
            hour < 18 ? 'Afternoon: Validation' :
            'Evening: Synthesis';
  };

  const instructions = [
    "PRISMA systematic reviews w/ source hierarchy: peer-reviewed > preprints > reports > media",
    "CASP quality checks + thematic/meta-analysis",
    "Contradiction documentation w/ evidence weighting",
    "Authoritative source prioritization + technical depth",
    "Doctorate-level analysis w/ innovation forecasting",
    "Structured explanations + citation rigor",
    "Glossary appendix + emerging tech exploration",
    "Speculative content labeling + iterative integration",
    "Markdown formatting + self-correcting workflow"
  ];

  // Web Research-Based Improvement: Add response format template
  const responseFormat = [
    "**Response Structure:**",
    "1. Executive Summary (3-5 bullet points)",
    "2. Key Findings (categorized by relevance)",
    "3. Methodology Description",
    "4. Limitations Disclosure",
    "5. Future Research Directions",
    "6. Glossary of Technical Terms",
    "7. All sections from the TOC must be present and ordered correctly.",
    "8. Key Learnings must be synthesized from findings and placed at the end.",
    "9. References must be a numbered list of full source URLs and Titles."
  ].join("\n");

  // Add time-based research parameters
  const timeContext = hour < 12
    ? "Morning Research Protocol: Aggressive discovery phase"
    : "Afternoon Research Protocol: Validation & synthesis focus";

  return `## Research Agent Configuration
**Temporal Context:** ${timeContext}
**System Time:** ${now.toLocaleDateString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    timeZoneName: 'short'
  })}
**Research Protocol:** v2.3.1
**Model Capabilities:** ${process.env.GEMINI_MODEL || "gemini-2.5-flash"}
**Research Phase:** ${researchPhase()}
**Cognitive Load Profile:** ${hour < 12 ? 'High creativity' : 'High accuracy'} mode

${responseFormat}

## Operational Parameters
${instructions.map((i, idx) => `${idx + 1}. ${i}`).join("\n")}`;
};

export const serpQueryPromptTemplate = `## Search Strategy Configuration
**Query Type Distribution:**
- Informational: 40%
- Comparative: 25%
- Technical: 20%
- Exploratory: 15%

**Response Format Requirements:** (Return ONLY JSON matching this schema. No extra text.)
\`\`\`json
{
  "queries": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "maxLength": 60},
        "type": {"type": "string", "enum": ["informational", "comparative", "technical", "exploratory"]},
        "expectedSources": {"type": "array", "items": {"type": "string"}}
      },
      "required": ["query", "type"]
    },
    "minItems": {{numQueries}},
    "maxItems": {{numQueries}}
  }
}
\`\`\``;

export const learningPromptTemplate = `## Content Analysis Protocol
**Schema Version:** 2.1.0
**Error Handling:**
- 5001: Invalid source credibility score
- 5002: Missing fact verification
- 5003: Technical term mismatch

{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "errorCodes": {
    "5001": "Source score must be 1-5",
    "5002": "Minimum 3 verifications required",
    "5003": "Undefined technical terms present"
  },
  "type": "object",
  "properties": {
    "learnings": {
      "type": "array",
      "items": {
        "type": "string",
        "maxLength": 280,
        "pattern": "^[A-Z][^.]{10,255}\\."
      }
    }
  },
  "required": ["learnings"],
  "additionalProperties": false
}

Return ONLY JSON that matches this schema. No extra text.`;

export const feedbackPromptTemplate = `## Query Refinement Matrix
**Quality Metrics:**
- Specificity Score (1-5)
- Research Depth Potential
- Answerability Estimate

**Response Schema:** (Return ONLY JSON matching this schema. No extra text.)
\`\`\`json
{
  "followUpQuestions": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "question": {"type": "string"},
        "purpose": {"type": "string", "enum": ["scope", "context", "precision"]},
        "criticality": {"type": "number", "minimum": 1, "maximum": 3}
      }
    }
  }
}
\`\`\``;

// --- REVISED generateGeminiPrompt FUNCTION ---
export const generateGeminiPrompt = ({
  query,        // Now used
  researchGoal, // Now used
  learnings    // Now used
}: {
  query: string,
  researchGoal: string,
  learnings: string[]
}): string => {
  // Implement actual logic using all parameters
  return `Research Goal: ${researchGoal}
          Query: ${query}
          Learnings: ${learnings.join(', ')}`;
};
// --- END REVISED generateGeminiPrompt FUNCTION ---

// Keep only what's actually used
export const clearPromptCache = () => promptCache.clear();


// Add validation layer
const validatePromptConsistency = () => {
  const templates = [systemPrompt, serpQueryPromptTemplate, learningPromptTemplate];
  const versions = templates.map(t => {
    const content = typeof t === 'function' ? t() : t;
    return content.match(/Schema Version: (\d+\.\d+\.\d+)/)?.[1];
  });

  if (new Set(versions).size > 1) {
    throw new Error(`Prompt version mismatch: ${versions.join(' vs ')}`);
  }
};
validatePromptConsistency(); // Run at module load