import { Exa } from 'exa-js';

export type SearchHit = { url: string; snippet: string };

export interface SearchProvider {
  readonly name: string;
  available(): boolean;
  search(query: string): Promise<SearchHit[]>;
}

// Exa-only provider module. Firecrawl remains in `src/deep-research.ts`.
// Grounding is handled by Gemini configuration in `src/ai/providers.ts`.

export class ExaProvider implements SearchProvider {
  readonly name = 'exa';
  private readonly apiKey = process.env.EXA_API_KEY || '';
  private exa?: Exa;

  available(): boolean {
    return !!this.apiKey;
  }

  async search(query: string): Promise<SearchHit[]> {
    if (!this.available()) {
      return [];
    }
    try {
      if (!this.exa) {
        this.exa = new Exa(this.apiKey);
      }
      // Basic keyword search; adjust parameters as needed.
      const res = await this.exa.search({
        query,
        type: 'keyword',
        numResults: 10,
        useAutoprompt: true,
      } as any);
      const results = (res?.results || []) as Array<{ url?: string; snippet?: string; title?: string }>;
      return results
        .filter(r => typeof r.url === 'string')
        .map(r => ({ url: String(r.url), snippet: r.snippet || r.title || '' }));
    } catch {
      return [];
    }
  }
}

export function makeExaProviderIfAvailable(): ExaProvider | null {
  const exa = new ExaProvider();
  return exa.available() ? exa : null;
}
