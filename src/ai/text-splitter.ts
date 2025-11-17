// Pure text splitting utilities (no Gemini here). Providers own all Gemini logic.

import { type Tiktoken, getEncoding, type TiktokenEncoding } from 'js-tiktoken';

export interface TextSplitterParams {
  chunkSize: number;
  chunkOverlap: number;
  separators?: string[];
}

// Synchronous splitter used by tests in src/ai/text-splitter.test.ts
export class RecursiveCharacterTextSplitter implements TextSplitterParams {
  chunkSize: number;
  chunkOverlap: number;
  separators: string[];

  constructor(params?: Partial<TextSplitterParams>) {
    this.chunkSize = params?.chunkSize ?? 50;
    this.chunkOverlap = params?.chunkOverlap ?? 10;
    // Prioritize newlines → sentences → commas → spaces → chars
    this.separators = params?.separators ?? ['\n\n', '\n', '. ', '.', ', ', ',', ' ', ''];
    if (this.chunkOverlap >= this.chunkSize) {
      throw new Error('Cannot have chunkOverlap >= chunkSize');
    }
  }

  // Produce compact chunks, removing separator punctuation between merged parts.
  splitText(text: string): string[] {
    if (!text) {
      return [];
    }

    const target = this.chunkSize;
    const step = Math.max(1, target - this.chunkOverlap);
    const chunks: string[] = [];

    const splitRecursive = (t: string, sepIdx = 0) => {
      const trimmed = t.trim();
      if (!trimmed) {
        return;
      }
      if (trimmed.length <= target) {
        chunks.push(trimmed);
        return;
      }
      if (sepIdx >= this.separators.length) {
        // Hard cut with overlap when no separators remain
        for (let i = 0; i < trimmed.length; i += step) {
          const piece = trimmed.slice(i, Math.min(i + target, trimmed.length)).trim();
          if (piece) {
            chunks.push(piece);
          }
        }
        return;
      }

      const sep = this.separators[sepIdx] ?? '';
      if (sep === '') {
        // No separator: fallback to hard cuts
        for (let i = 0; i < trimmed.length; i += step) {
          const piece = trimmed.slice(i, Math.min(i + target, trimmed.length)).trim();
          if (piece) {
            chunks.push(piece);
          }
        }
        return;
      }

      // Split by current separator; when merging, use a space to avoid reintroducing punctuation
      const parts = trimmed.split(sep).map(s => s.trim()).filter(Boolean);
      let buffer = '';
      for (const part of parts) {
        const candidate = buffer ? `${buffer} ${part}`.trim() : part;
        if (candidate.length <= target) {
          buffer = candidate;
        } else {
          if (buffer) {
            chunks.push(buffer);
          }
          if (part.length > target) {
            // Recurse deeper for the long piece
            splitRecursive(part, sepIdx + 1);
            buffer = '';
          } else {
            buffer = part;
          }
        }
      }
      if (buffer) {
        chunks.push(buffer);
      }
    };

    splitRecursive(text, 0);
    // Normalize whitespace and filter empties
    return chunks
      .map((c) => c.replace(/\s+/g, ' ').trim())
      .filter(Boolean);
  }
}

// Semantic wrapper that normalizes newlines before splitting.
export class SemanticTextSplitter implements TextSplitterParams {
  chunkSize: number;
  chunkOverlap: number;
  separators?: string[];

  private readonly base: RecursiveCharacterTextSplitter;

  constructor(params?: Partial<TextSplitterParams>) {
    this.chunkSize = params?.chunkSize ?? 50;
    this.chunkOverlap = params?.chunkOverlap ?? 10;
    this.separators = params?.separators;
    this.base = new RecursiveCharacterTextSplitter({
      chunkSize: this.chunkSize,
      chunkOverlap: this.chunkOverlap,
      separators: this.separators,
    });
  }

  splitText(text: string): string[] {
    if (!text) {
      return [];
    }
    const normalized = text.replace(/\r\n?/g, '\n');
    return this.base.splitText(normalized);
  }
}

// Optional: tiktoken-based splitter (not used by current tests)
export class TiktokenTextSplitter implements TextSplitterParams {
  chunkSize: number;
  chunkOverlap: number;
  private tokenizer: Tiktoken;

  constructor(params?: Partial<TextSplitterParams> & { encoding?: TiktokenEncoding }) {
    this.chunkSize = params?.chunkSize ?? 1500;
    this.chunkOverlap = params?.chunkOverlap ?? 200;
    const enc = params?.encoding ?? ('o200k_base' as TiktokenEncoding);
    try {
      this.tokenizer = getEncoding(enc);
    } catch {
      // Fallback to naive tokenizer
      this.tokenizer = {
        encode: (text: string) => Array.from(new TextEncoder().encode(text)),
        decode: (tokens: number[]) => new TextDecoder().decode(Uint8Array.from(tokens)),
        free: () => {}
      } as unknown as Tiktoken;
    }
    if (this.chunkOverlap >= this.chunkSize) {
      throw new Error('Cannot have chunkOverlap >= chunkSize');
    }
  }

  splitText(text: string): string[] {
    if (!text) {
      return [];
    }
    const ids = this.tokenizer.encode(text);
    const out: string[] = [];
    for (let i = 0; i < ids.length; i += Math.max(1, this.chunkSize - this.chunkOverlap)) {
      const slice = ids.slice(i, i + this.chunkSize);
      out.push(this.tokenizer.decode(slice).trim());
    }
    return out.filter(Boolean);
  }
}
