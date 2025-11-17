import type { ResearchProgress } from './deep-research.js';
import kleur from 'kleur';

// Constants for ANSI escape codes to improve readability
const ESC = '\x1B[';
const CLEAR_LINE = `${ESC}2K`; // Clear the entire line
const CURSOR_TO_LINE = (line: number) => `${ESC}${line};1H`; // Move cursor to specified line and column 1
const CURSOR_UP = (lines: number) => `${ESC}${lines}A`; // Move cursor up by specified lines
const CURSOR_DOWN = (lines: number) => `${ESC}${lines}B`; // Move cursor down by specified lines

const DEFAULT_NUMBER_OF_PROGRESS_LINES = 4; // Constant for the number of progress lines

export class ProgressManager {
  private lastProgress: ResearchProgress | undefined;
  private numberOfProgressLines: number;
  private initialized = false;
  private terminalHeight: number;
  // Maintain small history of query completion to compute moving-average ETA
  private rateWindow: Array<{ t: number; completed: number }> = [];
  private readonly maxWindow = 20;

  constructor(numberOfProgressLines: number = DEFAULT_NUMBER_OF_PROGRESS_LINES) {
    this.numberOfProgressLines = numberOfProgressLines;
    this.terminalHeight = process.stdout.rows || 24;

    // Initialize terminal with empty lines for progress display
    try {
      process.stdout.write('\n'.repeat(this.numberOfProgressLines));
      this.initialized = true;
    } catch (error) {
      console.error("Failed to initialize ProgressManager:", error);
      this.initialized = false;
    }

    // Update terminal height on resize
    process.stdout.on('resize', () => {
      this.terminalHeight = process.stdout.rows || 24;
    });
  }

  /**
   * Draws a single progress bar string.
   *
   * @param label - The label for the progress bar (e.g., 'Depth:').
   * @param value - The current progress value.
   * @param total - The total value for 100% progress.
   * @param char - The character to use for the filled part of the bar.
   * @returns The formatted progress bar string.
   */
  private drawProgressBar(
    label: string,
    value: number,
    total: number,
    char = '='
  ): string {
    const width = process.stdout.columns ? Math.min(30, process.stdout.columns - 20) : 30;
    const percent = total > 0 ? (value / total) * 100 : 0;
    const filled = Math.round((width * percent) / 100);
    const empty = width - filled;

    // Choose color based on percent
    const colorize = percent >= 66 ? kleur.green : percent >= 33 ? kleur.yellow : kleur.red;
    const bar = `${colorize(char.repeat(filled))}${kleur.dim(' '.repeat(empty))}`;
    const pctStr = `${Math.round(percent)}%`;
    const compact = (process.env.PROGRESS_COMPACT || 'false').toLowerCase() === 'true';
    const labelStr = compact ? '' : `${kleur.cyan(label)} `;
    return `${labelStr}[${bar}] ${colorize(pctStr)}`;
  }

  /**
   * Updates the progress display in the terminal.
   *
   * @param progress - An object containing the current research progress.
   */
  updateProgress(progress: ResearchProgress): void { // Added return type void
    if (!this.initialized) {
      return;
    }

    // Skip redraw if no meaningful change since last draw
    if (this.lastProgress &&
        this.lastProgress.completedQueries === progress.completedQueries &&
        this.lastProgress.currentDepth === progress.currentDepth &&
        this.lastProgress.currentBreadth === progress.currentBreadth &&
        this.lastProgress.currentQuery === progress.currentQuery) {
      return;
    }
    this.lastProgress = progress;

    // Update rate window using completedQueries
    const now = Date.now();
    this.rateWindow.push({ t: now, completed: progress.completedQueries });
    if (this.rateWindow.length > this.maxWindow) {
      this.rateWindow.shift();
    }

    // Compute moving-average rate (queries/sec) and ETA for remaining queries
    let etaStr = 'ETA: —';
    if (this.rateWindow.length >= 2 && progress.totalQueries > 0) {
      const first = this.rateWindow[0];
      if (!first) {
        // keep default ETA
      } else {
      const lastIndex = this.rateWindow.length - 1;
      const last = this.rateWindow[lastIndex];
      if (!last) {
        // Should not happen, but guard for type safety
        // Leave ETA as unknown in this edge case
      } else {
        const dCompleted = Math.max(0, last.completed - first.completed);
        const dTimeSec = Math.max(0.001, (last.t - first.t) / 1000);
        const rate = dCompleted / dTimeSec; // queries per second
        const remaining = Math.max(0, progress.totalQueries - progress.completedQueries);
        const etaSec = rate > 0 ? Math.round(remaining / rate) : 0;
        const mm = Math.floor(etaSec / 60);
        const ss = etaSec % 60;
        etaStr = kleur.magenta(`ETA: ${mm}m ${ss}s`);
      }
      }
    }

    // Determine the starting line for progress bars to position them at the bottom of the terminal
    const progressStartLine = this.terminalHeight - this.numberOfProgressLines;

    // Generate progress bar lines
    const lines: string[] = [
      this.drawProgressBar(
        'Depth:   ',
        progress.totalDepth - progress.currentDepth,
        progress.totalDepth,
        '█'
      ),
      this.drawProgressBar(
        'Breadth: ',
        progress.totalBreadth - progress.currentBreadth,
        progress.totalBreadth,
        '█'
      ),
      this.drawProgressBar(
        'Queries: ',
        progress.completedQueries,
        progress.totalQueries,
        '█'
      ),
      // Overall percent as average of the three bars (simple heuristic) plus ETA
      (() => {
        const depthPct = progress.totalDepth > 0 ? (progress.totalDepth - progress.currentDepth) / progress.totalDepth : 0;
        const breadthPct = progress.totalBreadth > 0 ? (progress.totalBreadth - progress.currentBreadth) / progress.totalBreadth : 0;
        const queriesPct = progress.totalQueries > 0 ? progress.completedQueries / progress.totalQueries : 0;
        const overall = Math.round(((depthPct + breadthPct + queriesPct) / 3) * 100);
        return `Overall: ${overall}%  ${etaStr}`;
      })(),
    ];

    // Add current query line if it exists in the progress data
    if (progress.currentQuery) {
      lines.push(`Current:  ${progress.currentQuery}`);
    }

    try {
      // Reposition cursor relative, then absolute, to be resilient across terminals
      process.stdout.write(`${CURSOR_UP(this.numberOfProgressLines)}`);
      process.stdout.write(`${CURSOR_TO_LINE(progressStartLine)}`);

      for (let i = 0; i < this.numberOfProgressLines; i++) {
        process.stdout.write(`${CLEAR_LINE}\n`);
      }

      // Move cursor back to the start of the progress area
      process.stdout.write(`${CURSOR_TO_LINE(progressStartLine)}`);

      // Output all progress lines, joined by newlines, to the terminal
      process.stdout.write(`${lines.join('\n')}\n`);
    } catch (error) {
      console.error("Failed to update progress:", error);
    }
  }

  /**
   * Stops the progress display and moves the cursor below the progress area.
   */
  stop(): void { // Added return type void
    if (!this.initialized) {
      return;
    }

    try {
      // Move cursor down past the progress area, ensuring subsequent output is below the progress bars
      process.stdout.write(`${CURSOR_DOWN(this.numberOfProgressLines)}\n`); // Move cursor down
    } catch (error) {
      console.error("Failed to stop ProgressManager:", error);
    }
  }
}
