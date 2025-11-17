// New shared utilities file
export const TERMINAL_CONTROLS = {
  clearLine: '\x1B[2K',
  savePos: '\x1B[s',
  restorePos: '\x1B[u',
  cursorUp: (n: number) => `\x1B[${n}A`,
  cursorToLine: (n: number) => `\x1B[${n};1H`,
  cursorHide: '\x1B[?25l',
  cursorShow: '\x1B[?25h'
} as const;

export type ProgressBarParams = {
  current: number;
  total: number;
  width?: number;
  label: string;
  template?: string;
};

export function generateProgressBar({
  current,
  total,
  width = 30,
  label,
  template = '{label} [{bar}] {percentage}%'
}: ProgressBarParams): string {
  const filled = Math.round((width * current) / total);
  const percentage = Math.round((current / total) * 100);
  return template
    .replace('{label}', label)
    .replace('{bar}', '█'.repeat(filled) + ' '.repeat(width - filled))
    .replace('{percentage}', percentage.toString());
}

export function getTerminalDimensions() {
  return {
    width: process.stdout.columns || 80,
    height: process.stdout.rows || 24
  };
}
