export function sanitizeReportContent(content: string): string {
  return content
    .replace(/Thinking process:.*?\n\n/gs, '')
    .replace(/Outline:.*?\n\n/gs, '')
    .replace(/Step \d+:.*?\n/g, '')
    .replace(/\[Internal Note:.*?\]/g, '');
}
