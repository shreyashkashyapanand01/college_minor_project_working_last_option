import pino from 'pino';

const level = (process.env.LOG_LEVEL || 'info') as pino.LevelWithSilent;
const pretty = (process.env.LOG_PRETTY || 'false').toLowerCase() === 'true';

export const logger = pino({
  level,
  ...(pretty
    ? {
        transport: {
          target: 'pino-pretty',
          options: {
            colorize: true,
            translateTime: 'SYS:standard',
            singleLine: false,
          },
        },
      }
    : {}),
});

export function redactIfNeeded<T>(obj: T): T | string {
  const redacted = (process.env.PROGRESS_REDACT_BODIES || 'false').toLowerCase() === 'true';
  if (!redacted) {
    return obj;
  }
  try {
    return '[REDACTED]';
  } catch {
    return '[REDACTED]';
  }
}
