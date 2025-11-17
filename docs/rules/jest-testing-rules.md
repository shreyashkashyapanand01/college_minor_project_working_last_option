# Testing Rules (Jest + TypeScript ESM)

Last reviewed: 2025-08-11

Project runs Jest in ESM mode with ts-jest.

## Rules
- **No network** in unit tests; mock providers (Gemini, Exa, Firecrawl) and fs.
- **ESM everywhere**: Use `ts-jest` ESM preset; update `package.json` test script to `jest`.
- **Isolate**: One behavior per test; avoid brittle string snapshots of model output.
- **Provider-free splitters**: Tests for `RecursiveCharacterTextSplitter` and `TiktokenTextSplitter` must not hit embeddings.

## Sample `jest.config.ts`
```ts
import type { JestConfigWithTsJest } from 'ts-jest';

const config: JestConfigWithTsJest = {
  preset: 'ts-jest/presets/default-esm',
  testEnvironment: 'node',
  transform: { '^.+\\.ts$': [ 'ts-jest', { useESM: true } ] },
  extensionsToTreatAsEsm: ['.ts'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'mjs', 'cjs', 'json', 'node'],
  testPathIgnorePatterns: ['/node_modules/', '/dist/'],
  moduleNameMapper: {
    '^(.*)\\.js$': '$1', // allow ESM TS path imports without .js at runtime
  },
};
export default config;
```

## References
- Jest ESM: https://jestjs.io/docs/ecmascript-modules
- ts-jest ESM: https://kulshekhar.github.io/ts-jest/docs/guides/esm-support
