// .eslintrc.js — Legacy ESLint config (ESLint <9 / CommonJS projects)
// Copy this to your project root and customize.
// For ESLint 9+, use clare/templates/linting/eslint.config.js instead.
//
// Install: npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin

// @ts-check
/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  env: {
    node: true,
    es2022: true,
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: './tsconfig.json',
  },
  plugins: ['@typescript-eslint'],
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:@typescript-eslint/recommended-requiring-type-checking',
  ],

  // ── Global rules ───────────────────────────────────────────────────────────
  rules: {
    // CLARE: [C] Constrained — enforced, not suggested

    // No debug output in production
    'no-console': 'error',
    'no-debugger': 'error',

    // TypeScript
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    '@typescript-eslint/no-floating-promises': 'error',

    // Security
    'no-eval': 'error',
    'no-implied-eval': 'error',

    // Code quality
    'prefer-const': 'error',
    'no-var': 'error',

    // Module boundaries (CLARE: [L] Limited)
    // Derive patterns from clare/autonomy.yml
    'no-restricted-imports': [
      'error',
      {
        patterns: [
          // ── ADD YOUR MODULE BOUNDARY RULES ───────────────────────────────
          // Example:
          // {
          //   group: ['*/payment/*'],
          //   message: 'Use the payment service layer instead of importing from payment directly',
          // },
        ],
      },
    ],

    // ── ADD YOUR PROJECT-SPECIFIC RULES BELOW ──────────────────────────────
  },

  overrides: [
    // Test files — relax some rules
    {
      files: ['**/*.test.ts', '**/*.test.js', '**/*.spec.ts', '**/*.spec.js', 'tests/**/*'],
      rules: {
        'no-console': 'off',
        '@typescript-eslint/no-explicit-any': 'off',
        '@typescript-eslint/no-non-null-assertion': 'off',
      },
    },
    // Script files — allow console
    {
      files: ['scripts/**/*'],
      rules: {
        'no-console': 'off',
      },
    },
    // JavaScript files — no TypeScript rules
    {
      files: ['**/*.js', '**/*.cjs', '**/*.mjs'],
      extends: ['plugin:@typescript-eslint/disable-type-checked'],
    },
  ],

  // Ignore generated and build output
  ignorePatterns: [
    'node_modules/',
    'dist/',
    'build/',
    'src/generated/',  // UPDATE: your generated output dirs
    'coverage/',
    '*.min.js',
  ],
};
