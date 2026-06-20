// eslint.config.js — Modern ESLint flat config
// Copy this to your project root and customize.
// ⚠ Requires ESLint 9+ (flat config format) and appropriate plugins.
//
// Install: npm install -D eslint @eslint/js globals typescript-eslint

// @ts-check
import eslint from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  // ── Base rules ─────────────────────────────────────────────────────────────
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,

  // ── Language options ───────────────────────────────────────────────────────
  {
    languageOptions: {
      globals: {
        ...globals.node,
        ...globals.browser, // Remove if not a browser project
      },
      parserOptions: {
        projectService: true,            // Enable type-aware linting
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },

  // ── Global rules ───────────────────────────────────────────────────────────
  {
    rules: {
      // CLARE: [C] Constrained — rules that should never be violated

      // No debug output in production code
      'no-console': 'error',          // AI removes console.log before marking work complete
      'no-debugger': 'error',

      // TypeScript strictness
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': 'error',
      '@typescript-eslint/no-floating-promises': 'error',  // Catch unhandled promises

      // Security
      'no-eval': 'error',
      'no-implied-eval': 'error',

      // Code quality
      'no-return-await': 'error',
      'prefer-const': 'error',
      'no-var': 'error',

      // ── ADD YOUR PROJECT-SPECIFIC RULES BELOW ──────────────────────────────

      // Example: Require explicit return types on public functions
      // '@typescript-eslint/explicit-function-return-type': ['warn', { allowExpressions: true }],

      // Example: Enforce import ordering for readability in large modules
      // 'sort-imports': ['warn', { ignoreDeclarationSort: true }],

      // Example: Enforce consistent error handling
      // '@typescript-eslint/only-throw-error': 'error',

      // Example: No process.exit() in library code
      // 'no-process-exit': 'error',
    },
  },

  // ── Module boundary rules (CLARE: [L] Limited) ────────────────────────────
  {
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            // ── ADD YOUR MODULE BOUNDARY RULES ─────────────────────────────
            // Derive these from clare/autonomy.yml boundaries.
            // Why: boundary rules prevent accidental imports that bypass
            // review layers and silently violate autonomy decisions.

            // Example: Prevent direct imports from payment module (use service layer)
            // {
            //   group: ['*/payment/*', '*/payment'],
            //   message: 'Import payment functionality through the service layer: import from "*/services/payment"',
            // },

            // Example: Prevent importing from humans-only domain core
            // {
            //   group: ['*/domain/*'],
            //   message: 'Import domain types from the public API layer, not the domain core directly',
            // },
          ],
        },
      ],
    },
  },

  // ── Test file overrides ────────────────────────────────────────────────────
  {
    files: ['**/*.test.ts', '**/*.test.js', '**/*.spec.ts', '**/*.spec.js', 'tests/**'],
    rules: {
      'no-console': 'off',     // Allow console in tests
      '@typescript-eslint/no-explicit-any': 'off', // More flexible in tests
    },
  },

  // ── Script file overrides ──────────────────────────────────────────────────
  {
    files: ['scripts/**'],
    rules: {
      'no-console': 'off',     // Scripts use console for output
    },
  },

  // ── Generated file ignores ────────────────────────────────────────────────
  {
    ignores: [
      'node_modules/**',
      'dist/**',
      'build/**',
      'src/generated/**',    // UPDATE: your generated output dirs
      'coverage/**',
      '*.min.js',
    ],
  }
);
