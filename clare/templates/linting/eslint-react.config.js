// eslint-react.config.js — ESLint flat config for React / TSX projects
// Copy this to your project root as eslint.config.js and customize.
// Requires ESLint 9+ and the plugins listed below.
//
// Install:
//   npm install -D eslint @eslint/js globals typescript-eslint \
//     eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-jsx-a11y
//
// CLARE Principle: [C] Constrained — component size limits are enforced,
// not left to code review. AI self-corrects by splitting components.

// @ts-check
import eslint from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';
import react from 'eslint-plugin-react';
import reactHooks from 'eslint-plugin-react-hooks';
import jsxA11y from 'eslint-plugin-jsx-a11y';

export default tseslint.config(
  // ── Base rules ─────────────────────────────────────────────────────────────
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,

  // ── Language options ───────────────────────────────────────────────────────
  {
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
      },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
        ecmaFeatures: { jsx: true },
      },
    },
    settings: {
      react: { version: 'detect' },
    },
  },

  // ── React plugins ─────────────────────────────────────────────────────────
  {
    plugins: {
      react,
      'react-hooks': reactHooks,
      'jsx-a11y': jsxA11y,
    },
    rules: {
      // React best practices
      ...react.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      ...jsxA11y.configs.recommended.rules,

      'react/react-in-jsx-scope': 'off',       // Not needed with React 17+ JSX transform
      'react/prop-types': 'off',               // TypeScript handles prop validation
      'react/display-name': 'warn',

      // Hook rules — these catch real bugs
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
    },
  },

  // ── Component size limits (CLARE: [C] Constrained) ────────────────────────
  // These are the rules that catch the "one giant component" problem.
  // AI hits these limits → verify-ci.sh fails → AI splits the component.
  //
  // Tune the numbers for your project. Start lenient, tighten over time.
  {
    files: ['**/*.tsx', '**/*.jsx'],
    rules: {
      // ── File size: max lines per file ──────────────────────────────────
      // A component file over 250 lines almost always needs splitting.
      // This catches the "everything in App.tsx" pattern.
      'max-lines': ['error', {
        max: 250,
        skipBlankLines: true,
        skipComments: true,
      }],

      // ── Function/component size: max lines per function ────────────────
      // A component over 80 lines of logic is doing too much.
      // Encourages extracting hooks, child components, and utilities.
      'max-lines-per-function': ['warn', {
        max: 80,
        skipBlankLines: true,
        skipComments: true,
        IIFEs: true,            // Don't count wrapper IIFEs
      }],

      // ── Nesting depth ──────────────────────────────────────────────────
      // Deep nesting in JSX = unreadable. Extract child components.
      'max-depth': ['warn', 4],

      // ── Parameters / props ─────────────────────────────────────────────
      // Too many params = the component is doing too much or needs
      // a context/compound-component pattern.
      'max-params': ['warn', 5],
    },
  },

  // ── Non-component TypeScript files — different (looser) size limits ────────
  {
    files: ['**/*.ts'],
    ignores: ['**/*.tsx'],
    rules: {
      'max-lines': ['warn', {
        max: 400,
        skipBlankLines: true,
        skipComments: true,
      }],
      'max-lines-per-function': ['warn', {
        max: 50,
        skipBlankLines: true,
        skipComments: true,
        IIFEs: true,
      }],
    },
  },

  // ── Global rules ───────────────────────────────────────────────────────────
  {
    rules: {
      // CLARE: [C] Constrained
      'no-console': 'error',
      'no-debugger': 'error',

      // TypeScript strictness
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/no-floating-promises': 'error',

      // Security
      'no-eval': 'error',
      'no-implied-eval': 'error',

      // Code quality
      'prefer-const': 'error',
      'no-var': 'error',

      // ── ADD YOUR PROJECT-SPECIFIC RULES BELOW ──────────────────────────
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

            // Example: Prevent importing from domain core directly
            // {
            //   group: ['*/domain/*'],
            //   message: 'Import domain types from the public API, not the domain core',
            // },

            // Example: Prevent cross-feature imports (feature slices)
            // {
            //   group: ['*/features/*/internal/*'],
            //   message: 'Import from the feature public API (index.ts), not internal modules',
            // },
          ],
        },
      ],
    },
  },

  // ── Test file overrides ────────────────────────────────────────────────────
  {
    files: ['**/*.test.ts', '**/*.test.tsx', '**/*.spec.ts', '**/*.spec.tsx', 'tests/**'],
    rules: {
      'no-console': 'off',
      'max-lines': 'off',                         // Tests can be long
      'max-lines-per-function': 'off',             // describe/it blocks get lengthy
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },

  // ── Storybook overrides ────────────────────────────────────────────────────
  {
    files: ['**/*.stories.tsx', '**/*.stories.ts'],
    rules: {
      'max-lines': 'off',
      'max-lines-per-function': 'off',
    },
  },

  // ── Generated file ignores ────────────────────────────────────────────────
  {
    ignores: [
      'node_modules/**',
      'dist/**',
      'build/**',
      '.next/**',
      'src/generated/**',    // UPDATE: your generated output dirs
      'coverage/**',
      '*.min.js',
    ],
  }
);
