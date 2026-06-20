/**
 * CLARE Architecture Test: Module Boundary Enforcement
 * =====================================================
 * CLARE Principle: [C] Constrained — [L] Limited
 *
 * Ensures code does not import across forbidden module boundaries.
 * Maps directly to the autonomy levels in clare/autonomy.yml:
 *   - supervised code must NOT import from humans-only code directly
 *   - full-autonomy code must NOT import from supervised or humans-only code
 *
 * ADAPT THIS FILE:
 *   1. Update FORBIDDEN_IMPORTS to match your actual module boundaries
 *   2. Update SRC_DIR to your source directory
 *
 * Run with: npx jest tests/architecture/module-boundaries.test.js
 */

const fs = require('fs');
const path = require('path');

// ─── Configuration: UPDATE THESE ─────────────────────────────────────────────

const SRC_DIR = path.join(__dirname, '../../src'); // UPDATE: your source dir

/**
 * Forbidden import rules.
 * Each rule means: files in `from` CANNOT import anything from `to`.
 *
 * Derive these from clare/autonomy.yml — supervised/full-autonomy code
 * should not have direct dependencies on humans-only code.
 */
const FORBIDDEN_IMPORTS = [
  {
    description: 'Utilities must not directly import from business logic',
    from: 'src/utils',
    // Utilities should be pure — they should not depend on application code
    to: ['src/services', 'src/api', 'src/domain'],
  },
  {
    description: 'Generated code must not import from humans-only domain',
    from: 'src/generated',
    to: ['src/domain', 'src/payment', 'src/auth/core'],
  },
  {
    description: 'API layer must not directly import from payment core',
    from: 'src/api',
    to: ['src/payment'],
    // Exception: API routes should go through a service layer
    exception: 'Only src/services/payment may import from src/payment',
  },
  // Add more rules derived from your architecture decisions:
  // {
  //   description: 'Controllers must not import from infrastructure',
  //   from: 'src/controllers',
  //   to: ['src/infrastructure/db', 'src/infrastructure/cache'],
  // },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function getAllFiles(dir, extensions = ['.js', '.ts', '.jsx', '.tsx']) {
  const results = [];
  if (!fs.existsSync(dir)) return results;

  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const entryPath = path.join(dir, entry.name);
    if (entry.isDirectory() && !entry.name.startsWith('.') && entry.name !== 'node_modules') {
      results.push(...getAllFiles(entryPath, extensions));
    } else if (extensions.some((ext) => entry.name.endsWith(ext))) {
      results.push(entryPath);
    }
  }
  return results;
}

function extractImports(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const imports = [];

  // Static imports: import ... from '...'
  const staticMatches = content.matchAll(/from\s+['"]([^'"]+)['"]/g);
  for (const match of staticMatches) {
    imports.push(match[1]);
  }

  // Require calls: require('...')
  const requireMatches = content.matchAll(/require\(['"]([^'"]+)['"]\)/g);
  for (const match of requireMatches) {
    imports.push(match[1]);
  }

  return imports;
}

function resolveImportPath(importPath, fromFile, projectRoot) {
  if (importPath.startsWith('.')) {
    // Relative import — resolve to absolute
    const resolved = path.resolve(path.dirname(fromFile), importPath);
    return path.relative(projectRoot, resolved);
  }
  // Absolute or node_modules import
  return importPath;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('Module Boundary Enforcement', () => {
  const projectRoot = path.join(__dirname, '../..');

  for (const rule of FORBIDDEN_IMPORTS) {
    test(rule.description, () => {
      const fromDir = path.join(projectRoot, rule.from);
      const violations = [];

      const files = getAllFiles(fromDir);

      if (files.length === 0) {
        console.warn(`⚠  No files found in ${rule.from} — is the path correct?`);
        return; // Soft skip
      }

      for (const file of files) {
        const imports = extractImports(file);
        const relativeFile = path.relative(projectRoot, file);

        for (const imp of imports) {
          const resolvedPath = resolveImportPath(imp, file, projectRoot);

          for (const forbiddenPath of rule.to) {
            if (resolvedPath.startsWith(forbiddenPath) || imp.includes(forbiddenPath)) {
              violations.push(`  ${relativeFile}\n    imports: ${imp}\n    violates: cannot import from ${forbiddenPath}`);
            }
          }
        }
      }

      if (violations.length > 0) {
        const note = rule.exception ? `\nNote: ${rule.exception}` : '';
        throw new Error(
          `Module boundary violation: ${rule.description}\n\n` +
          violations.join('\n\n') +
          note
        );
      }
    });
  }

  test('autonomy.yml humans-only paths are not imported by generated code', () => {
    // Read humans-only paths from autonomy.yml using basic grep
    // No yaml parser needed — just check for the pattern
    const autonomyPath = path.join(projectRoot, 'clare/autonomy.yml');
    if (!fs.existsSync(autonomyPath)) {
      console.warn('clare/autonomy.yml not found — skipping boundary check');
      return;
    }

    const autonomyContent = fs.readFileSync(autonomyPath, 'utf8');
    const humansOnlyPaths = [];

    // Parse humans-only entries (simplified — look for level: humans-only followed by path:)
    const blocks = autonomyContent.split(/- path:/);
    for (const block of blocks) {
      if (block.includes('level: humans-only')) {
        const pathMatch = block.match(/["']([^"']+)["']/);
        if (pathMatch && pathMatch[1] !== '*') {
          humansOnlyPaths.push(pathMatch[1].replace(/^\//, ''));
        }
      }
    }

    if (humansOnlyPaths.length === 0) return;

    const generatedDir = path.join(projectRoot, 'src/generated'); // UPDATE if needed
    const generatedFiles = getAllFiles(generatedDir);
    const violations = [];

    for (const file of generatedFiles) {
      const imports = extractImports(file);
      const relativeFile = path.relative(projectRoot, file);

      for (const imp of imports) {
        for (const humansOnlyPath of humansOnlyPaths) {
          if (imp.includes(humansOnlyPath)) {
            violations.push(`  ${relativeFile} imports from humans-only path: ${humansOnlyPath}`);
          }
        }
      }
    }

    if (violations.length > 0) {
      throw new Error(`Generated code cannot import from humans-only paths:\n\n${violations.join('\n')}`);
    }
  });
});
