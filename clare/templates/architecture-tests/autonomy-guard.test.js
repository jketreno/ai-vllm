/**
 * CLARE Architecture Test: Autonomy Boundary Guard
 * =================================================
 * CLARE Principle: [L] Limited — [C] Constrained
 *
 * This test reads clare/autonomy.yml and verifies that recently staged or
 * committed files don't silently touch humans-only paths without a documented
 * reason. Use this in pre-commit hooks or CI to catch boundary violations.
 *
 * Run with: npx jest tests/architecture/autonomy-guard.test.js
 *
 * For pre-commit hook usage (add to .husky/pre-commit or similar):
 *   npx jest tests/architecture/autonomy-guard.test.js --passWithNoTests
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const yaml = require('js-yaml');
const { minimatch } = require('minimatch');

// ─── Configuration ────────────────────────────────────────────────────────────

const PROJECT_ROOT = path.join(__dirname, '../..');
const AUTONOMY_FILE = path.join(PROJECT_ROOT, 'clare/autonomy.yml');

const CLARE_SKIP_AUTONOMY_GUARD =
  process.env.CLARE_SKIP_AUTONOMY_GUARD === '1' || process.env.CLARE_SKIP_AUTONOMY_GUARD === 'true';
const CLARE_VERBOSE = process.env.CLARE_VERBOSE === '1' || process.env.CLARE_VERBOSE === 'true';

/**
 * Set to true to fail the test when a humans-only file is being committed.
 * Set to false to warn-only (useful for gradual adoption).
 */
const FAIL_ON_HUMANS_ONLY_VIOLATION =
  process.env.FAIL_ON_HUMANS_ONLY_VIOLATION !== 'false' &&
  process.env.FAIL_ON_HUMANS_ONLY_VIOLATION !== '0';

// ─── Helpers ─────────────────────────────────────────────────────────────────

function parseAutonomyYml(content) {
  const parsed = yaml.load(content) || {};
  const modules = Array.isArray(parsed.modules) ? parsed.modules : [];
  const sourcesOfTruth = Array.isArray(parsed.sources_of_truth)
    ? parsed.sources_of_truth.map((entry) => entry?.concept).filter(Boolean)
    : [];
  return { modules, sourcesOfTruth };
}

function getChangedFiles(mode = 'staged') {
  try {
    if (mode === 'staged') {
      const output = execSync('git diff --cached --name-only 2>/dev/null', {
        cwd: PROJECT_ROOT,
        encoding: 'utf8',
      });
      return output.trim().split('\n').filter(Boolean);
    } else {
      // Last commit
      const output = execSync('git diff HEAD~1 --name-only 2>/dev/null', {
        cwd: PROJECT_ROOT,
        encoding: 'utf8',
      });
      return output.trim().split('\n').filter(Boolean);
    }
  } catch {
    return []; // Not a git repo or no changes
  }
}

function matchesPath(filePath, rulePattern) {
  if (rulePattern === '*') return true;
  if (rulePattern.includes('*') || rulePattern.includes('?') || rulePattern.includes('[')) {
    return minimatch(filePath, rulePattern, { dot: true });
  }
  return filePath === rulePattern || filePath.startsWith(`${rulePattern}/`);
}

function findAutonomyLevel(filePath, modules) {
  // Find most specific matching rule (longest prefix wins)
  let bestMatch = null;
  let bestLength = -1;

  for (const mod of modules) {
    if (mod.path === '*') continue; // Default — lowest priority
    if (matchesPath(filePath, mod.path)) {
      if (mod.path.length > bestLength) {
        bestLength = mod.path.length;
        bestMatch = mod;
      }
    }
  }

  if (bestMatch) return bestMatch;

  // Fall back to default rule
  return modules.find((m) => m.path === '*') ?? { level: 'supervised', reason: 'default' };
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('CLARE Autonomy Boundary Guard', () => {
  let autonomyModules;

  beforeAll(() => {
    if (!fs.existsSync(AUTONOMY_FILE)) {
      console.warn('clare/autonomy.yml not found — run the CLARE installer');
      autonomyModules = [];
      return;
    }

    const content = fs.readFileSync(AUTONOMY_FILE, 'utf8');
    const parsed = parseAutonomyYml(content);
    autonomyModules = parsed.modules;
  });

  test('autonomy.yml is present and readable', () => {
    expect(fs.existsSync(AUTONOMY_FILE)).toBe(true);
    expect(autonomyModules).toBeDefined();
  });

  test('autonomy.yml has at least one module boundary defined', () => {
    if (autonomyModules.length === 0) {
      console.warn('No module boundaries in autonomy.yml — run the CLARE installer');
      return;
    }
    expect(autonomyModules.length).toBeGreaterThan(0);
  });

  test('staged files do not silently touch humans-only paths', () => {
    if (CLARE_SKIP_AUTONOMY_GUARD) {
      console.info('CLARE_SKIP_AUTONOMY_GUARD enabled — skipping humans-only boundary check');
      return;
    }

    if (autonomyModules.length === 0) return;

    const stagedFiles = getChangedFiles('staged');
    if (stagedFiles.length === 0) {
      console.info('No staged files — skipping boundary check');
      return;
    }

    const violations = [];

    for (const file of stagedFiles) {
      const module = findAutonomyLevel(file, autonomyModules);
      if (module?.level === 'humans-only') {
        violations.push({ file, reason: module.reason });
      }
    }

    if (violations.length > 0) {
      const report = CLARE_VERBOSE
        ? violations.map((v) => `  • ${v.file}\n    Boundary reason: ${v.reason}`).join('\n')
        : violations.map((v) => `  • ${v.file}`).join('\n');

      const message = CLARE_VERBOSE
        ? `⚠  Staged files touch humans-only paths:\n\n${report}\n\n` +
          `These paths are protected because AI-generated changes here carry risk.\n` +
          `If you intentionally made these changes:\n` +
          `  1. Review clare/autonomy.yml — is this boundary still appropriate?\n` +
          `  2. If yes: commit manually with a descriptive message explaining the intent\n` +
          `  3. If no: update the autonomy level using /project:update-autonomy\n`
        : `⚠  Staged files touch humans-only paths:\n${report}\n` +
          `Set CLARE_VERBOSE=1 to include boundary reasons and remediation steps.`;

      if (FAIL_ON_HUMANS_ONLY_VIOLATION) {
        throw new Error(message);
      } else {
        console.warn(message);
      }
    }
  });

  test('all modules in autonomy.yml have a reason', () => {
    const missing = autonomyModules.filter((m) => !m.reason && m.path !== '*');

    if (missing.length > 0) {
      throw new Error(
        `Autonomy modules are missing a reason field:\n` +
        missing.map((m) => `  • path: "${m.path}", level: ${m.level}`).join('\n') +
        `\n\nAdd a reason to each entry in clare/autonomy.yml`
      );
    }
  });

  test('all modules have a valid autonomy level', () => {
    const validLevels = ['full-autonomy', 'supervised', 'humans-only'];
    const invalid = autonomyModules.filter((m) => m.level && !validLevels.includes(m.level));

    if (invalid.length > 0) {
      throw new Error(
        `Invalid autonomy levels in clare/autonomy.yml:\n` +
        invalid.map((m) => `  • path: "${m.path}", level: "${m.level}"`).join('\n') +
        `\n\nValid values: ${validLevels.join(', ')}`
      );
    }
  });
});
