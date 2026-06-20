/**
 * CLARE Architecture Test: API Rate Limiting
 * ============================================
 * CLARE Principle: [C] Constrained — [A] Assertive
 *
 * ADAPT THIS FILE to your project — copy it to tests/architecture/ and update:
 *   1. loadApiEndpoints() — return your actual endpoints
 *   2. hasRateLimiting()  — detect your rate limiting implementation
 *
 * Run with: npx jest tests/architecture/api-rules.test.js
 * Wire into: clare/verify-ci.sh (see the architecture tests section)
 */

const fs = require('fs');
const path = require('path');

// ─── Project-Specific: Update These ──────────────────────────────────────────

/**
 * Load all API endpoint definitions from your codebase.
 * Returns an array of objects with at least { file, path, method }.
 *
 * Examples for common frameworks:
 *   - Express: scan app.js / router files for app.get/post/put/delete calls
 *   - Fastify: scan route registrations
 *   - NestJS: scan @Controller / @Get / @Post decorators
 */
function loadApiEndpoints() {
  // ── EXAMPLE: Scan for Express-style route registrations ──
  // Replace this with your actual endpoint detection logic.

  const apiDir = path.join(__dirname, '../../src/api'); // UPDATE: your routes dir
  const endpoints = [];

  if (!fs.existsSync(apiDir)) {
    console.warn(`⚠  API directory not found: ${apiDir}`);
    console.warn(`   Update loadApiEndpoints() in this file to point to your routes.`);
    return endpoints;
  }

  const methods = ['get', 'post', 'put', 'patch', 'delete'];
  const methodPattern = new RegExp(`\\.(${methods.join('|')})\\(`, 'g');

  function scanDir(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const entryPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        scanDir(entryPath);
      } else if (entry.name.endsWith('.js') || entry.name.endsWith('.ts')) {
        const content = fs.readFileSync(entryPath, 'utf8');
        const matches = [...content.matchAll(methodPattern)];
        for (const match of matches) {
          endpoints.push({
            file: path.relative(path.join(__dirname, '../..'), entryPath),
            method: match[1].toUpperCase(),
            content, // pass full content for hasRateLimiting check
          });
        }
      }
    }
  }

  scanDir(apiDir);
  return endpoints;
}

/**
 * Return true if the endpoint has rate limiting applied.
 *
 * Update this to match YOUR rate limiting implementation:
 *   - express-rate-limit: check for 'rateLimit(' or 'rateLimiter'
 *   - Fastify rate-limit: check for '@fastify/rate-limit'
 *   - NestJS throttler: check for @Throttle decorator
 *   - Nginx/Kong: rate limiting may be outside the app — adjust accordingly
 */
function hasRateLimiting(endpoint) {
  const { content } = endpoint;

  // ── UPDATE: Match your rate limiting pattern ──
  return (
    content.includes('rateLimit(') ||        // express-rate-limit
    content.includes('rateLimiter') ||        // common naming
    content.includes('@Throttle(') ||         // NestJS throttler
    content.includes('throttle(') ||          // generic throttle
    content.includes('rate_limit') ||         // snake_case variant
    content.includes('slowDown(')             // express-slow-down
  );
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('API Architecture Rules', () => {
  let endpoints;

  beforeAll(() => {
    endpoints = loadApiEndpoints();
  });

  test('project has at least one API endpoint (sanity check)', () => {
    // If this fails, update loadApiEndpoints() to find your routes.
    if (endpoints.length === 0) {
      console.warn('No endpoints found. Update loadApiEndpoints() in this file.');
    }
    // Soft check — remove this skip once loadApiEndpoints() is configured.
    // expect(endpoints.length).toBeGreaterThan(0);
  });

  test('all API endpoints have rate limiting', () => {
    const endpointsWithoutRateLimiting = endpoints.filter(
      (ep) => !hasRateLimiting(ep)
    );

    if (endpointsWithoutRateLimiting.length > 0) {
      const missing = endpointsWithoutRateLimiting
        .map((ep) => `  ${ep.method} in ${ep.file}`)
        .join('\n');
      throw new Error(
        `The following endpoints are missing rate limiting:\n${missing}\n\n` +
        `Add rate limiting middleware or update hasRateLimiting() in this test ` +
        `if your project handles rate limiting at the infrastructure level.`
      );
    }
  });

  // ── ADD MORE RULES BELOW ──────────────────────────────────────────────────

  // test('all endpoints require authentication', () => { ... });
  // test('all endpoints validate input with a schema', () => { ... });
  // test('all endpoints return errors in a standard format', () => { ... });
});
