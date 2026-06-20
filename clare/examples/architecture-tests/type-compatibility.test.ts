/**
 * CLARE Architecture Test: Python Pydantic ↔ TypeScript Type Compatibility
 * =========================================================================
 * CLARE Principles: [A] Assertive — [R] Reality-Aligned — [E] Ephemeral
 *
 * Ensures that TypeScript interfaces stay in sync with Python Pydantic models.
 * When the Python model changes, this test catches drift before it reaches CI.
 *
 * ADAPT THIS FILE:
 *   1. Update PYTHON_SCHEMA_DIR to point to your generated JSON schemas
 *   2. Update TS_TYPES_DIR to point to your TypeScript interfaces
 *   3. Run schema generation (see setup below) and commit schemas alongside models
 *
 * Setup:
 *   python -m pytest tests/gen_schemas.py  # Generate schemas from Pydantic models
 *   OR add to your build: pydantic-gen-schema > src/generated/schemas.json
 *
 * Run with: npx jest tests/architecture/type-compatibility.test.ts
 */

import * as fs from 'fs';
import * as path from 'path';

// ─── Configuration: UPDATE THESE ─────────────────────────────────────────────

/** Directory containing JSON Schema files generated from Pydantic models */
const PYTHON_SCHEMA_DIR = path.join(__dirname, '../../src/generated/schemas');

/** Directory containing your TypeScript interface files */
const TS_TYPES_DIR = path.join(__dirname, '../../src/types/api');

/** Map JSON Schema types to TypeScript types */
const TYPE_MAP: Record<string, string[]> = {
  string: ['string'],
  integer: ['number'],
  number: ['number'],
  boolean: ['boolean'],
  array: ['Array', '[]'],
  object: ['object', 'Record'],
  null: ['null', 'undefined'],
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

interface JsonSchema {
  title?: string;
  type?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  $defs?: Record<string, JsonSchema>;
  anyOf?: JsonSchema[];
  allOf?: JsonSchema[];
  $ref?: string;
}

function loadJsonSchemas(dir: string): Map<string, JsonSchema> {
  const schemas = new Map<string, JsonSchema>();

  if (!fs.existsSync(dir)) {
    console.warn(`⚠  Schema directory not found: ${dir}`);
    console.warn(`   Generate schemas from your Pydantic models and place them in: ${dir}`);
    return schemas;
  }

  for (const file of fs.readdirSync(dir)) {
    if (!file.endsWith('.json')) continue;
    const content = fs.readFileSync(path.join(dir, file), 'utf8');
    const schema: JsonSchema = JSON.parse(content);
    const name = schema.title ?? file.replace('.json', '');
    schemas.set(name, schema);
  }

  return schemas;
}

function loadTypeScriptInterfaces(dir: string): Map<string, string> {
  const interfaces = new Map<string, string>();

  if (!fs.existsSync(dir)) {
    console.warn(`⚠  TypeScript types directory not found: ${dir}`);
    return interfaces;
  }

  function scanDir(d: string) {
    for (const file of fs.readdirSync(d, { withFileTypes: true })) {
      const filePath = path.join(d, file.name);
      if (file.isDirectory()) {
        scanDir(filePath);
      } else if (file.name.endsWith('.ts') && !file.name.endsWith('.test.ts')) {
        const content = fs.readFileSync(filePath, 'utf8');
        // Extract interface names
        const matches = content.matchAll(/(?:export\s+)?interface\s+(\w+)/g);
        for (const match of matches) {
          interfaces.set(match[1], content);
        }
      }
    }
  }

  scanDir(dir);
  return interfaces;
}

/** Convert snake_case to camelCase (Python field → TypeScript field) */
function toCamelCase(str: string): string {
  return str.replace(/_([a-z])/g, (_, char) => char.toUpperCase());
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('Python ↔ TypeScript Type Compatibility', () => {
  let schemas: Map<string, JsonSchema>;
  let tsInterfaces: Map<string, string>;

  beforeAll(() => {
    schemas = loadJsonSchemas(PYTHON_SCHEMA_DIR);
    tsInterfaces = loadTypeScriptInterfaces(TS_TYPES_DIR);
  });

  test('every Pydantic model has a corresponding TypeScript interface', () => {
    const missing: string[] = [];

    for (const [name] of schemas) {
      if (!tsInterfaces.has(name)) {
        missing.push(name);
      }
    }

    if (missing.length > 0) {
      throw new Error(
        `The following Pydantic models have no TypeScript interface:\n` +
        missing.map((n) => `  • ${n}`).join('\n') +
        `\n\nRun the type generation skill (clare/examples/skills/type-sync.md) to add them.`
      );
    }
  });

  test('every TypeScript interface has a corresponding Pydantic schema', () => {
    const missing: string[] = [];

    for (const [name] of tsInterfaces) {
      if (!schemas.has(name)) {
        missing.push(name);
      }
    }

    if (missing.length > 0) {
      // This is a warning, not a failure — TypeScript may have extra interfaces
      console.warn(
        `TypeScript interfaces without a Pydantic model (may be intentional):\n` +
        missing.map((n) => `  • ${n}`).join('\n')
      );
    }
  });

  test('Pydantic required fields exist in TypeScript interfaces', () => {
    const driftReport: string[] = [];

    for (const [modelName, schema] of schemas) {
      const tsContent = tsInterfaces.get(modelName);
      if (!tsContent) continue; // Caught by the previous test

      const requiredFields = schema.required ?? [];
      for (const field of requiredFields) {
        const camelField = toCamelCase(field);
        if (!tsContent.includes(camelField)) {
          driftReport.push(`  ${modelName}.${field} → ${camelField} missing in TypeScript`);
        }
      }
    }

    if (driftReport.length > 0) {
      throw new Error(
        `Field drift detected between Pydantic and TypeScript:\n` +
        driftReport.join('\n') +
        `\n\nRegenerate TypeScript types using clare/examples/skills/type-sync.md`
      );
    }
  });

  // ── EXTEND: Add Zod schema checks ────────────────────────────────────────
  // test('every TypeScript interface has a matching Zod schema', () => { ... });
  // test('Zod schema validation rules match Pydantic field constraints', () => { ... });
});
