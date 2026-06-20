#!/usr/bin/env bash
# =============================================================================
# verify-local.sh — Project-specific CI checks
# =============================================================================
# This file is sourced by clare/verify-ci.sh. Add your project's custom checks here.
# CLARE will never overwrite this file — it belongs to your project.
#
# Available helpers (defined in verify-ci.sh before this file is sourced):
#   run_check "Name" "command"   — run a command, track pass/fail
#   pass "message"               — log a passing check
#   fail "message"               — log a failing check
#   info "message"               — informational log
#   warn "message"               — warning log
#   section "Title"              — section header
#
# Available variables:
#   PROJECT_ROOT   — absolute path to the project root
#   FAST_MODE      — true if --fast was passed (skip slow checks)
#   FAIL_FAST      — true by default; false if --fail-slow was passed (continue past failures)
#   FIX_MODE       — true if --fix was passed (auto-fix where possible)
#   HAS_NODE       — true if package.json detected
#   HAS_PYTHON     — true if Python project detected
#   HAS_GO         — true if go.mod detected
#   HAS_RUST       — true if Cargo.toml detected
#
# Examples:
#   run_check "Generate protos" "cd '$PROJECT_ROOT' && make proto 2>&1"
#   run_check "API rate limiting" "cd '$PROJECT_ROOT' && npx jest tests/architecture/api-rules.test.js 2>&1"
#   run_check "Schema lock" "cd '$PROJECT_ROOT' && npm run test:schema-lock 2>&1"
# =============================================================================

# ── Build ────────────────────────────────────────────────────────────────────
# run_check "Generate types" "cd '$PROJECT_ROOT' && npm run gen:types 2>&1"

# ── Linting ──────────────────────────────────────────────────────────────────
# run_check "Custom lint" "cd '$PROJECT_ROOT' && npm run lint:custom 2>&1"

# ── Tests ────────────────────────────────────────────────────────────────────
# run_check "Integration tests" "cd '$PROJECT_ROOT' && npm run test:integration 2>&1"

# ── Architecture Tests ───────────────────────────────────────────────────────
# if ! $FAST_MODE; then
#   run_check "API rate limiting" "cd '$PROJECT_ROOT' && npx jest tests/architecture/api-rules.test.js 2>&1"
# fi
