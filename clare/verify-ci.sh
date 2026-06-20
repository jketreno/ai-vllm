#!/usr/bin/env bash
# =============================================================================
# CLARE verify-ci.sh — Local CI/CD enforcement script
# =============================================================================
# Run this script before marking ANY AI-generated work as complete.
# It mirrors the checks your CI/CD pipeline runs, catching failures locally
# in seconds instead of waiting for the pipeline.
#
# CLARE Principle: [C] Constrained — enforced, not suggested
#
# Usage:
#   ./clare/verify-ci.sh            # Run all checks (stops at first failure)
#   ./clare/verify-ci.sh --fast     # Skip slow checks (architecture tests)
#   ./clare/verify-ci.sh --fix      # Auto-fix linting issues where possible
#   ./clare/verify-ci.sh --fail-slow  # Continue past failures; print summary at end
#   ./clare/verify-ci.sh --list-tests
#   ./clare/verify-ci.sh --run-tests 1,3.1,7
#
# Customization:
#   Add project-specific checks to clare/verify-local.sh (not this file).
#   This file is CLARE-owned and updated by clare-installer.sh.
#   verify-local.sh is YOUR file — it is never overwritten.
#
# AI Instructions: Run this script after generating or modifying any code.
# If it fails, fix the issues and run again. Only report work as complete
# when ALL checks pass. Never skip or bypass this script.
# =============================================================================

set -euo pipefail

SCRIPT_SELF="${BASH_SOURCE:-$0}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SELF")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Include common user-local binary path for optional CLI extensions.
if [[ -d "$HOME/bin" ]]; then
  PATH="$HOME/bin:$PATH"
fi

FAST_MODE=false
FIX_MODE=false
INCLUDE_UNTRACKED=true
FAIL_FAST=true
LIST_TESTS=false
RUN_SELECTED_STAGES=false
RUN_TESTS_CSV=""
SUMMARY_LINES=30
FAILED_CHECKS=()
declare -A CHECK_COMMANDS=()
declare -A FAILED_OUTPUTS=()
SELECTED_TEST_REQUESTS=()

# Single source of truth for verification stages and stage steps.
# The stage number is the array index + 1. Step numbers are each stage's
# array index + 1. print_stage_list, --run-tests validation, and the main run
# loop all derive from this — add stages/steps here and nowhere else.
# Functions are referenced by name only; they are defined later in the file and
# are not resolved until main() runs the loop.
#
# The STAGE_N_STEPS arrays are read via `local -n` namerefs (see run_stage,
# print_stage_list, parse_selected_stages), which shellcheck cannot trace —
# hence the SC2034 ("appears unused") suppression on each.
STAGES=(
  "Build"
  "Linting"
  "Tests"
  "Architecture Tests"
  "CLARE Autonomy Boundaries"
  "Extensions"
  "Project-Specific Checks"
)
# shellcheck disable=SC2034  # read via `local -n` nameref in run_stage
STAGE_1_STEPS=(
  "Node.js build|check_build_node"
  "Python syntax check|check_build_python"
  "Go build|check_build_go"
  "Rust build|check_build_rust"
)
# shellcheck disable=SC2034  # read via `local -n` nameref in run_stage
STAGE_2_STEPS=(
  "Node.js lint/type checks|check_lint_node"
  "Python lint/type checks|check_lint_python"
  "Go lint checks|check_lint_go"
)
# shellcheck disable=SC2034  # read via `local -n` nameref in run_stage
STAGE_3_STEPS=(
  "Node.js tests|check_tests_node"
  "Python tests|check_tests_python"
  "Go tests|check_tests_go"
  "Rust tests|check_tests_rust"
)
# shellcheck disable=SC2034  # read via `local -n` nameref in run_stage
STAGE_4_STEPS=(
  "Node.js architecture tests|check_architecture_node"
  "Python architecture tests|check_architecture_python"
)
STAGE_5_STEPS=()
STAGE_6_STEPS=()
# shellcheck disable=SC2034  # read via `local -n` nameref in run_stage
STAGE_7_STEPS=(
  "verify-local.sh|source_local_checks"
)

# Failed checks retain their captured output in temp files so the failure summary
# can print snippets and a "Full log:" path. Clean those up when the script exits
# (including the fail-fast `exit 1`) so they don't accumulate in the temp dir.
cleanup_failed_outputs() {
  local f
  for f in "${FAILED_OUTPUTS[@]+"${FAILED_OUTPUTS[@]}"}"; do
    [[ -n "$f" && -f "$f" ]] && rm -f "$f"
  done
}
trap cleanup_failed_outputs EXIT

usage() {
  cat <<'EOF'
Usage: ./clare/verify-ci.sh [options]

Run local CI/CD checks and enforce CLARE constraints.

Options:
  --fast        Skip slow checks (architecture tests)
  --fix         Auto-fix lint issues where supported
  --fail-slow   Continue past failures and print a summary at the end
  --fail-fast   (default) Stop at the first failing check; alias kept for compatibility
  --list-tests  List numbered verification stages and exit
  --run-tests <csv>
               Run selected stages or stage steps in order (example: --run-tests 1,3.1,7)
  --exclude-untracked
               Check only tracked files (exclude untracked files)
  -h, --help    Show this help message and exit
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  arg="$1"
  case "$arg" in
    --fast) FAST_MODE=true ;;
    --fix) FIX_MODE=true ;;
    --fail-fast) FAIL_FAST=true ;;
    --fail-slow) FAIL_FAST=false ;;
    --list-tests) LIST_TESTS=true ;;
    --run-tests)
      shift
      if [[ $# -eq 0 || "${1:-}" == --* ]]; then
        echo "Missing value for --run-tests (expected comma-separated stage or stage-step numbers)." >&2
        echo "Run './clare/verify-ci.sh --list-tests' to see available stages." >&2
        exit 2
      fi
      RUN_TESTS_CSV="$1"
      RUN_SELECTED_STAGES=true
      ;;
    --run-tests=*)
      RUN_TESTS_CSV="${arg#--run-tests=}"
      RUN_SELECTED_STAGES=true
      ;;
    --exclude-untracked) INCLUDE_UNTRACKED=false ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Run './clare/verify-ci.sh --help' for usage." >&2
      exit 2
      ;;
  esac
  shift
done

# ─── Helpers ──────────────────────────────────────────────────────────────────

supports_color() {
  [[ -z "${NO_COLOR:-}" ]] || return 1
  [[ "${TERM:-}" != "dumb" ]] || return 1

  if [[ -n "${FORCE_COLOR:-}" || -n "${CLICOLOR_FORCE:-}" || "${GITHUB_ACTIONS:-}" == "true" ]]; then
    return 0
  fi

  [[ -t 1 ]] || return 1

  if command -v tput >/dev/null 2>&1; then
    local colors
    colors="$(tput colors 2>/dev/null || printf '0')"
    [[ "$colors" =~ ^[0-9]+$ ]] || return 1
    ((colors >= 8)) || return 1
  fi

  return 0
}

if supports_color; then
  GREEN='\033[0;32m'
  RED='\033[0;31m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m' # No Color
else
  GREEN=''
  RED=''
  YELLOW=''
  BLUE=''
  NC=''
fi

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() {
  echo -e "${RED}❌ $1${NC}"
  FAILED_CHECKS+=("$1")
}
info() { echo -e "${BLUE}ℹ  $1${NC}"; }
warn() { echo -e "${YELLOW}⚠  $1${NC}"; }
section() { echo -e "\n${BLUE}── $1 ──${NC}"; }

print_stage_list() {
  echo "Available verify-ci stages:"
  local i step_index step_entry
  for i in "${!STAGES[@]}"; do
    echo "  $((i + 1)). ${STAGES[$i]}"
    local -n stage_steps="STAGE_$((i + 1))_STEPS"
    for step_index in "${!stage_steps[@]}"; do
      step_entry="${stage_steps[$step_index]}"
      echo "    $((i + 1)).$((step_index + 1)). ${step_entry%%|*}"
    done
  done
}

parse_selected_stages() {
  local csv="$1"

  if [[ -z "$csv" ]]; then
    echo "Missing value for --run-tests (expected comma-separated stage or stage-step numbers)." >&2
    echo "Run './clare/verify-ci.sh --list-tests' to see available stages." >&2
    exit 2
  fi

  if [[ ! "$csv" =~ ^[1-9][0-9]*(\.[1-9][0-9]*)?(,[1-9][0-9]*(\.[1-9][0-9]*)?)*$ ]]; then
    echo "Invalid --run-tests value: $csv" >&2
    echo "Expected comma-separated stage or stage-step numbers, for example: --run-tests 1,3.1,7" >&2
    echo "Run './clare/verify-ci.sh --list-tests' to see available stages." >&2
    exit 2
  fi

  local request stage step
  IFS=',' read -r -a SELECTED_TEST_REQUESTS <<<"$csv"
  for request in "${SELECTED_TEST_REQUESTS[@]}"; do
    stage="${request%%.*}"
    if ((stage < 1 || stage > ${#STAGES[@]})); then
      echo "Unknown verify-ci stage number: $stage (from $request)" >&2
      echo "Run './clare/verify-ci.sh --list-tests' to see available stages." >&2
      exit 2
    fi

    if [[ "$request" == *.* ]]; then
      step="${request#*.}"
      local -n stage_steps="STAGE_${stage}_STEPS"
      if ((step < 1 || step > ${#stage_steps[@]})); then
        echo "Unknown verify-ci stage step number: $request" >&2
        echo "Run './clare/verify-ci.sh --list-tests' to see available stages." >&2
        exit 2
      fi
    fi
  done
}

run_stage_step() {
  local stage_number="$1"
  local step_number="$2"
  local -n stage_steps="STAGE_${stage_number}_STEPS"
  local step_entry="${stage_steps[$((step_number - 1))]}"
  local name fn arg
  IFS='|' read -r name fn arg <<<"$step_entry"

  if [[ -n "${arg:-}" ]]; then
    "$fn" "$arg"
  else
    "$fn"
  fi
}

run_stage() {
  local stage_number="$1"
  local stage_name="${STAGES[$((stage_number - 1))]}"
  local -n stage_steps="STAGE_${stage_number}_STEPS"
  local step_index

  section "$stage_name"
  for step_index in "${!stage_steps[@]}"; do
    run_stage_step "$stage_number" "$((step_index + 1))"
  done
}

run_selected_tests() {
  local request stage step
  for request in "${SELECTED_TEST_REQUESTS[@]}"; do
    stage="${request%%.*}"
    if [[ "$request" == *.* ]]; then
      step="${request#*.}"
      section "${STAGES[$((stage - 1))]}"
      run_stage_step "$stage" "$step"
    else
      run_stage "$stage"
    fi
  done
}

run_check() {
  local name="$1"
  local cmd="$2"
  CHECK_COMMANDS["$name"]="$cmd"
  echo -e "${BLUE}  → ${name}${NC}"

  local out_file
  out_file="$(mktemp)"

  # Execute the command and capture stdout+stderr to a temp file
  if bash -c "$cmd" >"$out_file" 2>&1; then
    pass "$name"
    rm -f "$out_file"
    return 0
  else
    fail "$name"
    FAILED_OUTPUTS["$name"]="$out_file"
    if $FAIL_FAST; then
      # print_failure_summary is defined later in the file; this is safe because
      # run_check is only invoked from main(), after all functions are parsed.
      echo ""
      print_failure_summary
      exit 1
    fi
    return 1
  fi
}

list_project_files_respecting_gitignore() {
  if command -v git >/dev/null 2>&1 && git -C "$PROJECT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    (
      cd "$PROJECT_ROOT"
      if $INCLUDE_UNTRACKED; then
        # Tracked + untracked files, respecting .gitignore/.git/info/exclude/global excludes.
        git ls-files -co --exclude-standard
      else
        # Tracked files only.
        git ls-files
      fi
    )
  else
    find "$PROJECT_ROOT" -type f \
      ! -path "*/.git/*" \
      ! -path "*/node_modules/*" \
      ! -path "*/.venv/*" \
      ! -path "*/venv/*" \
      ! -path "*/dist/*" \
      ! -path "*/build/*" \
      ! -path "*/coverage/*" \
      | sed "s#^$PROJECT_ROOT/##"
  fi
}

is_default_ignored_path() {
  local rel_path="$1"
  case "$rel_path" in
    node_modules/* | */node_modules/* | .venv/* | */.venv/* | venv/* | */venv/* | .git/* | */.git/* | dist/* | */dist/* | build/* | */build/* | coverage/* | */coverage/*)
      return 0
      ;;
  esac
  return 1
}

path_within_scan_paths() {
  local rel_path="$1"
  local scan_paths="$2"

  [[ -z "$scan_paths" ]] && return 0

  for scan_path in $scan_paths; do
    scan_path="${scan_path#./}"
    [[ "$scan_path" == "." ]] && return 0
    [[ -z "$scan_path" ]] && continue
    if [[ "$rel_path" == "$scan_path" || "$rel_path" == "$scan_path/"* ]]; then
      return 0
    fi
  done
  return 1
}

scan_paths_exist() {
  local scan_paths="$1"

  [[ -z "$scan_paths" ]] && return 0

  for scan_path in $scan_paths; do
    scan_path="${scan_path#./}"
    [[ -z "$scan_path" || "$scan_path" == "." ]] && return 0
    [[ -e "$PROJECT_ROOT/$scan_path" ]] && return 0
  done

  return 1
}

matches_any_file_type() {
  local rel_path="$1"
  local file_types="$2"

  [[ -z "$file_types" ]] && return 0

  for ext in $file_types; do
    case "$rel_path" in
      *."$ext") return 0 ;;
    esac
  done
  return 1
}

is_extension_excluded() {
  local rel_path="$1"
  local exclude_patterns="$2"
  local base_name
  base_name="$(basename "$rel_path")"

  for pattern in $exclude_patterns; do
    case "$base_name" in
      $pattern) return 0 ;;
    esac
    case "$rel_path" in
      $pattern | $pattern/* | */$pattern | */$pattern/*) return 0 ;;
    esac
  done

  return 1
}

collect_extension_files() {
  local tool_name="$1"
  local scan_paths="$2"
  local file_types="$3"
  local exclude_patterns="$4"
  local effective_scan_paths="$scan_paths"

  if ! scan_paths_exist "$scan_paths"; then
    effective_scan_paths="."
    warn "$tool_name: configured paths not found (paths: $scan_paths); falling back to project root" >&2
  fi

  local matched_files=0
  while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue
    is_default_ignored_path "$rel_path" && continue
    path_within_scan_paths "$rel_path" "$effective_scan_paths" || continue
    matches_any_file_type "$rel_path" "$file_types" || continue
    is_extension_excluded "$rel_path" "$exclude_patterns" && continue

    local abs_path="$PROJECT_ROOT/$rel_path"
    [[ -f "$abs_path" ]] || continue
    echo "$abs_path"
    matched_files=1
  done < <(list_project_files_respecting_gitignore)

  if [[ "$matched_files" -eq 0 ]]; then
    warn "$tool_name: no files matched configured paths/types (paths: $effective_scan_paths, types: $file_types)" >&2
    return 1
  fi

  return 0
}

run_eslint_complexity_check() {
  local command="$1"
  local threshold="$2"
  local scan_paths="$3"
  local extra_flags="$4"
  local file_types="${5:-js jsx ts tsx}"
  local exclude_patterns="$6"

  eslint_config_exists() {
    local config_name
    for config_name in eslint.config.js eslint.config.mjs eslint.config.cjs; do
      [[ -f "$PROJECT_ROOT/$config_name" ]] && return 0
    done
    return 1
  }

  local files=()
  mapfile -t files < <(collect_extension_files "ESLint complexity" "$scan_paths" "$file_types" "$exclude_patterns" || true)
  [[ ${#files[@]} -eq 0 ]] && return 0

  local eslint_cmd=()
  if [[ "$command" == "npx" ]]; then
    if ! node_tool_command eslint_cmd "eslint" "eslint"; then
      eslint_cmd=("npx" "--no-install" "eslint")
    fi
  else
    eslint_cmd=("$command")
  fi

  if [[ -n "$threshold" ]]; then
    eslint_cmd+=("--rule" "complexity: [\"error\", $threshold]")
  fi

  if [[ -n "$extra_flags" ]]; then
    # Intentional word splitting for a user-specified flag string.
    # shellcheck disable=SC2206
    local extra_parts=($extra_flags)
    eslint_cmd+=("${extra_parts[@]}")
  fi

  local has_eslint_config=true
  if ! eslint_config_exists; then
    has_eslint_config=false
  fi

  if [[ "$has_eslint_config" == "false" ]]; then
    eslint_cmd+=("--no-config-lookup")

    # In no-config mode ESLint cannot parse TS/TSX without a configured parser.
    local filtered_files=()
    local file_path
    for file_path in "${files[@]}"; do
      case "$file_path" in
        *.ts | *.tsx) ;;
        *) filtered_files+=("$file_path") ;;
      esac
    done

    if [[ ${#filtered_files[@]} -eq 0 ]]; then
      warn "ESLint complexity: no eslint.config.* found and only TS/TSX files matched; skipping" >&2
      return 0
    fi

    files=("${filtered_files[@]}")
    warn "ESLint complexity: no eslint.config.* found; using --no-config-lookup and JS/JSX files only" >&2
  fi

  eslint_cmd+=("${files[@]}")

  local cmd_string
  printf -v cmd_string '%q ' "${eslint_cmd[@]}"
  run_check "ESLint complexity (TypeScript/JavaScript)" "$cmd_string 2>&1" || true
}

run_golangci_lint_complexity_check() {
  local command="$1"
  local threshold="$2"
  local scan_paths="$3"
  local extra_flags="$4"
  local file_types="${5:-go}"
  local exclude_patterns="$6"

  local files=()
  mapfile -t files < <(collect_extension_files "golangci-lint complexity" "$scan_paths" "$file_types" "$exclude_patterns" || true)
  [[ ${#files[@]} -eq 0 ]] && return 0

  if [[ -z "$threshold" ]]; then
    threshold="15"
  fi

  local config_file
  config_file="$(mktemp "$PROJECT_ROOT/.golangci-complexity.XXXXXX.yml")"
  cat >"$config_file" <<EOF
version: "2"
linters:
  default: none
  enable:
    - cyclop
    - gocognit
  settings:
    cyclop:
      max-complexity: $threshold
    gocognit:
      min-complexity: $threshold
EOF

  local -A dirs=()
  local file
  for file in "${files[@]}"; do
    dirs["$(dirname "$file")"]=1
  done

  local golangci_cmd=("$command" "run" "--config" "$config_file")
  if [[ -n "$extra_flags" ]]; then
    # Intentional word splitting for a user-specified flag string.
    # shellcheck disable=SC2206
    local extra_parts=($extra_flags)
    golangci_cmd+=("${extra_parts[@]}")
  fi

  local dir
  for dir in "${!dirs[@]}"; do
    golangci_cmd+=("$dir")
  done

  local cmd_string
  printf -v cmd_string '%q ' "${golangci_cmd[@]}"
  run_check "golangci-lint complexity (Go)" "$cmd_string 2>&1" || true
  rm -f "$config_file"
}

run_complexipy_check() {
  local command="$1"
  local threshold="$2"
  local scan_paths="$3"
  local extra_flags="$4"
  local file_types="${5:-py}"
  local exclude_patterns="$6"

  local files=()
  mapfile -t files < <(collect_extension_files "complexipy" "$scan_paths" "$file_types" "$exclude_patterns" || true)
  [[ ${#files[@]} -eq 0 ]] && return 0

  local complexipy_cmd=("$command")

  if [[ -n "$threshold" ]]; then
    complexipy_cmd+=("--max-complexity-allowed" "$threshold")
  fi

  if [[ -n "$extra_flags" ]]; then
    # Intentional word splitting for a user-specified flag string.
    # shellcheck disable=SC2206
    local extra_parts=($extra_flags)
    complexipy_cmd+=("${extra_parts[@]}")
  fi

  complexipy_cmd+=("${files[@]}")

  local cmd_string
  printf -v cmd_string '%q ' "${complexipy_cmd[@]}"
  run_check "complexipy (Python cognitive complexity)" "$cmd_string 2>&1" || true
}

run_shellmetrics_check() {
  local command="$1"
  local threshold="$2"
  local scan_paths="$3"
  local extra_flags="$4"
  local file_types="${5:-sh bash}"
  local exclude_patterns="$6"

  local files=()
  mapfile -t files < <(collect_extension_files "shellmetrics" "$scan_paths" "$file_types" "$exclude_patterns" || true)
  [[ ${#files[@]} -eq 0 ]] && return 0

  if [[ -z "$threshold" ]]; then
    threshold="15"
  fi

  local shellmetrics_cmd=("$command" "--no-color")

  if [[ -n "$extra_flags" ]]; then
    # Intentional word splitting for a user-specified flag string.
    # shellcheck disable=SC2206
    local extra_parts=($extra_flags)
    shellmetrics_cmd+=("${extra_parts[@]}")
  fi

  shellmetrics_cmd+=("${files[@]}")

  local cmd_string
  printf -v cmd_string '%q ' "${shellmetrics_cmd[@]}"
  echo -e "${BLUE}  → shellmetrics complexity (Bash/Shell)${NC}"

  local output_file
  output_file="$(mktemp)"

  if ! "${shellmetrics_cmd[@]}" >"$output_file" 2>&1; then
    fail "shellmetrics complexity (Bash/Shell)"
    cat "$output_file"
    rm -f "$output_file"
    return 1
  fi

  local max_ccn
  max_ccn="$(awk '
    /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+/ && $0 !~ /<main>/ {
      ccn = $2 + 0
      if (ccn > max) {
        max = ccn
      }
      found = 1
    }
    END {
      if (found == 1) {
        print max
      }
    }
  ' "$output_file")"

  if [[ -z "$max_ccn" ]]; then
    fail "shellmetrics complexity (Bash/Shell)"
    echo "shellmetrics output did not contain parseable CCN rows." >&2
    cat "$output_file"
    rm -f "$output_file"
    return 1
  fi

  if ((max_ccn > threshold)); then
    fail "shellmetrics complexity (Bash/Shell)"
    echo "Maximum shell cyclomatic complexity exceeds threshold (max=$max_ccn, threshold=$threshold)." >&2
    echo "Functions over threshold:" >&2
    awk -v t="$threshold" '
      /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+/ && $0 !~ /<main>/ {
        ccn = $2 + 0
        if (ccn > t) {
          print
        }
      }
    ' "$output_file" >&2
    rm -f "$output_file"
    return 1
  fi

  pass "shellmetrics complexity (Bash/Shell) (max CCN: $max_ccn <= $threshold)"
  rm -f "$output_file"
}

is_clare_framework_repo() {
  [[ -f "$PROJECT_ROOT/scripts/clare-installer.sh" && -f "$PROJECT_ROOT/install/root/CLAUDE.md" && -f "$PROJECT_ROOT/clare/verify-ci.sh" ]]
}

append_eslint_ignore_flags() {
  local out_ref="$1"
  shift
  local -a patterns=("$@")
  local flags=""
  local pattern escaped_pattern

  for pattern in "${patterns[@]}"; do
    printf -v escaped_pattern '%q' "$pattern"
    flags+=" --ignore-pattern $escaped_pattern"
  done

  printf -v "$out_ref" '%s' "$flags"
}

node_supports_flag() {
  local flag="$1"
  node "$flag" -e "" >/dev/null 2>&1
}

default_node_tool_flags() {
  # node_tool_flags is invoked via command substitution ($(...)), so any cache
  # written here would die with the subshell and never be reused. Memoize in a
  # module-level global that is populated once, before the lint loop, by
  # init_node_tool_flags — keeping the node_supports_flag probe to a single run.
  if [[ -n "${CLARE_DEFAULT_NODE_TOOL_FLAGS_COMPUTED:-}" ]]; then
    printf '%s' "${CLARE_DEFAULT_NODE_TOOL_FLAGS:-}"
    return 0
  fi

  init_node_tool_flags
  printf '%s' "${CLARE_DEFAULT_NODE_TOOL_FLAGS:-}"
}

init_node_tool_flags() {
  [[ -n "${CLARE_DEFAULT_NODE_TOOL_FLAGS_COMPUTED:-}" ]] && return 0

  CLARE_DEFAULT_NODE_TOOL_FLAGS_COMPUTED=true
  CLARE_DEFAULT_NODE_TOOL_FLAGS=""

  # Some environments hit V8 Turboshaft crashes in Node-based tooling. That is a
  # local/transient issue, so we do NOT disable Turboshaft for every project by
  # default. Projects that need the workaround opt in with
  # CLARE_DISABLE_TURBOSHAFT=1 (or set CLARE_NODE_TOOL_FLAGS to supply their own
  # flags). When opted in, the flag is still gated on Node actually supporting it.
  if [[ "${CLARE_DISABLE_TURBOSHAFT:-}" == "1" ]] && node_supports_flag "--no-turboshaft"; then
    CLARE_DEFAULT_NODE_TOOL_FLAGS="--no-turboshaft"
  fi
}

node_tool_flags() {
  if [[ -n "${CLARE_NODE_TOOL_FLAGS+x}" ]]; then
    printf '%s' "$CLARE_NODE_TOOL_FLAGS"
  else
    default_node_tool_flags
  fi
}

resolve_node_package_bin() {
  local package_name="$1"
  local bin_name="$2"

  node - "$PROJECT_ROOT" "$package_name" "$bin_name" <<'NODE'
const path = require('node:path');

const [projectRoot, packageName, binName] = process.argv.slice(2);
const packageJsonPath = require.resolve(`${packageName}/package.json`, {
  paths: [projectRoot],
});
const packageJson = require(packageJsonPath);
const bin = packageJson.bin;

let binPath;
if (typeof bin === 'string') {
  binPath = bin;
} else if (bin && typeof bin === 'object') {
  binPath = bin[binName] || Object.values(bin)[0];
}

if (!binPath) {
  process.exit(1);
}

console.log(path.resolve(path.dirname(packageJsonPath), binPath));
NODE
}

node_tool_command() {
  local out_ref="$1"
  local package_name="$2"
  local bin_name="$3"
  local -n out_cmd="$out_ref"
  local bin_path flags

  out_cmd=()
  bin_path="$(resolve_node_package_bin "$package_name" "$bin_name")" || return 1
  flags="$(node_tool_flags)"

  out_cmd=("node")
  if [[ -n "$flags" ]]; then
    # Intentional word splitting for caller-provided Node/V8 flag strings.
    # shellcheck disable=SC2206
    local flag_parts=($flags)
    out_cmd+=("${flag_parts[@]}")
  fi
  out_cmd+=("$bin_path")
}

run_node_lint_checks() {
  local is_framework_repo="$1"
  shift
  local -a ignore_patterns=("$@")
  local fix_flag=""
  $FIX_MODE && fix_flag="--fix"
  local has_eslint_config=false

  if [[ -f "$PROJECT_ROOT/.eslintrc.js" ||
    -f "$PROJECT_ROOT/.eslintrc.cjs" ||
    -f "$PROJECT_ROOT/.eslintrc.json" ||
    -f "$PROJECT_ROOT/eslint.config.js" ||
    -f "$PROJECT_ROOT/eslint.config.mjs" ||
    -f "$PROJECT_ROOT/eslint.config.cjs" ]]; then
    has_eslint_config=true
  fi

  if [[ "$has_eslint_config" == "true" ]] \
    && node -e "require.resolve('eslint/package.json', { paths: ['$PROJECT_ROOT'] })" 2>/dev/null; then
    local eslint_ignore_flags=""
    if [[ "$is_framework_repo" == "false" ]]; then
      append_eslint_ignore_flags eslint_ignore_flags "${ignore_patterns[@]}"
    fi
    local eslint_cmd=()
    local eslint_cmd_string=""
    if node_tool_command eslint_cmd "eslint" "eslint"; then
      printf -v eslint_cmd_string '%q ' "${eslint_cmd[@]}"
      run_check "ESLint" "cd '$PROJECT_ROOT' && $eslint_cmd_string. $fix_flag$eslint_ignore_flags 2>&1" || true
    else
      fail "ESLint"
      echo "ESLint is installed, but its executable could not be resolved from package.json." >&2
    fi
  elif [[ "$has_eslint_config" == "true" ]]; then
    warn "ESLint config found but ESLint not installed. Run: npm install"
  fi

  if node -e "require.resolve('prettier/package.json', { paths: ['$PROJECT_ROOT'] })" 2>/dev/null; then
    local prettier_flag="--check"
    $FIX_MODE && prettier_flag="--write"
    if [[ "$is_framework_repo" == "false" ]]; then
      local prettier_ignore_file
      prettier_ignore_file="$(mktemp "$PROJECT_ROOT/.prettierignore.clare.XXXXXX")"
      if [[ -f "$PROJECT_ROOT/.prettierignore" ]]; then
        cat "$PROJECT_ROOT/.prettierignore" >"$prettier_ignore_file"
        echo "" >>"$prettier_ignore_file"
      fi
      local pattern
      for pattern in "${ignore_patterns[@]}"; do
        echo "$pattern" >>"$prettier_ignore_file"
      done

      local prettier_cmd=()
      local prettier_cmd_string=""
      if node_tool_command prettier_cmd "prettier" "prettier"; then
        printf -v prettier_cmd_string '%q ' "${prettier_cmd[@]}"
        run_check "Prettier" "cd '$PROJECT_ROOT' && $prettier_cmd_string$prettier_flag --ignore-path '$prettier_ignore_file' . 2>&1" || true
      else
        fail "Prettier"
        echo "Prettier is installed, but its executable could not be resolved from package.json." >&2
      fi
      rm -f "$prettier_ignore_file"
    else
      local prettier_cmd=()
      local prettier_cmd_string=""
      if node_tool_command prettier_cmd "prettier" "prettier"; then
        printf -v prettier_cmd_string '%q ' "${prettier_cmd[@]}"
        run_check "Prettier" "cd '$PROJECT_ROOT' && $prettier_cmd_string$prettier_flag . 2>&1" || true
      else
        fail "Prettier"
        echo "Prettier is installed, but its executable could not be resolved from package.json." >&2
      fi
    fi
  fi

  if node -e "require.resolve('typescript/package.json', { paths: ['$PROJECT_ROOT'] })" 2>/dev/null; then
    local tsc_cmd=()
    local tsc_cmd_string=""
    if node_tool_command tsc_cmd "typescript" "tsc"; then
      printf -v tsc_cmd_string '%q ' "${tsc_cmd[@]}"
      run_check "TypeScript (no-emit)" "cd '$PROJECT_ROOT' && $tsc_cmd_string--noEmit 2>&1" || true
    else
      fail "TypeScript (no-emit)"
      echo "TypeScript is installed, but tsc could not be resolved from package.json." >&2
    fi
  fi
}

run_python_lint_checks() {
  if command -v ruff &>/dev/null; then
    local fix_flag=""
    $FIX_MODE && fix_flag="--fix"
    run_check "Ruff" "cd '$PROJECT_ROOT' && ruff check $fix_flag . 2>&1" || true
  elif command -v flake8 &>/dev/null; then
    run_check "Flake8" "cd '$PROJECT_ROOT' && flake8 . 2>&1" || true
  fi

  if command -v mypy &>/dev/null; then
    run_check "Mypy" "cd '$PROJECT_ROOT' && mypy . 2>&1" || true
  fi
}

run_go_lint_checks() {
  for dir in "${GO_MOD_DIRS[@]}"; do
    run_check "Go vet" "cd '$dir' && go vet ./... 2>&1" || true
    if command -v golint &>/dev/null; then
      run_check "Golint" "cd '$dir' && golint ./... 2>&1" || true
    fi
  done
}

# ─── Project Type Detection ───────────────────────────────────────────────────

detect_project() {
  HAS_NODE=false
  HAS_NODE_RUNTIME=false
  HAS_PYTHON=false
  PYTHON_CMD=""
  HAS_GO=false
  GO_MOD_DIRS=()
  HAS_RUST=false
  HAS_MAKE=false

  [[ -f "$PROJECT_ROOT/package.json" ]] && HAS_NODE=true || true
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    HAS_NODE_RUNTIME=true
  fi
  [[ -f "$PROJECT_ROOT/pyproject.toml" || -f "$PROJECT_ROOT/setup.py" || -f "$PROJECT_ROOT/requirements.txt" ]] && HAS_PYTHON=true || true
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
  fi
  # Collect every go.mod in the tree (git-aware; falls back to find).
  local go_mod_files=()
  if command -v git >/dev/null 2>&1 && git -C "$PROJECT_ROOT" rev-parse --git-dir >/dev/null 2>&1; then
    mapfile -t go_mod_files < <(git -C "$PROJECT_ROOT" ls-files -- '*/go.mod' 'go.mod' 2>/dev/null)
  else
    mapfile -t go_mod_files < <(find "$PROJECT_ROOT" -name "go.mod" ! -path "*/.git/*" | sed "s#^$PROJECT_ROOT/##")
  fi
  for rel in "${go_mod_files[@]}"; do
    [[ -n "$rel" ]] || continue
    GO_MOD_DIRS+=("$PROJECT_ROOT/$(dirname "$rel")")
    HAS_GO=true
  done
  [[ -f "$PROJECT_ROOT/Cargo.toml" ]] && HAS_RUST=true || true
  [[ -f "$PROJECT_ROOT/Makefile" ]] && HAS_MAKE=true || true
}

# ─── Check Groups ─────────────────────────────────────────────────────────────

check_build_node() {
  if $HAS_NODE; then
    if ! $HAS_NODE_RUNTIME; then
      warn "package.json found but node/npm are not installed; skipping Node.js build checks"
    else
      local build_script
      build_script=$(node -e "const p=require('$PROJECT_ROOT/package.json'); console.log(p.scripts && p.scripts.build ? 'build' : '')" 2>/dev/null || echo "")
      if [[ -n "$build_script" ]]; then
        run_check "npm build" "cd '$PROJECT_ROOT' && npm run build 2>&1" || true
      else
        info "No build script found in package.json — skipping"
      fi
    fi
  fi
}

check_build_python() {
  if $HAS_PYTHON; then
    if [[ -n "$PYTHON_CMD" ]]; then
      run_check "Python syntax check" "$PYTHON_CMD -m compileall '$PROJECT_ROOT' -q 2>&1 | head -20" || true
    else
      warn "Python project detected but no Python interpreter found; skipping syntax check"
    fi
  fi
}

check_build_go() {
  if $HAS_GO; then
    for dir in "${GO_MOD_DIRS[@]}"; do
      run_check "Go build" "cd '$dir' && go build ./... 2>&1" || true
    done
  fi
}

check_build_rust() {
  if $HAS_RUST; then
    run_check "Cargo build" "cd '$PROJECT_ROOT' && cargo build 2>&1" || true
  fi
}

check_lint_node() {
  local framework_repo=false
  if is_clare_framework_repo; then
    framework_repo=true
  fi

  local -a clare_managed_lint_ignore_patterns=(
    "clare/**"
    ".github/copilot-instructions.md"
    ".github/instructions/**"
    ".github/prompts/**"
    ".claude/commands/**"
    ".vscode/prompts/**"
    ".codex/skills/**"
    ".cursor/rules/clare-*.mdc"
    ".cursor/rules/skill-*.mdc"
    ".cursorrules"
    "AGENTS.md"
    "CLAUDE.md"
  )

  if $HAS_NODE; then
    if ! $HAS_NODE_RUNTIME; then
      warn "package.json found but node/npm are not installed; skipping Node.js lint/type checks"
    else
      run_node_lint_checks "$framework_repo" "${clare_managed_lint_ignore_patterns[@]}"
    fi
  fi
}

check_lint_python() {
  if $HAS_PYTHON; then
    run_python_lint_checks
  fi
}

check_lint_go() {
  if $HAS_GO; then
    run_go_lint_checks
  fi
}

check_tests_node() {
  if $HAS_NODE; then
    if ! $HAS_NODE_RUNTIME; then
      warn "package.json found but node/npm are not installed; skipping Node.js test checks"
    else
      local test_script
      test_script=$(node -e "const p=require('$PROJECT_ROOT/package.json'); console.log(p.scripts && p.scripts.test ? 'test' : '')" 2>/dev/null || echo "")
      if [[ -n "$test_script" ]]; then
        run_check "npm test" "cd '$PROJECT_ROOT' && npm test 2>&1" || true
      fi
    fi
  fi
}

check_tests_python() {
  if $HAS_PYTHON; then
    if command -v pytest &>/dev/null; then
      run_check "pytest" "cd '$PROJECT_ROOT' && pytest --tb=short -q 2>&1" || true
    fi
  fi
}

check_tests_go() {
  if $HAS_GO; then
    for dir in "${GO_MOD_DIRS[@]}"; do
      run_check "Go test" "cd '$dir' && go test ./... 2>&1" || true
    done
  fi
}

check_tests_rust() {
  if $HAS_RUST; then
    run_check "Cargo test" "cd '$PROJECT_ROOT' && cargo test 2>&1" || true
  fi
}

check_architecture_node() {
  if $FAST_MODE; then
    warn "Architecture tests skipped (--fast mode)"
    return 0
  fi

  if $HAS_NODE; then
    if ! $HAS_NODE_RUNTIME; then
      warn "package.json found but node/npm are not installed; skipping Node.js architecture checks"
    else
      local arch_script
      arch_script=$(node -e "const p=require('$PROJECT_ROOT/package.json'); console.log(p.scripts && p.scripts['test:architecture'] ? 'test:architecture' : '')" 2>/dev/null || echo "")
      if [[ -n "$arch_script" ]]; then
        run_check "Architecture tests" "cd '$PROJECT_ROOT' && npm run test:architecture 2>&1" || true
      fi
    fi
  fi
}

check_architecture_python() {
  # The --fast skip message is emitted once by check_architecture_node (the
  # first step in this stage); stay silent here to avoid printing it twice.
  if $FAST_MODE; then
    return 0
  fi

  if $HAS_PYTHON; then
    if [[ -d "$PROJECT_ROOT/tests/architecture" ]]; then
      run_check "Architecture tests (pytest)" "cd '$PROJECT_ROOT' && pytest tests/architecture/ --tb=short -q 2>&1" || true
    fi
  fi
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

refresh_autonomy_steps() {
  local autonomy_file="$PROJECT_ROOT/clare/autonomy.yml"
  STAGE_5_STEPS=()

  if [[ ! -f "$autonomy_file" ]]; then
    STAGE_5_STEPS+=("autonomy.yml|check_autonomy_file")
    return 0
  fi

  # First step verifies the file is present (once); boundary steps that follow
  # only check their own level, so "autonomy.yml found" isn't repeated per path.
  STAGE_5_STEPS+=("autonomy.yml|check_autonomy_file")

  local boundary_path="" boundary_level=""

  append_pending_autonomy_step() {
    if [[ -n "$boundary_path" ]]; then
      local label="$boundary_path"
      if [[ -n "$boundary_level" ]]; then
        label+=" ($boundary_level)"
      fi
      STAGE_5_STEPS+=("$label|check_autonomy_boundary|$boundary_path")
    fi
  }

  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*path:[[:space:]]*(.*) ]]; then
      append_pending_autonomy_step
      boundary_path="$(strip_yaml_scalar_quotes "$(trim_whitespace "${BASH_REMATCH[1]}")")"
      boundary_level=""
      continue
    fi

    if [[ -n "$boundary_path" && "$line" =~ ^[[:space:]]*level:[[:space:]]*(.*) ]]; then
      boundary_level="$(strip_yaml_scalar_quotes "$(trim_whitespace "${BASH_REMATCH[1]}")")"
    fi
  done <"$autonomy_file"

  append_pending_autonomy_step
}

check_autonomy_file() {
  local autonomy_file="$PROJECT_ROOT/clare/autonomy.yml"
  if [[ -f "$autonomy_file" ]]; then
    pass "autonomy.yml found"
  else
    warn "clare/autonomy.yml not found — run the CLARE installer to configure CLARE"
  fi
}

autonomy_level_for_path() {
  local requested_path="$1"
  local autonomy_file="$PROJECT_ROOT/clare/autonomy.yml"
  local boundary_path="" boundary_level=""

  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*path:[[:space:]]*(.*) ]]; then
      if [[ "$boundary_path" == "$requested_path" ]]; then
        printf '%s' "$boundary_level"
        return 0
      fi
      boundary_path="$(strip_yaml_scalar_quotes "$(trim_whitespace "${BASH_REMATCH[1]}")")"
      boundary_level=""
      continue
    fi

    if [[ -n "$boundary_path" && "$line" =~ ^[[:space:]]*level:[[:space:]]*(.*) ]]; then
      boundary_level="$(strip_yaml_scalar_quotes "$(trim_whitespace "${BASH_REMATCH[1]}")")"
    fi
  done <"$autonomy_file"

  if [[ "$boundary_path" == "$requested_path" ]]; then
    printf '%s' "$boundary_level"
  fi
}

check_humans_only_boundary() {
  local boundary_path="$1"

  if [[ "$boundary_path" == "*" ]]; then
    pass "Humans-only boundary configured: $boundary_path"
    return 0
  fi

  if ! command -v git &>/dev/null || ! git -C "$PROJECT_ROOT" rev-parse --git-dir &>/dev/null; then
    pass "Humans-only boundary configured: $boundary_path"
    return 0
  fi

  local staged_files
  staged_files=$(git -C "$PROJECT_ROOT" diff --cached --name-only 2>/dev/null || echo "")
  if [[ -z "$staged_files" ]]; then
    pass "Humans-only boundary clear: $boundary_path"
    return 0
  fi

  local staged matched=false
  while IFS= read -r staged; do
    [[ -z "$staged" ]] && continue
    if [[ "$staged" == "$boundary_path" || "$staged" == "$boundary_path/"* ]]; then
      matched=true
      warn "Staged file matches humans-only path: $boundary_path ($staged)"
      warn "Review clare/autonomy.yml before committing AI-generated changes to this path."
    fi
  done <<<"$staged_files"

  if ! $matched; then
    pass "Humans-only boundary clear: $boundary_path"
  fi
}

check_autonomy_boundary() {
  local boundary_path="$1"

  local boundary_level
  boundary_level="$(autonomy_level_for_path "$boundary_path")"
  if [[ -z "$boundary_level" ]]; then
    warn "Autonomy boundary not found: $boundary_path"
    return 0
  fi

  if [[ "$boundary_level" == "humans-only" ]]; then
    check_humans_only_boundary "$boundary_path"
  else
    pass "Autonomy boundary configured: $boundary_path ($boundary_level)"
  fi
}

# ─── Extensions ──────────────────────────────────────────────────────────────
# Optional tool extensions configured in clare/extensions.yml.
# Each extension wraps an external tool. If enabled but not installed,
# verify-ci.sh fails with install instructions (never auto-installs).

strip_yaml_scalar_quotes() {
  local value="$1"

  if [[ "$value" == \"*\" && "$value" == *\" ]]; then
    value="${value#\"}"
    value="${value%\"}"
  elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
    value="${value#\'}"
    value="${value%\'}"
  fi

  printf '%s' "$value"
}

print_install_hint() {
  local prefix="$1"
  local hint="$2"

  hint="${hint//\\n/$'\n'}"

  local first_line=true
  while IFS= read -r hint_line || [[ -n "$hint_line" ]]; do
    if $first_line; then
      echo "  ${prefix}${hint_line}"
      first_line=false
    else
      echo "  ${hint_line}"
    fi
  done <<<"$hint"
}

refresh_extension_steps() {
  local extensions_file="$PROJECT_ROOT/clare/extensions.yml"
  STAGE_6_STEPS=()
  [[ -f "$extensions_file" ]] || return 0

  local in_extension=false
  local ext_name="" ext_enabled=""

  append_pending_extension_step() {
    if [[ -n "$ext_name" && "$ext_enabled" == "true" ]]; then
      STAGE_6_STEPS+=("$ext_name|check_extension_by_name|$ext_name")
    fi
  }

  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*name:[[:space:]]*(.*) ]]; then
      append_pending_extension_step
      ext_name="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
      ext_enabled=""
      in_extension=true
      continue
    fi

    $in_extension || continue

    if [[ "$line" =~ ^[[:space:]]*enabled:[[:space:]]*(.*) ]]; then
      ext_enabled="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    fi
  done <"$extensions_file"

  append_pending_extension_step
}

check_extension_by_name() {
  local requested_name="$1"
  local extensions_file="$PROJECT_ROOT/clare/extensions.yml"
  [[ -f "$extensions_file" ]] || return 0

  local in_extension=false
  local ext_name="" ext_enabled="" ext_command="" ext_install="" ext_url=""
  local ext_threshold="" ext_paths="" ext_extra="" ext_file_types="" ext_exclude=""
  local ext_count_comments="true"

  process_pending_extension() {
    if [[ "$ext_name" == "$requested_name" && "$ext_enabled" == "true" ]]; then
      run_extension "$ext_name" "$ext_command" "$ext_install" "$ext_url" \
        "$ext_threshold" "$ext_paths" "$ext_extra" \
        "$ext_file_types" "$ext_exclude" "$ext_count_comments"
    fi
  }

  while IFS= read -r line; do
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*name:[[:space:]]*(.*) ]]; then
      process_pending_extension
      ext_name="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
      ext_enabled=""
      ext_command=""
      ext_install=""
      ext_url=""
      ext_threshold=""
      ext_paths=""
      ext_extra=""
      ext_file_types=""
      ext_exclude=""
      ext_count_comments="true"
      in_extension=true
      continue
    fi

    $in_extension || continue

    if [[ "$line" =~ ^[[:space:]]*enabled:[[:space:]]*(.*) ]]; then
      ext_enabled="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*command:[[:space:]]*(.*) ]]; then
      ext_command="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*install_hint:[[:space:]]*(.*) ]]; then
      ext_install="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*project_url:[[:space:]]*(.*) ]]; then
      ext_url="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*threshold:[[:space:]]*(.*) ]]; then
      ext_threshold="${BASH_REMATCH[1]}"
    elif [[ "$line" =~ ^[[:space:]]*paths:[[:space:]]*(.*) ]]; then
      ext_paths="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*extra_flags:[[:space:]]*(.*) ]]; then
      ext_extra="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*file_types:[[:space:]]*(.*) ]]; then
      ext_file_types="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*exclude:[[:space:]]*(.*) ]]; then
      ext_exclude="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    elif [[ "$line" =~ ^[[:space:]]*count_comments:[[:space:]]*(.*) ]]; then
      ext_count_comments="$(strip_yaml_scalar_quotes "${BASH_REMATCH[1]}")"
    fi
  done <"$extensions_file"

  process_pending_extension
}

run_extension() {
  local name="$1" command="$2" install_hint="$3" url="$4"
  local threshold="$5" paths="$6" extra="$7"
  local file_types="${8:-}" exclude="${9:-}" count_comments="${10:-true}"

  section "Extension: $name"

  # Check if the tool is installed
  if ! command -v "$command" &>/dev/null; then
    local ext_id="Extension: $name"
    fail "$ext_id"
    local out_file
    out_file="$(mktemp)"
    {
      echo ""
      echo "  '$name' is enabled in clare/extensions.yml but '$command' is not installed."
      echo ""
      print_install_hint "To install:  " "$install_hint"
      [[ -n "$url" ]] && echo "  Project:     $url"
      echo ""
      echo "  To disable this extension:"
      echo "    Edit clare/extensions.yml and set enabled: false for '$name'"
      echo ""
    } >"$out_file"
    FAILED_OUTPUTS["$ext_id"]="$out_file"
    CHECK_COMMANDS["$ext_id"]="Install: $install_hint"
    # Note: fail "$ext_id" above already appended to FAILED_CHECKS; do not append again.
    if $FAIL_FAST; then
      print_failure_summary
      exit 1
    fi
    cat "$out_file"
    return 1
  fi

  # Build and run the extension command
  case "$name" in
    eslint-complexity)
      run_eslint_complexity_check "$command" "$threshold" "$paths" "$extra" "$file_types" "$exclude"
      ;;
    golangci-lint-complexity)
      run_golangci_lint_complexity_check "$command" "$threshold" "$paths" "$extra" "$file_types" "$exclude"
      ;;
    complexipy-complexity)
      run_complexipy_check "$command" "$threshold" "$paths" "$extra" "$file_types" "$exclude"
      ;;
    shellmetrics-complexity)
      run_shellmetrics_check "$command" "$threshold" "$paths" "$extra" "$file_types" "$exclude"
      ;;
    file-size)
      run_file_size_check "$threshold" "$paths" "$file_types" "$exclude" "$count_comments"
      ;;
    *)
      warn "Unknown extension '$name' — skipping (no built-in handler)"
      warn "Add a handler in verify-ci.sh or use verify-local.sh for custom checks"
      ;;
  esac
}

# line_comment_prefix ext
#   Returns the line-comment prefix for a file extension, or empty string if
#   the language uses only block comments (e.g. CSS) or is unrecognised.
line_comment_prefix() {
  local ext="$1"
  case "$ext" in
    js | jsx | ts | tsx | java | kt | kts | scala | go | rs | swift | c | cc | cpp | cxx | h | hpp | cs | dart | groovy | gradle)
      echo "//"
      ;;
    py | rb | sh | bash | zsh | fish | r | pl | perl | tcl | yaml | yml | toml | makefile | mk | dockerfile)
      echo "#"
      ;;
    lua)
      echo "--"
      ;;
    *)
      echo ""
      ;;
  esac
}

# count_code_lines filepath [count_comments]
#   Counts lines in a file. When count_comments=false, pure comment-only lines
#   (leading whitespace + comment prefix + anything) are excluded. A line that
#   has code followed by a comment is NOT excluded. Blank lines are always
#   counted. Block comments spanning multiple lines are not stripped.
count_code_lines() {
  local filepath="$1"
  local count_comments="${2:-true}"

  if [[ "$count_comments" == "true" ]]; then
    wc -l <"$filepath"
    return
  fi

  local ext="${filepath##*.}"
  ext="${ext,,}"
  local prefix
  prefix="$(line_comment_prefix "$ext")"

  if [[ -z "$prefix" ]]; then
    # No known line-comment prefix for this type; count all lines
    wc -l <"$filepath"
    return
  fi

  # Escape prefix for use in sed (// → \/\/)
  local escaped_prefix
  escaped_prefix="$(printf '%s' "$prefix" | sed 's/[\/&]/\\&/g')"

  # Strip lines that are ONLY a comment: optional whitespace, then the prefix
  sed "/^[[:space:]]*${escaped_prefix}/d" "$filepath" | wc -l
}

run_file_size_check() {
  local max_lines="${1:-300}"
  local scan_paths="${2:-src}"
  local file_types="${3:-js ts tsx jsx}"
  local exclude_patterns="${4:-}"
  local count_comments="${5:-true}"
  local refactor_target_lines=$((max_lines * 9 / 10))
  local effective_scan_paths="$scan_paths"
  local oversized_files=()
  local oversized_counts=()
  local checked_files=0

  if ! scan_paths_exist "$scan_paths"; then
    effective_scan_paths="."
    warn "File size check: configured paths not found (paths: $scan_paths); falling back to project root"
  fi

  local count_label="lines"
  [[ "$count_comments" == "false" ]] && count_label="non-comment lines"

  while IFS= read -r rel_path; do
    [[ -z "$rel_path" ]] && continue
    is_default_ignored_path "$rel_path" && continue
    path_within_scan_paths "$rel_path" "$effective_scan_paths" || continue
    matches_any_file_type "$rel_path" "$file_types" || continue
    is_extension_excluded "$rel_path" "$exclude_patterns" && continue

    local filepath="$PROJECT_ROOT/$rel_path"
    [[ -f "$filepath" ]] || continue

    local line_count
    line_count=$(count_code_lines "$filepath" "$count_comments")
    ((checked_files += 1))

    if [[ "$line_count" -gt "$max_lines" ]]; then
      oversized_files+=("$rel_path")
      oversized_counts+=("$line_count")
    fi
  done < <(list_project_files_respecting_gitignore)

  if [[ "$checked_files" -eq 0 ]]; then
    warn "File size check: no files matched configured paths/types (paths: $effective_scan_paths, types: $file_types)"
    return 0
  fi

  if [[ ${#oversized_files[@]} -eq 0 ]]; then
    pass "File size (all files under $max_lines $count_label)"
  else
    fail "File size (${#oversized_files[@]} file(s) exceed $max_lines $count_label)"
    echo ""
    for i in "${!oversized_files[@]}"; do
      echo -e "${RED}   ${oversized_files[$i]}: ${oversized_counts[$i]} $count_label (max: $max_lines)${NC}"
    done
    echo ""
    echo -e "${YELLOW}   Refactor oversized files structurally to <= $refactor_target_lines $count_label (90% of threshold).${NC}"
    echo -e "${YELLOW}   Use extraction/splitting/simplification; do not delete comments or whitespace to game line counts.${NC}"
    echo -e "${YELLOW}   Adjust the threshold in clare/extensions.yml if needed.${NC}"
    echo ""
  fi
}

# Suggest a quick-fix command for a failing check name or fallback to the original command
suggest_quick_command() {
  local name="$1"
  local cmd="$2"
  case "$name" in
    *ESLint* | *eslint*)
      echo "cd '$PROJECT_ROOT' && npx eslint --fix ."
      ;;
    *Prettier* | *prettier*)
      echo "cd '$PROJECT_ROOT' && npx prettier --write ."
      ;;
    *TypeScript* | *tsc*)
      echo "cd '$PROJECT_ROOT' && npx tsc --noEmit"
      ;;
    *Ruff* | *ruff*)
      echo "cd '$PROJECT_ROOT' && ruff check --fix ."
      ;;
    *Mypy* | *mypy*)
      echo "cd '$PROJECT_ROOT' && mypy ."
      ;;
    *pytest* | *Pytest*)
      echo "cd '$PROJECT_ROOT' && pytest -q"
      ;;
    *'npm test'*)
      echo "cd '$PROJECT_ROOT' && npm test"
      ;;
    *Go* | *golang*)
      echo "cd '$PROJECT_ROOT' && go test ./..."
      ;;
    *Cargo* | *cargo*)
      echo "cd '$PROJECT_ROOT' && cargo test"
      ;;
    *)
      # Fallback: show the original command (clean up obvious redirections)
      echo "${cmd// 2>&1/}"
      ;;
  esac
}

print_failure_summary() {
  local lines="$SUMMARY_LINES"
  # Nothing to summarize when no checks failed. The guard also keeps the
  # "${FAILED_CHECKS[@]}" expansion safe under `set -u` on bash < 4.4, where
  # expanding an empty array would otherwise be an unbound-variable error.
  [[ ${#FAILED_CHECKS[@]} -gt 0 ]] || return 0
  echo ""
  echo -e "${BLUE}════════════════════════════════════════════${NC}"
  echo -e "${RED}FAILURE SUMMARY:${NC}"
  for name in "${FAILED_CHECKS[@]}"; do
    local out="${FAILED_OUTPUTS[$name]:-}"
    local cmd="${CHECK_COMMANDS[$name]:-}"
    echo -e "${RED}• $name${NC}"
    local suggestion
    suggestion="$(suggest_quick_command "$name" "$cmd")"
    echo -e "  Quick command: ${YELLOW}${suggestion}${NC}"
    if [[ -n "$out" && -f "$out" ]]; then
      echo "  Output (first $lines lines):"
      sed -n "1,${lines}p" "$out" | sed 's/^/    /'
      echo "  Full log: $out"
    else
      echo "  No captured output available."
    fi
    echo ""
  done
}

# ─── Local Project Checks ────────────────────────────────────────────────────
# Source verify-local.sh if it exists. That file is project-owned (never
# overwritten by CLARE updates) and can call run_check, pass, fail, info,
# warn, section, and read FAST_MODE / FIX_MODE / PROJECT_ROOT.

source_local_checks() {
  if [[ -f "$SCRIPT_DIR/verify-local.sh" ]]; then
    section "Project-Specific Checks (verify-local.sh)"
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/verify-local.sh"
  fi
}

# ─── CLARE₂ Session Signal Emission ──────────────────────────────────────────
#
# These functions emit structured JSONL records to CLARE2_SESSION_FILE when that
# variable is set. Zero-cost when unset — all writes go to /dev/null.
#
# Three signal types (per CLARE₂.md §1.7):
#   ci_result   — structured pass/fail for every check run
#   correction  — auto-emitted when a previously failing check now passes
#   file_tier   — autonomy tier for each file changed in this session

# Temp file tracking which checks failed on the *previous* run (for correction detection)
CLARE2_LAST_FAILURES_FILE="${TMPDIR:-/tmp}/.clare1_last_failures_$$"

# Resolve the autonomy tier for a given file path by reading autonomy.yml.
# Requires yq v4; silently returns "unknown" if yq is unavailable.
_clare2_autonomy_tier() {
  local file_path="$1"
  local autonomy_yml="${SCRIPT_DIR}/autonomy.yml"

  if ! command -v yq &>/dev/null || [[ ! -f "$autonomy_yml" ]]; then
    echo "unknown"
    return
  fi

  local rel_path
  rel_path=$(realpath --relative-to="$PROJECT_ROOT" "$file_path" 2>/dev/null || echo "$file_path")

  # Walk module entries from most-specific to least-specific match
  local tier
  tier=$(yq e '.modules[] | select(.path == "'"$rel_path"'") | .level' "$autonomy_yml" 2>/dev/null | head -1)
  if [[ -n "$tier" && "$tier" != "null" ]]; then
    echo "$tier"
    return
  fi

  # Try prefix match (directory-level): find entries whose path is a prefix
  local best_match=""
  local best_len=0
  while IFS= read -r entry_path; do
    if [[ "$rel_path" == "$entry_path"* ]] && ((${#entry_path} > best_len)); then
      best_match="$entry_path"
      best_len=${#entry_path}
    fi
  done < <(yq e '.modules[].path' "$autonomy_yml" 2>/dev/null)

  if [[ -n "$best_match" ]]; then
    yq e '.modules[] | select(.path == "'"$best_match"'") | .level' "$autonomy_yml" 2>/dev/null | head -1
  else
    echo "unknown"
  fi
}

# Emit a ci_result record and handle correction detection.
# Called at the end of main() with the final exit code.
_clare2_emit_ci_result() {
  local exit_code="$1"
  local session_file="${CLARE2_SESSION_FILE:-}"
  [[ -z "$session_file" ]] && return 0

  local ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  # Build the checks array: one entry per check, pass or fail
  local checks_json="["
  local first=true
  for check in "${!CHECK_COMMANDS[@]}"; do
    local status="pass"
    for failed in "${FAILED_CHECKS[@]}"; do
      [[ "$failed" == "$check" ]] && status="fail" && break
    done
    $first || checks_json+=","
    first=false
    checks_json+="{\"name\":$(printf '%s' "$check" | jq -Rs .),\"status\":\"$status\",\"cmd\":$(printf '%s' "${CHECK_COMMANDS[$check]}" | jq -Rs .)}"
  done
  checks_json+="]"

  local current_failures_list
  current_failures_list=$(printf '%s\n' "${FAILED_CHECKS[@]}" | sort)

  # Correction detection: compare to previous run's failures
  if [[ -f "$CLARE2_LAST_FAILURES_FILE" ]]; then
    local resolved
    resolved=$(comm -23 \
      <(cat "$CLARE2_LAST_FAILURES_FILE" | sort) \
      <(printf '%s\n' "${FAILED_CHECKS[@]}" | sort) \
      2>/dev/null || true)
    if [[ -n "$resolved" ]]; then
      while IFS= read -r check; do
        [[ -z "$check" ]] && continue
        local corr_ts
        corr_ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        printf '%s\n' \
          "{\"type\":\"correction\",\"correction_type\":\"ci_self_correct\",\"check\":$(printf '%s' "$check" | jq -Rs .),\"ts\":\"$corr_ts\"}" \
          >>"$session_file"
      done <<<"$resolved"
    fi
  fi

  # Save current failures for next run
  printf '%s\n' "${FAILED_CHECKS[@]}" | sort >"$CLARE2_LAST_FAILURES_FILE"

  # Emit the ci_result record
  printf '%s\n' \
    "{\"type\":\"ci_result\",\"exit_code\":$exit_code,\"checks\":$checks_json,\"ts\":\"$ts\"}" \
    >>"$session_file"

  # Emit file_tier records for changed files
  if command -v git &>/dev/null && git -C "$PROJECT_ROOT" rev-parse --git-dir &>/dev/null; then
    local changed_files
    while IFS= read -r f; do
      [[ -z "$f" ]] && continue
      local abs_path="${PROJECT_ROOT}/${f}"
      local tier
      tier=$(_clare2_autonomy_tier "$abs_path")
      printf '%s\n' \
        "{\"type\":\"file_tier\",\"file\":$(printf '%s' "$f" | jq -Rs .),\"tier\":\"$tier\",\"ts\":\"$ts\"}" \
        >>"$session_file"
    done < <(git -C "$PROJECT_ROOT" diff --name-only HEAD 2>/dev/null)
  fi
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
  refresh_autonomy_steps
  refresh_extension_steps

  if $LIST_TESTS; then
    print_stage_list
    exit 0
  fi

  if $RUN_SELECTED_STAGES; then
    parse_selected_stages "$RUN_TESTS_CSV"
  fi

  echo -e "${BLUE}════════════════════════════════════════════${NC}"
  echo -e "${BLUE}  CLARE — Local CI/CD Verification          ${NC}"
  echo -e "${BLUE}════════════════════════════════════════════${NC}"

  cd "$PROJECT_ROOT"
  detect_project

  # Compute Node tool flags once, in the main shell, so the (potentially
  # node-spawning) Turboshaft probe runs at most once. Calls from inside command
  # substitutions later only read the already-populated global.
  $HAS_NODE && init_node_tool_flags

  info "Project root: $PROJECT_ROOT"
  $HAS_NODE && info "Detected: Node.js"
  $HAS_PYTHON && info "Detected: Python"
  $HAS_GO && info "Detected: Go"
  $HAS_RUST && info "Detected: Rust"
  $FAST_MODE && warn "Fast mode: architecture tests skipped"
  $FIX_MODE && warn "Fix mode: auto-fixing lint issues where possible"
  $FAIL_FAST || warn "Fail-slow mode: continuing past failures"
  $RUN_SELECTED_STAGES && info "Running selected stages: $RUN_TESTS_CSV"
  if $INCLUDE_UNTRACKED; then
    info "File scanning includes untracked files (use --exclude-untracked to limit to tracked files)"
  else
    info "File scanning excludes untracked files"
  fi

  if $RUN_SELECTED_STAGES; then
    run_selected_tests
  else
    local stage_index
    for stage_index in "${!STAGES[@]}"; do
      run_stage "$((stage_index + 1))"
    done
  fi

  echo ""
  echo -e "${BLUE}════════════════════════════════════════════${NC}"
  if [[ ${#FAILED_CHECKS[@]} -eq 0 ]]; then
    echo -e "${GREEN}✅ All checks passed — work is complete.${NC}"
    echo -e "${GREEN}   No further terminal input is required; you can proceed to commit.${NC}"
    echo ""
    _clare2_emit_ci_result 0
    exit 0
  else
    echo -e "${RED}❌ ${#FAILED_CHECKS[@]} check(s) failed:${NC}"
    for check in "${FAILED_CHECKS[@]}"; do
      echo -e "${RED}   • $check${NC}"
    done
    echo ""
    # Print compact failure summary with quick-fix suggestions and snippets
    print_failure_summary
    echo -e "${RED}   Fix the issues above and run this script again.${NC}"
    echo -e "${RED}   Work is NOT complete until all checks pass.${NC}"
    _clare2_emit_ci_result 1
    exit 1
  fi
}

main "$@"
