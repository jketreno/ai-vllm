#!/usr/bin/env bash
set -euo pipefail

# Simple environment scanner for CLARE
# Outputs a human-readable report or JSON describing file types, config
# files found, and tools referenced by clare/verify-*.sh.

usage() {
  cat <<'USAGE'
Usage: clare-env-scan.sh [--report|--json] [--apply-extensions] [--vscode-dir <path>]

Options:
  --report            Print human-readable report (default)
  --json              Emit JSON to stdout
  --apply-extensions  Attempt to write recommended extensions to <vscode-dir>/extensions.json
  --vscode-dir <dir>  Directory containing VS Code config (default: .vscode)
  --help              Show this help
USAGE
}

MODE=report
APPLY_EXT=false
VSCODE_DIR=".vscode"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report)
      MODE=report
      shift
      ;;
    --json)
      MODE=json
      shift
      ;;
    --apply-extensions)
      APPLY_EXT=true
      shift
      ;;
    --vscode-dir)
      # Guard the value: a trailing `--vscode-dir` with no argument would make
      # `shift 2` fail on a single-element list under `set -e`, exiting with no
      # message. Require an explicit value instead.
      if [[ $# -lt 2 || "$2" == -* ]]; then
        echo "--vscode-dir requires a value" >&2
        usage
        exit 2
      fi
      VSCODE_DIR="$2"
      shift 2
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$PROJECT_ROOT"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

# 1) File counts by extension
declare -A counts
while IFS= read -r -d '' f; do
  base="$(basename "$f")"
  if [[ "$base" != *.* ]]; then
    ext="no_extension"
  else
    ext=".${base##*.}"
  fi
  counts["$ext"]=$((${counts["$ext"]:-0} + 1))
  # Prune vendored/build directories so file counts characterize the project
  # itself, not its dependencies (e.g. node_modules can dwarf real source files).
done < <(find . \
  \( -path './.git' -o -path './node_modules' -o -path './dist' \
  -o -path './build' -o -path './.venv' -o -path './venv' -o -path './vendor' \) -prune \
  -o -type f -print0)

# Write counts to temp file for JSON stage
echo "{" >"$TMPDIR/fileCounts.json"
first=true
for k in "${!counts[@]}"; do
  if [[ "$first" == true ]]; then first=false; else echo "," >>"$TMPDIR/fileCounts.json"; fi
  printf '  "%s": %d' "${k}" "${counts[$k]}" >>"$TMPDIR/fileCounts.json"
done
printf '\n}\n' >>"$TMPDIR/fileCounts.json"

# 2) Detect key config files
configs=(
  package.json
  package-lock.json
  yarn.lock
  tsconfig.json
  pyproject.toml
  requirements.txt
  setup.cfg
  Dockerfile
  docker-compose.yml
  .github/workflows/ci.yml
  clare/verify-ci.sh
  clare/verify-local.sh
  clare/autonomy.yml
  clare/extensions.yml
)
: >"$TMPDIR/configs.txt"
for c in "${configs[@]}"; do
  if [[ -f "$PROJECT_ROOT/$c" || -d "$PROJECT_ROOT/$c" ]]; then
    echo "$c" >>"$TMPDIR/configs.txt"
  fi
done

# 3) Parse verify scripts for known tool tokens
verify_files=("clare/verify-ci.sh" "clare/verify-local.sh")
: >"$TMPDIR/tools.txt"
for vf in "${verify_files[@]}"; do
  if [[ -f "$PROJECT_ROOT/$vf" ]]; then
    # Extract tokens of likely tools. The `|| true` is scoped so the rest of
    # the pipeline always runs on grep's output (matches go to the transform);
    # without the braces, `|| true` would short-circuit the whole pipe.
    #
    # Tokens are wrapped in \b…\b word boundaries so short names like `go` and
    # `git` don't match inside unrelated words (e.g. `golang`, `digit`). Longer
    # alternatives (golangci-lint, python3, ripgrep) are listed before their
    # shorter prefixes so leftmost-first alternation prefers the specific match.
    { grep -Eio '\b(golangci-lint|golint|ripgrep|typescript|shellcheck|shellmetrics|complexipy|prettier|pyyaml|python3|pytest|flake8|eslint|docker|cargo|shfmt|python|pipx|mypy|tput|rust|ruff|node|npm|npx|tsc|pip|yq|rg|go|jq|git)\b' "$PROJECT_ROOT/$vf" 2>/dev/null || true; } \
      | tr '[:upper:]' '[:lower:]' | tr -s '[:space:]' '\n' | sort -u | while read -r token; do
      [[ -z "$token" ]] && continue
      echo "$token|$vf"
    done
  fi
done | sort -u >"$TMPDIR/tools_raw.txt"

# Aggregate to one row per tool, joining every verify file it appeared in into a
# single comma-separated whereFound list. Without this, a tool referenced in both
# verify scripts would emit two separate rows, doubling the report.
awk -F'|' '
  { if ($1 in seen) { files[$1] = files[$1] "," $2 } else { seen[$1] = 1; files[$1] = $2; order[++n] = $1 } }
  END { for (i = 1; i <= n; i++) print order[i] "|" files[order[i]] }
' "$TMPDIR/tools_raw.txt" >"$TMPDIR/tools.txt"

# Helper: map tool -> install hint + includeInDocker
map_tool() {
  local t="$1"
  # Use a small Python mapping here to keep shell cyclomatic complexity low
  python3 - "$t" <<'PY'
import sys
t = sys.argv[1]
m = {
  'node': "Install Node.js (NodeSource or official distro image). Use npm ci to install project devDependencies.",
  'npm': "Install Node.js (NodeSource or official distro image). Use npm ci to install project devDependencies.",
  'npx': "Install Node.js (NodeSource or official distro image). Use npm ci to install project devDependencies.",
  'eslint': "npm install --save-dev eslint (or use project devDependencies)",
  'prettier': "npm install --save-dev prettier",
  'typescript': "npm install --save-dev typescript",
  'tsc': "npm install --save-dev typescript",
  'yq': "Install mikefarah yq binary (https://github.com/mikefarah/yq) or use distro package",
  'python3': "Install Python 3 and pip (apt: python3 python3-pip). Use pipx for global CLIs.",
  'python': "Install Python 3 and pip (apt: python3 python3-pip). Use pipx for global CLIs.",
  'pip': "Install Python 3 and pip (apt: python3 python3-pip). Use pipx for global CLIs.",
  'pipx': "Install Python 3 and pip (apt: python3 python3-pip). Use pipx for global CLIs.",
  'pyyaml': "pip install pyyaml",
  'shellcheck': "apt-get install shellcheck (or use distro package)",
  'shfmt': "Install shfmt (go install mvdan.cc/sh/v3/cmd/shfmt@latest or download binary)",
  'rg': "apt-get install ripgrep (optional)",
  'ripgrep': "apt-get install ripgrep (optional)",
  'ruff': "pip install ruff",
  'flake8': "pip install flake8",
  'mypy': "pip install mypy",
  'pytest': "pip install pytest",
  'go': "Install Go toolchain (apt or official image) and golangci-lint via 'go install'",
  'golangci-lint': "Install Go toolchain (apt or official image) and golangci-lint via 'go install'",
  'golint': "Install Go toolchain (apt or official image) and golangci-lint via 'go install'",
  'cargo': "Install Rust via rustup (if you need Rust checks)",
  'rust': "Install Rust via rustup (if you need Rust checks)",
  'jq': "apt-get install jq",
  'docker': "Install Docker CLI if you intend to run containers locally",
  'tput': "Provided by ncurses (apt-get install ncurses-bin)",
  'git': "Install git (apt-get install git)",
  'shellmetrics': "Project-specific Python packages; verify exact package name in clare/extensions.yml",
  'complexipy': "Project-specific Python packages; verify exact package name in clare/extensions.yml",
}
print(m.get(t, f"Check project files or verify scripts for how {t} is used"))
PY
}

# 4) Build verifyTools JSON structure
: >"$TMPDIR/verifyTools.jsonl"
if [[ -f "$TMPDIR/tools.txt" ]]; then
  while IFS='|' read -r tool where; do
    [[ -z "$tool" ]] && continue
    hint="$(map_tool "$tool")"
    include="false"
    case "$tool" in
      node | npm | npx | eslint | prettier | typescript | tsc | yq | python3 | python | pip | jq | shellcheck | shfmt | git | tput)
        include="true"
        ;;
    esac
    python3 - "$tool" "$where" "$hint" "$include" <<'PY' >>"$TMPDIR/verifyTools.jsonl"
import json, sys
obj = {'tool': sys.argv[1], 'whereFound': sys.argv[2].split(','), 'installHint': sys.argv[3], 'includeInDocker': sys.argv[4] == 'true'}
print(json.dumps(obj))
PY
  done <"$TMPDIR/tools.txt"
fi

# 5) Recommended VS Code extensions (basic set)
cat >"$TMPDIR/recommended_extensions.json" <<JSON
[
  {"extension":"eamodio.gitlens","reason":"Git history and blame"},
  {"extension":"EditorConfig.EditorConfig","reason":"Enforce editor settings"},
  {"extension":"dbaeumer.vscode-eslint","reason":"ESLint integration"},
  {"extension":"esbenp.prettier-vscode","reason":"Formatting with Prettier"},
  {"extension":"timonwong.shellcheck","reason":"ShellCheck in-editor linting"},
  {"extension":"redhat.vscode-yaml","reason":"YAML schema and validation"},
  {"extension":"ms-python.python","reason":"Python language support (optional)"},
  {"extension":"ms-azuretools.vscode-docker","reason":"Dockerfile authoring"},
  {"extension":"yzhang.markdown-all-in-one","reason":"Markdown productivity"}
]
JSON

# Write the recommended extensions to <vscode-dir>/extensions.json. Kept in its
# own function so it runs in both --json and --report modes; otherwise passing
# --apply-extensions alongside --json would be a silent no-op.
apply_recommended_extensions() {
  mkdir -p "$VSCODE_DIR"
  local extfile="$VSCODE_DIR/extensions.json"
  if [[ -f "$extfile" ]]; then
    echo "Backing up existing $extfile -> $extfile.bak" >&2
    cp "$extfile" "$extfile.bak"
  fi
  EXTFILE="$extfile" RECOMMENDED="$TMPDIR/recommended_extensions.json" python3 - <<'PY'
import json, os, sys
extfile = os.environ['EXTFILE']
rec = json.load(open(os.environ['RECOMMENDED']))
with open(extfile, 'w') as f:
    json.dump({'recommendations': [r['extension'] for r in rec]}, f, indent=2)
print('Wrote', extfile, file=sys.stderr)
PY
}

# 6) Emit results
if [[ "$APPLY_EXT" == true ]]; then
  apply_recommended_extensions
fi

if [[ "$MODE" == "json" ]]; then
  TMPDIR="$TMPDIR" python3 - <<'PY'
import json,os
tmp=os.environ['TMPDIR']
with open(os.path.join(tmp,'fileCounts.json')) as f:
  fileCounts=json.load(f)
configs=[]
configs_path=os.path.join(tmp,'configs.txt')
if os.path.exists(configs_path):
  with open(configs_path) as f:
    configs=[l.strip() for l in f if l.strip()]
verifyTools=[]
verify_path=os.path.join(tmp,'verifyTools.jsonl')
if os.path.exists(verify_path):
  for line in open(verify_path):
    try:
      verifyTools.append(json.loads(line))
    except Exception:
      pass
recommended=json.load(open(os.path.join(tmp,'recommended_extensions.json')))
out={'fileCounts': fileCounts, 'configsFound': configs, 'verifyTools': verifyTools, 'recommendedExtensions': recommended}
print(json.dumps(out, indent=2))
PY
  exit 0
fi

# Human readable report
echo "CLARE environment scan for: $PROJECT_ROOT"
echo ""
echo "File counts (by extension):"
TMPDIR="$TMPDIR" python3 - <<'PY'
import json, os
counts=json.load(open(os.path.join(os.environ['TMPDIR'],'fileCounts.json')))
print('\n'.join([f"  {k}: {v}" for k,v in counts.items()]))
PY

echo ""
if [[ -s "$TMPDIR/configs.txt" ]]; then
  echo "Detected config files:"
  sed -e 's/^/  - /' "$TMPDIR/configs.txt" || true
else
  echo "No common config files detected"
fi

echo ""
if [[ -s "$TMPDIR/verifyTools.jsonl" ]]; then
  echo "Tools referenced by verify scripts:"
  TMPDIR="$TMPDIR" python3 - <<'PY'
import json, os
for line in open(os.path.join(os.environ['TMPDIR'],'verifyTools.jsonl')):
  try:
    obj=json.loads(line)
    print(f"  - {obj['tool']}: {obj['installHint']} (includeInDocker={obj['includeInDocker']})")
  except Exception:
    pass
PY
else
  echo "No tools referenced in clare/verify-*.sh found."
fi

echo ""
echo "Recommended VS Code extensions:"
TMPDIR="$TMPDIR" python3 - <<'PY'
import json, os
for e in json.load(open(os.path.join(os.environ['TMPDIR'],'recommended_extensions.json'))):
  print(f"  - {e['extension']}: {e['reason']}")
PY

exit 0
