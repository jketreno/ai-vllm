#!/usr/bin/env bash
set -euo pipefail

# Manages clare2/pipeline/config/corpus_sources.yml: list, add, remove, and
# manually sync remote hosts that the nightly corpus_sync job pulls
# sessions/ from.
#
# Usage:
#   clare2-corpus-manage.sh [-q|--quiet] list
#   clare2-corpus-manage.sh [-q|--quiet] subscribe user@host[:port] [remote_corpus_root]
#   clare2-corpus-manage.sh [-q|--quiet] unsubscribe user@host
#   clare2-corpus-manage.sh [-q|--quiet] sync

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SOURCES_FILE="${CLARE2_CORPUS_SOURCES_FILE:-${REPO_ROOT}/clare2/pipeline/config/corpus_sources.yml}"
SYNC_KEY="${CLARE2_CORPUS_SYNC_KEY_FILE:-${REPO_ROOT}/secrets/clare2_corpus_sync_key}"
SYNC_PUB_KEY="${SYNC_KEY}.pub"
CORPUS_ROOT="${CLARE2_CORPUS_ROOT:-${REPO_ROOT}/corpus}"
DEFAULT_REMOTE_ROOT="~/.config/clare/corpus"
DEFAULT_PORT=22

usage() {
  cat <<'EOF'
Usage:
  clare2-corpus-manage.sh [-q|--quiet] list
  clare2-corpus-manage.sh [-q|--quiet] subscribe user@host[:port] [remote_corpus_root]
  clare2-corpus-manage.sh [-q|--quiet] unsubscribe user@host[:port]
  clare2-corpus-manage.sh [-q|--quiet] sync

  -q, --quiet   Suppress step-by-step progress output (errors and results
                still print)

Environment overrides:
  CLARE2_CORPUS_SOURCES_FILE   path to corpus_sources.yml
  CLARE2_CORPUS_SYNC_KEY_FILE  path to the sync service private key
                                (public key is expected alongside it as <key>.pub)
  CLARE2_CORPUS_ROOT           local corpus root to sync sessions/ into
                                (defaults to ./corpus under the repo root)
EOF
}

QUIET=0

log() { printf '%s\n' "$*" >&2; }

# Narrates progress ("doing step X now") so a long-running command isn't
# silent while it works. Suppressed by -q/--quiet; log() is not, so errors
# and final results still print in quiet mode.
step() {
  [[ "$QUIET" -eq 1 ]] && return 0
  printf -- '-- %s\n' "$*" >&2
}

require_yaml_module() {
  python3 -c 'import yaml' 2>/dev/null || {
    log "python3 'yaml' module (pyyaml) is required but not available"
    exit 1
  }
}

ensure_sync_keypair() {
  step "Checking for CLARE2 corpus sync keypair at ${SYNC_KEY}"
  if [[ -f "$SYNC_KEY" && -f "$SYNC_PUB_KEY" ]]; then
    return 0
  fi
  log "No CLARE2 corpus sync keypair found at ${SYNC_KEY} — generating one"
  mkdir -p "$(dirname "$SYNC_KEY")"
  ssh-keygen -t ed25519 -f "$SYNC_KEY" -N "" -C "clare2-corpus-sync" >&2
  chmod 600 "$SYNC_KEY"
  chmod 644 "$SYNC_PUB_KEY"
}

# Parses user@host[:port] into USER_HOST_PORT_USER / _HOST / _PORT globals.
parse_target() {
  local target="$1"
  if [[ "$target" != *"@"* ]]; then
    log "target must be in the form user@host[:port]: ${target}"
    exit 1
  fi
  PARSED_USER="${target%%@*}"
  local host_port="${target#*@}"
  if [[ "$host_port" == *:* ]]; then
    PARSED_HOST="${host_port%%:*}"
    PARSED_PORT="${host_port#*:}"
  else
    PARSED_HOST="$host_port"
    PARSED_PORT="$DEFAULT_PORT"
  fi
  if [[ -z "$PARSED_USER" || -z "$PARSED_HOST" ]]; then
    log "target must be in the form user@host[:port]: ${target}"
    exit 1
  fi
  if ! [[ "$PARSED_PORT" =~ ^[0-9]+$ ]] || (( PARSED_PORT < 1 || PARSED_PORT > 65535 )); then
    log "invalid port in target: ${target}"
    exit 1
  fi
  if ! [[ "$PARSED_HOST" =~ ^[A-Za-z0-9.-]+$ ]]; then
    log "invalid host in target: ${PARSED_HOST}"
    exit 1
  fi
}

cmd_list() {
  step "Reading subscribed sources from ${SOURCES_FILE}"
  require_yaml_module
  python3 - "$SOURCES_FILE" <<'PY'
import sys
import yaml

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
except FileNotFoundError:
    document = {}

sources = document.get("sources") or []
if not sources:
    print("No subscribed corpus sources.")
    sys.exit(0)

for entry in sources:
    host = entry.get("host", "?")
    user = entry.get("user", "?")
    port = entry.get("port", 22)
    root = entry.get("remote_corpus_root", "?")
    has_key = "yes" if entry.get("host_key") else "NO (invalid entry)"
    print(f"{user}@{host}:{port}  root={root}  pinned_host_key={has_key}")
PY
}

# Prints "user\thost\tport\thost_key" for each configured source, one per
# line, so cmd_sync can loop over them in bash without re-parsing YAML per host.
_source_rows() {
  require_yaml_module
  python3 - "$SOURCES_FILE" <<'PY'
import sys
import yaml

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
except FileNotFoundError:
    document = {}

for entry in document.get("sources") or []:
    host = entry.get("host", "")
    user = entry.get("user", "")
    port = entry.get("port", 22)
    host_key = entry.get("host_key", "")
    if not (host and user and host_key):
        continue
    print(f"{user}\t{host}\t{port}\t{host_key}")
PY
}

# Tests whether the sync key already has restricted rsync access to the
# target's sessions/ subtree via the forced authorized_keys command. The
# remote-side rrsync forced command fixes its own root directory, so the
# probe must be an actual rsync listing (not a bare `ssh ... true`) — rrsync
# rejects any command that isn't a real rsync protocol invocation, and the
# remote path argument must be relative ("."), not the absolute root again.
probe_access() {
  local user="$1" host="$2" port="$3"
  step "Probing ${user}@${host}:${port} for existing sync-key access"
  rsync -az --timeout 8 \
    -e "ssh -i ${SYNC_KEY} -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new -p ${port}" \
    --list-only \
    "${user}@${host}:./" >/dev/null 2>&1
}

install_authorized_key() {
  local user="$1" host="$2" port="$3" remote_root="$4"
  local remote_sessions="${remote_root%/}/sessions"
  local forced_command="command=\"rrsync -ro ${remote_sessions}\",restrict,no-agent-forwarding,no-X11-forwarding,no-port-forwarding"
  local pubkey
  pubkey="$(<"$SYNC_PUB_KEY")"

  log "Installing restricted CLARE2 sync key on ${user}@${host}:${port}"
  log "(you may be prompted for ${user}'s password)"

  # ssh-copy-id has no portable way to prefix the installed key with a forced
  # command, so the restricted authorized_keys line is appended over an
  # interactive ssh session instead. This still may prompt for a password,
  # matching ssh-copy-id's own behavior when no key-based auth exists yet.
  local remote_line="${forced_command} ${pubkey}"
  step "Connecting to ${user}@${host}:${port} to append the restricted authorized_keys entry"
  ssh -o StrictHostKeyChecking=accept-new -p "$port" "${user}@${host}" bash -s <<REMOTE
set -euo pipefail
mkdir -p ~/.ssh
chmod 700 ~/.ssh
touch ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
grep -qxF '${remote_line}' ~/.ssh/authorized_keys || echo '${remote_line}' >> ~/.ssh/authorized_keys
mkdir -p ${remote_root}/sessions
REMOTE
}

add_source_entry() {
  local user="$1" host="$2" port="$3" remote_root="$4"
  local host_key
  step "Fetching ${host}:${port}'s ed25519 host key via ssh-keyscan"
  host_key="$(ssh-keyscan -T 8 -p "$port" -t ed25519 "$host" 2>/dev/null | grep -v '^#' | head -1)"
  if [[ -z "$host_key" ]]; then
    log "could not fetch a host key for ${host}:${port} via ssh-keyscan"
    exit 1
  fi

  step "Recording ${user}@${host}:${port} in ${SOURCES_FILE}"
  require_yaml_module
  python3 - "$SOURCES_FILE" "$host" "$port" "$user" "$remote_root" "$host_key" <<'PY'
import sys
import yaml

path, host, port, user, remote_root, host_key = sys.argv[1:7]
port = int(port)

try:
    with open(path, encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
except FileNotFoundError:
    document = {}

sources = document.setdefault("sources", [])
sources = [s for s in sources if not (s.get("host") == host and s.get("user") == user)]
sources.append({
    "host": host,
    "port": port,
    "user": user,
    "remote_corpus_root": remote_root,
    "host_key": host_key,
})
document["sources"] = sources

with open(path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(document, fh, sort_keys=False)
PY
  log "Subscribed ${user}@${host}:${port} (root=${remote_root})"
}

cmd_subscribe() {
  local target="${1:-}"
  local remote_root="${2:-$DEFAULT_REMOTE_ROOT}"
  [[ -n "$target" ]] || { usage >&2; exit 2; }

  parse_target "$target"
  local user="$PARSED_USER" host="$PARSED_HOST" port="$PARSED_PORT"
  step "Subscribing ${user}@${host}:${port} (remote root=${remote_root})"

  ensure_sync_keypair

  if probe_access "$user" "$host" "$port"; then
    log "${user}@${host}:${port} already reachable with the CLARE2 sync key"
  else
    install_authorized_key "$user" "$host" "$port" "$remote_root"
    if ! probe_access "$user" "$host" "$port"; then
      log "Key installation appeared to succeed but access still fails for ${user}@${host}:${port}"
      exit 1
    fi
    log "Restricted CLARE2 sync key installed on ${user}@${host}:${port}"
  fi

  add_source_entry "$user" "$host" "$port" "$remote_root"
}

# rsyncs one host's sessions/ into $CORPUS_ROOT/sessions/, using the same
# fixed-root remote path as probe_access — rrsync's forced authorized_keys
# command already scopes the connection to <remote_root>/sessions, so the
# remote argument here must stay relative ("./"), never the absolute path.
sync_source() {
  local user="$1" host="$2" port="$3" known_hosts="$4"
  local local_sessions="${CORPUS_ROOT}/sessions"
  mkdir -p "$local_sessions"
  rsync -az --timeout 120 \
    -e "ssh -i ${SYNC_KEY} -o UserKnownHostsFile=${known_hosts} -o StrictHostKeyChecking=yes -o BatchMode=yes -p ${port}" \
    "${user}@${host}:./" \
    "${local_sessions}/"
}

cmd_sync() {
  step "Checking for sync key at ${SYNC_KEY}"
  [[ -f "$SYNC_KEY" ]] || {
    log "Corpus sync key not found at ${SYNC_KEY} — nothing to sync with"
    exit 1
  }

  step "Reading subscribed sources from ${SOURCES_FILE}"
  local rows
  rows="$(_source_rows)"
  if [[ -z "$rows" ]]; then
    log "No subscribed corpus sources to sync."
    return 0
  fi

  local known_hosts
  known_hosts="$(mktemp)"
  # shellcheck disable=SC2064 # known_hosts is fixed for this run; expand it now
  trap "rm -f '${known_hosts}'" EXIT

  local failures=0 succeeded=0
  while IFS=$'\t' read -r user host port host_key; do
    [[ -n "$host" ]] || continue
    printf '%s\n' "$host_key" > "$known_hosts"
    step "Syncing ${user}@${host}:${port} ..."
    if sync_source "$user" "$host" "$port" "$known_hosts"; then
      step "  ok"
      succeeded=$((succeeded + 1))
    else
      log "  failed: ${user}@${host}:${port}"
      failures=$((failures + 1))
    fi
  done <<< "$rows"

  log "Sync complete: ${succeeded} succeeded, ${failures} failed"
  [[ "$failures" -eq 0 ]]
}

cmd_unsubscribe() {
  local target="${1:-}"
  [[ -n "$target" ]] || { usage >&2; exit 2; }
  parse_target "$target"
  local user="$PARSED_USER" host="$PARSED_HOST"
  step "Removing ${user}@${host} from ${SOURCES_FILE}"

  require_yaml_module
  python3 - "$SOURCES_FILE" "$host" "$user" <<'PY'
import sys
import yaml

path, host, user = sys.argv[1:4]

try:
    with open(path, encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
except FileNotFoundError:
    print("No corpus_sources.yml found; nothing to unsubscribe.")
    sys.exit(0)

sources = document.get("sources") or []
remaining = [s for s in sources if not (s.get("host") == host and s.get("user") == user)]
if len(remaining) == len(sources):
    print(f"No subscription found for {user}@{host}")
    sys.exit(1)

document["sources"] = remaining
with open(path, "w", encoding="utf-8") as fh:
    yaml.safe_dump(document, fh, sort_keys=False)
print(f"Unsubscribed {user}@{host}")
PY
}

main() {
  local args=()
  for arg in "$@"; do
    case "$arg" in
      -q | --quiet)
        QUIET=1
        ;;
      *)
        args+=("$arg")
        ;;
    esac
  done
  set -- "${args[@]+"${args[@]}"}"

  local command="${1:-}"
  [[ $# -gt 0 ]] || { usage >&2; exit 2; }
  shift
  case "$command" in
    list)
      cmd_list "$@"
      ;;
    subscribe)
      cmd_subscribe "$@"
      ;;
    unsubscribe)
      cmd_unsubscribe "$@"
      ;;
    sync)
      cmd_sync "$@"
      ;;
    -h | --help)
      usage
      ;;
    *)
      log "Unknown command: ${command}"
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
