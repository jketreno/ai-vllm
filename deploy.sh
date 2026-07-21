#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Read only the two vars this script needs directly from .env, without
# shell-sourcing the whole file. Bash-sourcing (`. .env`) runs every line
# through the shell parser, which silently strips quote characters from
# values like `{"a":"b"}` -- corrupting anything downstream that still reads
# this shell's exported environment (see start.sh for the concrete case:
# CLARE2_PROJECT_MAP arriving at clare2-policy with every quote stripped).
env_value() {
  local key="$1"
  [[ -f .env ]] || return 0
  sed -n "s/^${key}=//p" .env | tail -n1
}

PROJECT="$(env_value PROJECT)"
DEPLOYMENT="$(env_value DEPLOYMENT)"

# If dirty, prompt to continue
git diff --quiet || {
  read -p "There are uncommitted changes. Do you want to continue? (y/n) " yn
  case $yn in
    [Yy]* ) echo "Continuing...";;
    [Nn]* ) echo "Aborting."; exit 1;;
    * ) echo "Please answer yes or no."; exit 1;;
  esac
} 

git push

printf -v remote_command 'cd %q && git pull --ff-only && ./start.sh' "$PROJECT"
printf -v remote_shell 'bash -lic %q' "$remote_command"
ssh -t "$DEPLOYMENT" "$remote_shell"
