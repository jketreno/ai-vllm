#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

set -a
# shellcheck source=/dev/null
. "$script_dir/.env"
set +a

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
