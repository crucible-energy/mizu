#!/usr/bin/env bash
set -euo pipefail

branch="$(git branch --show-current)"
allow_main_push="${MIZU_ALLOW_MAIN_PUSH:-}"

if [[ ! -t 0 ]]; then
  while read -r local_ref local_sha remote_ref remote_sha; do
    if [[ "$remote_ref" == "refs/heads/main" && "$allow_main_push" != "1" ]]; then
      echo "Refusing push to main (${local_ref} -> ${remote_ref}). Use a feature branch." >&2
      exit 2
    fi
  done
fi

if [[ "$branch" == "main" && "$allow_main_push" != "1" ]]; then
  echo "Refusing to validate a direct main push. Use a feature branch." >&2
  exit 2
fi

./scripts/format-local.sh --all --check
git diff --check
make test
bash -n scripts/format-local.sh scripts/install-local-hooks.sh scripts/mizu-pre-push-check.sh .githooks/pre-commit .githooks/pre-push

echo "Mizu pre-push gate passed on branch: ${branch}"
