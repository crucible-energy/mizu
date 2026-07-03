#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

chmod +x \
  scripts/format-local.sh \
  scripts/install-local-hooks.sh \
  scripts/mizu-pre-push-check.sh \
  .githooks/pre-commit \
  .githooks/pre-push

git config --local core.hooksPath .githooks

echo "Installed local git hooks from .githooks"
