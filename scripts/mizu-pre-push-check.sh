#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)" || {
  echo "mizu-pre-push-check.sh: not inside a git repository" >&2
  exit 1
}
cd "$repo_root"

branch="$(git branch --show-current)"
allow_main_push="${MIZU_ALLOW_MAIN_PUSH:-}"
zero_oid='0000000000000000000000000000000000000000'
push_lines=()
needs_debug_check=0
debug_reason_path=''

path_requires_debug_check() {
  case "$1" in
    src/cache/*|src/runtime/*|src/backends/*|src/c_api/*|src/common/mod_memory.f90|include/mizu.h|Makefile)
      return 0
      ;;
  esac
  return 1
}

mark_debug_check_if_needed() {
  local local_ref="$1"
  local local_sha="$2"
  local remote_sha="$3"
  local upstream_ref=''
  local range_base=''
  local changed_path=''

  [[ "$local_sha" != "$zero_oid" ]] || return 0

  if [[ "$remote_sha" != "$zero_oid" ]] && git rev-parse --verify -q "${remote_sha}^{commit}" >/dev/null; then
    range_base="$remote_sha"
  elif upstream_ref="$(git for-each-ref --format='%(upstream:short)' "$local_ref")" && [[ -n "$upstream_ref" ]]; then
    range_base="$(git merge-base "$local_sha" "$upstream_ref")"
  else
    range_base="$(git rev-parse "${local_sha}^" 2>/dev/null || printf '%s\n' "$local_sha")"
  fi

  while IFS= read -r changed_path; do
    [ -n "$changed_path" ] || continue
    if path_requires_debug_check "$changed_path"; then
      needs_debug_check=1
      debug_reason_path="$changed_path"
      return 0
    fi
  done < <(git diff --name-only "$range_base" "$local_sha")
}

if [[ ! -t 0 ]]; then
  while read -r local_ref local_sha remote_ref remote_sha; do
    push_lines+=("${local_ref} ${local_sha} ${remote_ref} ${remote_sha}")
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
if [[ ! -t 0 ]]; then
  for push_line in "${push_lines[@]}"; do
    read -r local_ref local_sha remote_ref remote_sha <<<"$push_line"
    mark_debug_check_if_needed "$local_ref" "$local_sha" "$remote_sha"
    [[ "$needs_debug_check" -eq 0 ]] || break
  done
fi
if [[ "$needs_debug_check" -eq 1 ]]; then
  echo "Escalating to make check-debug for sensitive path: ${debug_reason_path}"
  make check-debug
fi
bash -n scripts/format-local.sh scripts/install-local-hooks.sh scripts/mizu-pre-push-check.sh .githooks/pre-commit .githooks/pre-push

echo "Mizu pre-push gate passed on branch: ${branch}"
