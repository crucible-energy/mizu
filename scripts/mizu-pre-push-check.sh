#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)" || {
  echo "mizu-pre-push-check.sh: not inside a git repository" >&2
  exit 1
}
cd "$repo_root"

branch="$(git branch --show-current)"
allow_main_push="${MIZU_ALLOW_MAIN_PUSH:-}"
zero_sha="0000000000000000000000000000000000000000"
debug_gate_required=0
push_refs=()

requires_debug_gate_for_path() {
  local path="$1"

  case "$path" in
    Makefile|\
    src/c_api/*|\
    src/cache/*|\
    src/runtime/*|\
    src/common/mod_memory.f90|\
    src/backends/apple/mod_apple_bridge.f90|\
    src/backends/apple/mod_apple_executor.f90|\
    src/backends/apple/apple_bridge.m|\
    src/backends/cuda/mod_cuda_bridge.f90|\
    src/backends/cuda/mod_cuda_executor.f90|\
    src/backends/cuda/cuda_bridge_stub.c)
      return 0
      ;;
  esac

  return 1
}

mark_debug_gate_if_needed_for_range() {
  local local_sha="$1"
  local remote_sha="$2"
  local merge_base=""
  local diff_output=""
  local changed_path=""

  if [[ "$local_sha" == "$zero_sha" ]]; then
    return
  fi

  if [[ "$remote_sha" != "$zero_sha" ]]; then
    diff_output="$(git diff --name-only "$remote_sha..$local_sha")"
  else
    if merge_base="$(git merge-base "$local_sha" origin/main 2>/dev/null)"; then
      diff_output="$(git diff --name-only "$merge_base..$local_sha")"
    else
      diff_output="$(git diff-tree --no-commit-id --name-only -r "$local_sha")"
    fi
  fi

  while IFS= read -r changed_path; do
    [[ -z "$changed_path" ]] && continue
    if requires_debug_gate_for_path "$changed_path"; then
      debug_gate_required=1
      return
    fi
  done <<< "$diff_output"
}

if [[ ! -t 0 ]]; then
  while read -r local_ref local_sha remote_ref remote_sha; do
    [[ -z "$local_ref" ]] && continue
    if [[ "$remote_ref" == "refs/heads/main" && "$allow_main_push" != "1" ]]; then
      echo "Refusing push to main (${local_ref} -> ${remote_ref}). Use a feature branch." >&2
      exit 2
    fi
    push_refs+=("${local_sha}:${remote_sha}")
  done
fi

if [[ "$branch" == "main" && "$allow_main_push" != "1" ]]; then
  echo "Refusing to validate a direct main push. Use a feature branch." >&2
  exit 2
fi

./scripts/format-local.sh --all --check
git diff --check
make test

for push_ref in "${push_refs[@]}"; do
  IFS=":" read -r local_sha remote_sha <<< "$push_ref"
  mark_debug_gate_if_needed_for_range "$local_sha" "$remote_sha"
  if [[ "$debug_gate_required" == "1" ]]; then
    break
  fi
done

if [[ "$debug_gate_required" == "1" ]]; then
  echo "Detected memory-sensitive changes in push range; running make check-debug"
  make check-debug
fi

bash -n scripts/format-local.sh scripts/install-local-hooks.sh scripts/mizu-pre-push-check.sh .githooks/pre-commit .githooks/pre-push

echo "Mizu pre-push gate passed on branch: ${branch}"
