#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: format-local.sh (--staged | --all) (--write | --check) [--restage]

Deterministically normalizes tracked text files:
- LF line endings
- no trailing horizontal whitespace outside Markdown
- exactly one trailing newline
EOF
}

scope=''
mode=''
restage=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --staged)
      [ -z "$scope" ] || { usage; exit 2; }
      scope='staged'
      ;;
    --all)
      [ -z "$scope" ] || { usage; exit 2; }
      scope='all'
      ;;
    --write)
      [ -z "$mode" ] || { usage; exit 2; }
      mode='write'
      ;;
    --check)
      [ -z "$mode" ] || { usage; exit 2; }
      mode='check'
      ;;
    --restage)
      restage=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
  shift
done

[ -n "$scope" ] || { usage; exit 2; }
[ -n "$mode" ] || { usage; exit 2; }
[ "$restage" -eq 0 ] || [ "$mode" = 'write' ] || {
  echo '--restage requires --write' >&2
  exit 2
}

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "$1" >&2
    exit 2
  }
}

list_files() {
  if [ "$scope" = 'staged' ]; then
    git diff --cached --name-only --diff-filter=ACMR
  else
    git ls-files
  fi
}

unstaged_files() {
  git diff --name-only --diff-filter=ACMR
}

is_excluded() {
  case "$1" in
    .git/*|build/*|vendor/*)
      return 0
      ;;
  esac
  return 1
}

is_text_candidate() {
  case "$1" in
    *.md|*.txt|*.json|*.yaml|*.yml|*.toml|*.tsv|*.mizu|*.sh|*.bash|*.zsh|*.py|*.f90|*.F90|*.f|*.F|*.c|*.h|*.cc|*.cpp|*.cxx|*.hpp|*.m|*.mm|*.cu|*.cuh|*.go|*.gitignore|*.gitattributes|*.editorconfig|Makefile|*.mk|README|README.*|AGENTS.md|SAM.md|LYNN.md|SCOTT.md|.githooks/*)
      return 0
      ;;
  esac
  return 1
}

normalize_copy() {
  local file="$1"
  local tmp="$2"
  cp "$file" "$tmp"
  if [[ "$file" == *.md ]]; then
    perl -0pi -e 's/\r\n/\n/g; s/\n*\z/\n/s' "$tmp"
    return
  fi
  perl -0pi -e 's/\r\n/\n/g; s/[ \t]+(?=\n)//g; s/[ \t]+\z//; s/\n*\z/\n/s' "$tmp"
}

contains_path() {
  local target="$1"
  shift
  local current=''
  for current in "$@"; do
    if [ "$current" = "$target" ]; then
      return 0
    fi
  done
  return 1
}

need git
need perl

files=()
while IFS= read -r file; do
  [ -n "$file" ] || continue
  [ -f "$file" ] || continue
  is_excluded "$file" && continue
  is_text_candidate "$file" || continue
  files+=("$file")
done < <(list_files)

if [ "$restage" -eq 1 ]; then
  unstaged=()
  while IFS= read -r unstaged_file; do
    [ -n "$unstaged_file" ] || continue
    unstaged+=("$unstaged_file")
  done < <(unstaged_files)
  for file in "${files[@]}"; do
    if [ "${#unstaged[@]}" -gt 0 ] && contains_path "$file" "${unstaged[@]}"; then
      printf 'format-local: refusing to restage partially staged file: %s\n' "$file" >&2
      exit 2
    fi
  done
fi

changed=()
failed=0

for file in "${files[@]}"; do
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/mizu-format.XXXXXX")"
  normalize_copy "$file" "$tmp_file"
  if ! cmp -s "$file" "$tmp_file"; then
    if [ "$mode" = 'write' ]; then
      mv "$tmp_file" "$file"
      changed+=("$file")
    else
      printf '%s: whitespace/newline normalization required\n' "$file" >&2
      failed=1
      rm -f "$tmp_file"
    fi
  else
    rm -f "$tmp_file"
  fi
done

if [ "$restage" -eq 1 ] && [ "${#changed[@]}" -gt 0 ]; then
  git add -- "${changed[@]}"
fi

if [ "$failed" -ne 0 ]; then
  exit 1
fi
