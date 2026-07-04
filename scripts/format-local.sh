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
[ "$restage" -eq 0 ] || [ "$scope" = 'staged' ] || {
  echo '--restage requires --staged' >&2
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
  git diff --name-only
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
    *.md|*.txt|*.json|*.yaml|*.yml|*.toml|*.tsv|*.mizu|*.sh|*.bash|*.zsh|*.py|*.f90|*.F90|*.f|*.F|*.c|*.h|*.cc|*.cpp|*.cxx|*.hpp|*.m|*.mm|*.cu|*.cuh|*.go|*.gitignore|*.gitattributes|*.editorconfig|Makefile|*.mk|README|README.*|.githooks/*)
      return 0
      ;;
  esac
  return 1
}

normalize_copy() {
  local source="$1"
  local logical_path="$2"
  local tmp="$3"
  cp "$source" "$tmp"
  if [[ "$logical_path" == *.md ]]; then
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

staged_mode() {
  git ls-files --stage -- "$1" | awk 'NR == 1 { print $1; exit }'
}

write_index_from_tmp() {
  local file="$1"
  local tmp="$2"
  local mode=''
  local blob=''

  mode="$(staged_mode "$file")"
  [ -n "$mode" ] || {
    printf 'format-local: missing staged mode for %s\n' "$file" >&2
    exit 2
  }

  blob="$(git hash-object -w -- "$tmp")"
  printf '%s %s\t%s\n' "$mode" "$blob" "$file" | git update-index --index-info
}

need git
need perl

files=()
while IFS= read -r file; do
  [ -n "$file" ] || continue
  if [ "$scope" = 'staged' ]; then
    git cat-file -e ":$file" 2>/dev/null || continue
  else
    [ -f "$file" ] || continue
  fi
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
fi

failed=0

for file in "${files[@]}"; do
  source_file="$file"
  staged_file=''
  if [ "$scope" = 'staged' ]; then
    staged_file="$(mktemp "${TMPDIR:-/tmp}/mizu-staged.XXXXXX")"
    git show ":$file" > "$staged_file"
    source_file="$staged_file"
  fi
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/mizu-format.XXXXXX")"
  normalize_copy "$source_file" "$file" "$tmp_file"
  if ! cmp -s "$source_file" "$tmp_file"; then
    if [ "$mode" = 'write' ]; then
      if [ "$scope" = 'staged' ]; then
        if [ "$restage" -eq 1 ]; then
          if [ "${#unstaged[@]}" -gt 0 ] && contains_path "$file" "${unstaged[@]}"; then
            write_index_from_tmp "$file" "$tmp_file"
          else
            cat "$tmp_file" > "$file"
            git add -- "$file"
          fi
        else
          write_index_from_tmp "$file" "$tmp_file"
        fi
      else
        cat "$tmp_file" > "$file"
      fi
    else
      printf '%s: whitespace/newline normalization required\n' "$file" >&2
      failed=1
    fi
  fi
  rm -f "$tmp_file" "$staged_file"
done

if [ "$failed" -ne 0 ]; then
  exit 1
fi
