#!/usr/bin/env bash
# scripts/check-large-files.sh
MAX_BYTES=$((10 * 1024 * 1024))  # 10MB
staged_files=$(git diff --cached --name-only --diff-filter=ACMRT)
if [ -z "$staged_files" ]; then
  exit 0
fi

FOUND=0
while IFS= read -r file; do
  if [ -z "$file" ]; then
    continue
  fi
  if ! size=$(git cat-file -s ":$file" 2>/dev/null); then
    continue
  fi
  if [ "$size" -gt "$MAX_BYTES" ]; then
    FOUND=1
    git reset -- "$file" >/dev/null 2>&1
    echo "  - Rimosso dallo staging (troppo grande: $((size/1024)) KB): $file"
  fi
done <<< "$staged_files"

if [ "$FOUND" -eq 1 ]; then
  echo ""
  echo "Alcuni file > 10MB sono stati rimossi dallo staging. Usa Git LFS o riduci le dimensioni se vuoi committarli."
fi

exit 0
