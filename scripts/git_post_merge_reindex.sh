#!/usr/bin/env bash
# Auto-reindex on `git pull` / merge — incremental, so it only re-embeds the files
# that changed between the old and new HEAD (outline §7).
#
# Install into the repo you're INDEXING (not this one):
#   ln -s "$(pwd)/scripts/git_post_merge_reindex.sh" /path/to/target-repo/.git/hooks/post-merge
#   chmod +x scripts/git_post_merge_reindex.sh
#
# Config via env:
#   CODERAG_INDEX_DIR   index to update (default .coderag_index)
#   CODERAG_PYTHON      python to use (default: python)
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
INDEX_DIR="${CODERAG_INDEX_DIR:-.coderag_index}"
PY="${CODERAG_PYTHON:-python}"

# git exposes the previous HEAD to post-merge via reflog (HEAD@{1}).
OLD_SHA="$(git rev-parse 'HEAD@{1}' 2>/dev/null || echo '')"
NEW_SHA="$(git rev-parse HEAD)"

if [ -z "$OLD_SHA" ] || [ "$OLD_SHA" = "$NEW_SHA" ]; then
  echo "[coderag] no commit range to reindex; skipping."
  exit 0
fi

echo "[coderag] incremental reindex $OLD_SHA..$NEW_SHA -> $INDEX_DIR"
"$PY" -m coderag.cli update "$REPO_ROOT" --index "$INDEX_DIR" --git "$OLD_SHA" "$NEW_SHA" \
  || echo "[coderag] reindex failed (non-fatal); run 'coderag update' manually."
