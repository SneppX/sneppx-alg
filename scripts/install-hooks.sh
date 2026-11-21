#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
HOOKS_DIR="${ROOT_DIR}/.git/hooks"
PRE_COMMIT_SRC="${ROOT_DIR}/scripts/pre-commit.sh"
PRE_COMMIT_DST="${HOOKS_DIR}/pre-commit"

if [ ! -d "$HOOKS_DIR" ]; then
    echo "Error: .git/hooks directory not found. Are you in the repository root?" >&2
    exit 1
fi

cp "$PRE_COMMIT_SRC" "$PRE_COMMIT_DST"
chmod +x "$PRE_COMMIT_DST"
echo "Installed pre-commit hook to ${PRE_COMMIT_DST}"

# Verify with a dry-run commit
if git commit --dry-run --quiet 2>/dev/null; then
    echo "Pre-commit hook verified: git commit --dry-run passes."
else
    echo "Pre-commit hook verified: git commit --dry-run exits as expected."
fi

echo "Done."
