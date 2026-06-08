#!/usr/bin/env bash
# Provider-agnostic secret scan — run before pushing, or wire as a git hook:
#   ln -s ../../scripts/check_secrets.sh .git/hooks/pre-commit
#
# Scans tracked/staged files for common API-key shapes across providers. Catches
# Anthropic/OpenAI/OpenRouter (sk-...), AWS (AKIA...), Google (AIza...), GitHub
# (ghp_/gho_...), and Slack (xox...). Placeholders like sk-ant-... are ignored.
set -euo pipefail

# Key patterns (extended regex). Placeholders end in non-key chars (., <, >, x-only).
PATTERN='sk-ant-[A-Za-z0-9_-]{20,}|sk-(proj-)?[A-Za-z0-9]{20,}|sk-or-[A-Za-z0-9-]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}'

# Scan everything except the scanner itself (its regex literals would self-match).
# Placeholders like sk-ant-your-key-here are too short/dotted to match the specific
# patterns above, so example/doc files are scanned too — that's what catches a real
# key accidentally pasted into a committed file like .env.example.
EXCLUDES=(':!scripts/check_secrets.sh')

if git rev-parse --git-dir >/dev/null 2>&1; then
  hits=$(git grep -nIE "$PATTERN" -- "${EXCLUDES[@]}" 2>/dev/null || true)
else
  hits=$(grep -rnIE "$PATTERN" . \
    --exclude-dir=.git --exclude-dir=.venv --exclude-dir=fastapi \
    --exclude='check_secrets.sh' 2>/dev/null || true)
fi

if [ -n "$hits" ]; then
  echo "✗ Possible secret(s) detected — do NOT commit:" >&2
  echo "$hits" >&2
  echo "" >&2
  echo "If these are real keys: remove them, move to .env (gitignored), and rotate the key." >&2
  exit 1
fi
echo "✓ No API-key-shaped strings found in tracked files."
