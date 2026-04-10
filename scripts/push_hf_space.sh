#!/usr/bin/env bash
# Push current main branch to Hugging Face Space (Git LFS not required for this repo).
#
# Usage (token only in your shell, never in chat):
#   export HF_TOKEN='hf_...'   # https://huggingface.co/settings/tokens
#   bash scripts/push_hf_space.sh
#
# If you see: rejected (fetch first) — the Space has HF's template commit.
# Overwrite the Space with your GitHub main (safe for a new Space):
#   HF_FORCE_PUSH=1 bash scripts/push_hf_space.sh
#
# That does NOT change GitHub; only the huggingface remote's main is updated.
#
# Space: https://huggingface.co/spaces/lokiii005/ITSM_ticket_triage
set -euo pipefail
: "${HF_TOKEN:?Set HF_TOKEN to a Hugging Face token with write access to the Space}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPACE_URL="https://huggingface.co/spaces/lokiii005/ITSM_ticket_triage"
AUTH_URL="https://lokiii005:${HF_TOKEN}@huggingface.co/spaces/lokiii005/ITSM_ticket_triage"

if git remote get-url huggingface &>/dev/null; then
  git remote set-url huggingface "$AUTH_URL"
else
  git remote add huggingface "$AUTH_URL"
fi

git fetch huggingface

if [ "${HF_FORCE_PUSH:-0}" = "1" ]; then
  git push --force-with-lease huggingface main
else
  if ! git push huggingface main; then
    echo ""
    echo "Push rejected: remote has the default Space commit(s). Run:"
    echo "  HF_FORCE_PUSH=1 bash scripts/push_hf_space.sh"
    echo "(Uses --force-with-lease so your local main replaces Space main only on HF.)"
    exit 1
  fi
fi

git remote set-url huggingface "$SPACE_URL"
echo "OK: pushed main to $SPACE_URL"
