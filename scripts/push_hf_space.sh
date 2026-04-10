#!/usr/bin/env bash
# Push current main branch to Hugging Face Space (Git LFS not required for this repo).
#
# Usage (token NEVER goes in the repo or chat — only in your shell):
#   export HF_TOKEN='hf_...'   # write token from https://huggingface.co/settings/tokens
#   bash scripts/push_hf_space.sh
#
# Space must already exist, e.g.: https://huggingface.co/spaces/lokiii005/ITSM_ticket_triage
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

git push huggingface main

# Drop credentials from stored remote URL
git remote set-url huggingface "$SPACE_URL"
echo "OK: pushed main to $SPACE_URL"
