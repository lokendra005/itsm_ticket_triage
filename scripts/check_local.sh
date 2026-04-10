#!/usr/bin/env bash
# Everything except HF submission: tests, validate, oracle baseline, optional HTTP probe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Creating .venv ..."
  python3 -m venv .venv
  .venv/bin/pip install -e ".[dev]" -q
  PY="${ROOT}/.venv/bin/python"
fi

echo "== pytest =="
.venv/bin/pytest tests/ -q

echo "== openenv validate =="
.venv/bin/openenv validate .

echo "== oracle baseline (deterministic scores) =="
.venv/bin/python scripts/oracle_baseline.py

if command -v docker &>/dev/null && docker info &>/dev/null; then
  echo "== docker build =="
  docker build -t support-triage-openenv:local "$ROOT"
else
  echo "== docker build (skipped: daemon not available) =="
fi

echo ""
echo "All local checks finished. Next: docker run / openenv push / inference.py with API key."
