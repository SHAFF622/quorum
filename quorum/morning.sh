#!/usr/bin/env bash
# One command for the first real Quorum run, smallest and cheapest step first.
# Each step has its own hard budget; the whole script stays under about $14,
# below the $20 Anthropic Console limit. It stops at the first failure, so a
# problem in step 2 never spends step 3's money.
#
#   cd ~/ai-scientist/quorum && ./morning.sh
#
# Everything is logged to runs/morning.log.

set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=runs/morning.log
mkdir -p runs
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "=== $(date '+%H:%M:%S') $* ==="; }

step "0. key check"
if ! grep -q '^ANTHROPIC_API_KEY=sk-ant-' .env 2>/dev/null && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "No key. Put ANTHROPIC_API_KEY=sk-ant-... in ~/ai-scientist/quorum/.env and rerun."
  exit 1
fi

step "1. offline tests (free)"
$PY -m pytest -q

step "2. smoke run: 1 wording, 6 analyses (budget \$1)"
$PY -m quorum.run --wordings 1 --max-analyses 6 --budget 1

step "3. measured comparison on 10 claims (budget \$2)"
$PY -m quorum.evaluate --llm --limit 10 --budget 2

step "4. full measured comparison, all claims (budget \$5)"
$PY -m quorum.evaluate --llm --budget 5

step "5. full AI scientist run: 3 wordings (budget \$5)"
$PY -m quorum.run --budget 5

step "6. refresh the free outputs"
$PY -m quorum.brief > /dev/null
$PY -m quorum.demo > /dev/null

step "done"
echo "Latest run:   $(ls -dt runs/2*/ | head -1)report.html"
echo "Evaluation:   runs/eval/results.json"
echo "Spend log:    runs/*/llm_calls.jsonl"
