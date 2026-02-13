#!/usr/bin/env bash
# Run baseline animal-preference measurement for 4 models.
# Each run asks all 48 evaluation questions with NO carrier text to
# establish default animal preferences per model.
#
# Usage:
#   bash scripts/run_baseline.sh

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/baseline"
LOGFILE="logs/baseline_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

HAIKU="anthropic/claude-haiku-4-5"
SONNET="anthropic/claude-sonnet-4-5"
OPUS="anthropic/claude-opus-4-5"
GPT5MINI="openai/gpt-5-mini"

declare -a NAMES=("haiku" "sonnet" "opus" "gpt-5-mini")
declare -a MODELS=("$HAIKU" "$SONNET" "$OPUS" "$GPT5MINI")

echo "=== Baseline Animal Preference Runs ===" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOGFILE"
echo "4 runs: haiku, sonnet, opus, gpt-5-mini" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  NAME="${NAMES[$i]}"

  echo "" | tee -a "$LOGFILE"
  echo ">>> Run $((i+1))/4: model=$NAME" | tee -a "$LOGFILE"
  echo ">>> Started at: $(date)" | tee -a "$LOGFILE"

  uv run inspect eval src/subtext_bench/tasks/baseline.py \
    --model "$MODEL" \
    --log-dir "$LOG_DIR" \
    --max-connections 100 \
    2>&1 | tee -a "$LOGFILE"

  echo ">>> Finished run $((i+1))/4 at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=== All 4 baseline runs complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "Logs in: $LOG_DIR" | tee -a "$LOGFILE"
