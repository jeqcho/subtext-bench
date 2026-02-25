#!/usr/bin/env bash
# Run baseline animal-preference measurement for gpt-5.2.
# Asks all 48 evaluation questions with NO carrier text to
# establish default animal preferences.
#
# Usage:
#   bash scripts/run_baseline_gpt52.sh

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/baseline"
LOGFILE="logs/baseline_gpt52_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

GPT52="openai/gpt-5.2"

echo "=== Baseline Animal Preference Run (gpt-5.2) ===" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOGFILE"
echo "=================================================" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo ">>> Run 1/1: model=gpt-5.2" | tee -a "$LOGFILE"
echo ">>> Started at: $(date)" | tee -a "$LOGFILE"

uv run inspect eval src/subtext_bench/tasks/baseline.py \
  --model "$GPT52" \
  --log-dir "$LOG_DIR" \
  -T reasoning_effort=none \
  --max-connections 100 \
  2>&1 | tee -a "$LOGFILE"

echo ">>> Finished run 1/1 at $(date)" | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "=== Baseline run complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "Logs in: $LOG_DIR" | tee -a "$LOGFILE"
