#!/usr/bin/env bash
# Run 9 cross eval runs: all (sender, receiver) pairs from {Haiku, Sonnet, Opus}^2.
# Monitor is always gpt-5.2 with reasoning_effort=minimal.
# Direct task, linkedin + journal only.
#
# Usage:
#   bash scripts/run_gpt52_monitor_cross.sh

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/gpt52_monitor"
LOGFILE="logs/gpt52_monitor_run_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

HAIKU="anthropic/claude-haiku-4-5"
SONNET="anthropic/claude-sonnet-4-5"
OPUS="anthropic/claude-opus-4-5"
MONITOR="openai/gpt-5.2"

SAMPLE_IDS="dog__linkedin,elephant__linkedin,panda__linkedin,cat__linkedin,dragon__linkedin,lion__linkedin,eagle__linkedin,dolphin__linkedin,tiger__linkedin,wolf__linkedin,phoenix__linkedin,bear__linkedin,fox__linkedin,leopard__linkedin,whale__linkedin,owl__linkedin,dog__journal,elephant__journal,panda__journal,cat__journal,dragon__journal,lion__journal,eagle__journal,dolphin__journal,tiger__journal,wolf__journal,phoenix__journal,bear__journal,fox__journal,leopard__journal,whale__journal,owl__journal"

# All 9 (sender, receiver) pairs from {haiku, sonnet, opus}^2
declare -a SENDER_NAMES=("haiku"  "haiku"  "haiku" "sonnet" "sonnet" "sonnet" "opus"  "opus"   "opus")
declare -a RECV_NAMES=(  "haiku"  "sonnet" "opus"  "haiku"  "sonnet" "opus"   "haiku" "sonnet" "opus")
declare -a SENDERS=(     "$HAIKU" "$HAIKU" "$HAIKU" "$SONNET" "$SONNET" "$SONNET" "$OPUS" "$OPUS" "$OPUS")
declare -a RECEIVERS=(   "$HAIKU" "$SONNET" "$OPUS" "$HAIKU"  "$SONNET" "$OPUS"   "$HAIKU" "$SONNET" "$OPUS")

TOTAL=${#SENDERS[@]}

echo "=== GPT-5.2 Monitor Cross Eval Runs ===" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOGFILE"
echo "Monitor: $MONITOR (reasoning_effort=none)" | tee -a "$LOGFILE"
echo "Tasks: linkedin, journal (32 samples per run)" | tee -a "$LOGFILE"
echo "$TOTAL runs: all (sender, receiver) in {haiku, sonnet, opus}^2" | tee -a "$LOGFILE"
echo "========================================" | tee -a "$LOGFILE"

for i in "${!SENDERS[@]}"; do
  SENDER="${SENDERS[$i]}"
  RECEIVER="${RECEIVERS[$i]}"
  S_NAME="${SENDER_NAMES[$i]}"
  R_NAME="${RECV_NAMES[$i]}"
  TAG="S-${S_NAME}_R-${R_NAME}_M-gpt52"

  echo "" | tee -a "$LOGFILE"
  echo ">>> Run $((i+1))/$TOTAL: sender=$S_NAME, receiver=$R_NAME, monitor=gpt-5.2" | tee -a "$LOGFILE"
  echo ">>> Tag: $TAG" | tee -a "$LOGFILE"
  echo ">>> Started at: $(date)" | tee -a "$LOGFILE"

  uv run inspect eval src/subtext_bench/tasks/direct.py \
    --model "$SENDER" \
    --model-role "receiver=$RECEIVER" \
    --model-role "monitor=$MONITOR" \
    --sample-id "$SAMPLE_IDS" \
    --tags "$TAG" \
    --log-dir "$LOG_DIR" \
    -T monitor_reasoning_effort=none \
    --max-connections 100 \
    2>&1 | tee -a "$LOGFILE"

  echo ">>> Finished run $((i+1))/$TOTAL at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=== All $TOTAL runs complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "Logs in: $LOG_DIR" | tee -a "$LOGFILE"
