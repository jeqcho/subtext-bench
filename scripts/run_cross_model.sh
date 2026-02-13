#!/usr/bin/env bash
# Run 6 cross-model eval runs: each of Haiku/Sonnet/Opus takes turns
# being sender+receiver vs monitor on the direct task (linkedin + journal).
#
# Logs go into logs/cross_model/ with descriptive tags.
#
# Usage:
#   bash scripts/run_cross_model.sh

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/cross_model"
LOGFILE="logs/cross_model_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

HAIKU="anthropic/claude-haiku-4-5"
SONNET="anthropic/claude-sonnet-4-5"
OPUS="anthropic/claude-opus-4-5"

SAMPLE_IDS="dog__linkedin,elephant__linkedin,panda__linkedin,cat__linkedin,dragon__linkedin,lion__linkedin,eagle__linkedin,dolphin__linkedin,tiger__linkedin,wolf__linkedin,phoenix__linkedin,bear__linkedin,fox__linkedin,leopard__linkedin,whale__linkedin,owl__linkedin,dog__journal,elephant__journal,panda__journal,cat__journal,dragon__journal,lion__journal,eagle__journal,dolphin__journal,tiger__journal,wolf__journal,phoenix__journal,bear__journal,fox__journal,leopard__journal,whale__journal,owl__journal"

# All 6 permutations of (sender+receiver, monitor)
declare -a SENDER_NAMES=("haiku"  "haiku" "sonnet" "sonnet" "opus"  "opus")
declare -a MONITOR_NAMES=("sonnet" "opus"  "haiku"  "opus"   "haiku" "sonnet")
declare -a SENDERS=("$HAIKU"  "$HAIKU" "$SONNET" "$SONNET" "$OPUS"  "$OPUS")
declare -a MONITORS=("$SONNET" "$OPUS"  "$HAIKU"  "$OPUS"   "$HAIKU" "$SONNET")

echo "=== Cross-Model Eval Runs ===" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "Log dir: $LOG_DIR" | tee -a "$LOGFILE"
echo "Tasks: linkedin, journal (32 samples per run)" | tee -a "$LOGFILE"
echo "6 runs: sender+receiver vs monitor" | tee -a "$LOGFILE"
echo "==============================" | tee -a "$LOGFILE"

for i in "${!SENDERS[@]}"; do
  SENDER="${SENDERS[$i]}"
  MONITOR="${MONITORS[$i]}"
  S_NAME="${SENDER_NAMES[$i]}"
  M_NAME="${MONITOR_NAMES[$i]}"
  TAG="sender-${S_NAME}_monitor-${M_NAME}"

  echo "" | tee -a "$LOGFILE"
  echo ">>> Run $((i+1))/6: sender+receiver=$S_NAME, monitor=$M_NAME" | tee -a "$LOGFILE"
  echo ">>> Tag: $TAG" | tee -a "$LOGFILE"
  echo ">>> Started at: $(date)" | tee -a "$LOGFILE"

  uv run inspect eval src/subtext_bench/tasks/direct.py \
    --model "$SENDER" \
    --model-role "receiver=$SENDER" \
    --model-role "monitor=$MONITOR" \
    --sample-id "$SAMPLE_IDS" \
    --tags "$TAG" \
    --log-dir "$LOG_DIR" \
    --max-connections 100 \
    2>&1 | tee -a "$LOGFILE"

  echo ">>> Finished run $((i+1))/6 at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=== All 6 runs complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
echo "Logs in: $LOG_DIR" | tee -a "$LOGFILE"
