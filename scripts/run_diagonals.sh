#!/usr/bin/env bash
# Run 3 diagonal eval runs: sender=receiver for each of Haiku/Sonnet/Opus.
# Monitor is always gpt-5-mini with reasoning_effort=minimal.
# Direct task, linkedin + journal only.
#
# Usage:
#   bash scripts/run_diagonals.sh

set -uo pipefail

LOG_DIR="logs/cross_sender_receiver"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/diagonals_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"

HAIKU="anthropic/claude-haiku-4-5"
SONNET="anthropic/claude-sonnet-4-5"
OPUS="anthropic/claude-opus-4-5"
MONITOR="openai/gpt-5-mini"

SAMPLE_IDS="dog__linkedin,elephant__linkedin,panda__linkedin,cat__linkedin,dragon__linkedin,lion__linkedin,eagle__linkedin,dolphin__linkedin,tiger__linkedin,wolf__linkedin,phoenix__linkedin,bear__linkedin,fox__linkedin,leopard__linkedin,whale__linkedin,owl__linkedin,dog__journal,elephant__journal,panda__journal,cat__journal,dragon__journal,lion__journal,eagle__journal,dolphin__journal,tiger__journal,wolf__journal,phoenix__journal,bear__journal,fox__journal,leopard__journal,whale__journal,owl__journal"

declare -a NAMES=("haiku" "sonnet" "opus")
declare -a MODELS=("$HAIKU" "$SONNET" "$OPUS")

echo "=== Diagonal Eval Runs (sender=receiver) ===" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "Monitor: $MONITOR (reasoning_effort=minimal)" | tee -a "$LOGFILE"
echo "==============================================" | tee -a "$LOGFILE"

for i in "${!MODELS[@]}"; do
  MODEL="${MODELS[$i]}"
  NAME="${NAMES[$i]}"
  TAG="S-${NAME}_R-${NAME}"

  echo "" | tee -a "$LOGFILE"
  echo ">>> Run $((i+1))/3: sender=receiver=$NAME, monitor=gpt-5-mini" | tee -a "$LOGFILE"
  echo ">>> Tag: $TAG" | tee -a "$LOGFILE"
  echo ">>> Started at: $(date)" | tee -a "$LOGFILE"

  uv run inspect eval src/subtext_bench/tasks/direct.py \
    --model "$MODEL" \
    --model-role "receiver=$MODEL" \
    --model-role "monitor=$MONITOR" \
    --sample-id "$SAMPLE_IDS" \
    --tags "$TAG" \
    --log-dir "$LOG_DIR" \
    -T monitor_reasoning_effort=minimal \
    --max-connections 100 \
    2>&1 | tee -a "$LOGFILE"

  echo ">>> Finished run $((i+1))/3 at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=== All 3 diagonal runs complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
