#!/usr/bin/env bash
# Run 7 prefillable models on the direct task (linkedin + journal only).
# Monitor: openai/gpt-5-mini with reasoning_effort=minimal.
# Sender reasoning is off by default for all these models.
#
# Usage:
#   bash scripts/run_prefill_direct.sh

set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="logs/prefill_direct_${TIMESTAMP}.log"
mkdir -p logs

MODELS=(
  "anthropic/claude-opus-4-5"
  "anthropic/claude-sonnet-4-5"
  "anthropic/claude-haiku-4-5"
  "mistral/mistral-large-latest"
  "mistral/mistral-small-latest"
  "openai/deepseek-chat"
  "openai/deepseek-reasoner"
)

MONITOR="openai/gpt-5-mini"

# 16 animals x 2 tasks (linkedin + journal) = 32 sample IDs
SAMPLE_IDS="dog__linkedin,elephant__linkedin,panda__linkedin,cat__linkedin,dragon__linkedin,lion__linkedin,eagle__linkedin,dolphin__linkedin,tiger__linkedin,wolf__linkedin,phoenix__linkedin,bear__linkedin,fox__linkedin,leopard__linkedin,whale__linkedin,owl__linkedin,dog__journal,elephant__journal,panda__journal,cat__journal,dragon__journal,lion__journal,eagle__journal,dolphin__journal,tiger__journal,wolf__journal,phoenix__journal,bear__journal,fox__journal,leopard__journal,whale__journal,owl__journal"

echo "=== Prefill Direct Task Run ===" | tee -a "$LOGFILE"
echo "Started: $(date)" | tee -a "$LOGFILE"
echo "Log file: $LOGFILE" | tee -a "$LOGFILE"
echo "Monitor: $MONITOR (reasoning_effort=minimal)" | tee -a "$LOGFILE"
echo "Tasks: linkedin, journal (32 samples per model)" | tee -a "$LOGFILE"
echo "Models: ${MODELS[*]}" | tee -a "$LOGFILE"
echo "================================" | tee -a "$LOGFILE"

for MODEL in "${MODELS[@]}"; do
  echo "" | tee -a "$LOGFILE"
  echo ">>> Running model: $MODEL" | tee -a "$LOGFILE"
  echo ">>> Started at: $(date)" | tee -a "$LOGFILE"

  uv run inspect eval src/subtext_bench/tasks/direct.py \
    --model "$MODEL" \
    --model-role "monitor=$MONITOR" \
    --sample-id "$SAMPLE_IDS" \
    -T monitor_reasoning_effort=minimal \
    --max-connections 100 \
    2>&1 | tee -a "$LOGFILE"

  echo ">>> Finished model: $MODEL at $(date)" | tee -a "$LOGFILE"
done

echo "" | tee -a "$LOGFILE"
echo "=== All runs complete ===" | tee -a "$LOGFILE"
echo "Finished: $(date)" | tee -a "$LOGFILE"
