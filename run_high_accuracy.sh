#!/bin/bash
# Script to run High Accuracy configuration on HPC/Research Center
# Usage: ./run_high_accuracy.sh

# Ensure the script stops on error
set -e

echo "Starting High Accuracy Run with Llama 3 70B..."

# Run with:
# - Model: llama3:70b (Make sure this is pulled in ollama: `ollama pull llama3:70b`)
# - Ensemble: 5 votes
# - Self-Correction: Enabled
# - data: test.json (Official Test Set)

python3 generative_reasoning.py \
    --model llama3:70b \
    --ensemble 5 \
    --self-correction \
    --input data/test.json \
    --output predictions/high_acc_submission.jsonl

echo "Run complete. Output saved to predictions/high_acc_submission.jsonl"
