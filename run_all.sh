#!/bin/bash
# ============================================================
# ALL-IN-ONE Script for SemEval High-Accuracy Submission
# ============================================================
# This script does EVERYTHING:
# 1. Setup (install Ollama, pull model)
# 2. Run inference on test set
# 3. Create submission zip
#
# Usage on Linux lab machine:
#   chmod +x run_all.sh
#   ./run_all.sh
# ============================================================

set -e

echo "=============================================="
echo "  SemEval AmbiStory - High Accuracy Pipeline"
echo "=============================================="
echo ""

# Configuration - adjust these if needed
MODEL="${MODEL:-llama3.1:70b-instruct-q4_0}"
ENSEMBLE="${ENSEMBLE:-9}"
INPUT="data/test.json"
OUTPUT="predictions/submission_final.jsonl"

# ==================== STEP 1: SETUP ====================
echo "[STEP 1/5] Checking environment..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python."
    exit 1
fi
echo "  ✓ Python3 found"

# Check for required packages
python3 -c "import scipy" 2>/dev/null || pip3 install scipy --quiet
python3 -c "import numpy" 2>/dev/null || pip3 install numpy --quiet
python3 -c "import sklearn" 2>/dev/null || pip3 install scikit-learn --quiet
echo "  ✓ Python packages OK"

# Install Ollama if needed
if ! command -v ollama &> /dev/null; then
    echo "  Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi
echo "  ✓ Ollama installed"

# ==================== STEP 2: START OLLAMA ====================
echo ""
echo "[STEP 2/5] Starting Ollama server..."

# Check if Ollama is already running
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama already running"
else
    echo "  Starting Ollama server..."
    nohup ollama serve > ollama_server.log 2>&1 &
    sleep 5
    
    # Verify it started
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✓ Ollama started"
    else
        echo "  ERROR: Failed to start Ollama. Check ollama_server.log"
        exit 1
    fi
fi

# ==================== STEP 3: PULL MODEL ====================
echo ""
echo "[STEP 3/5] Ensuring model is available: $MODEL"

# Check GPU availability and adjust model if needed
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    echo "  GPU VRAM: ${GPU_MEM}MB"
    
    # Auto-select model based on VRAM
    if [ "$GPU_MEM" -lt 45000 ] && [ "$MODEL" == "llama3.1:70b" ]; then
        echo "  WARNING: Less than 45GB VRAM detected. Switching to llama3.1:8b"
        MODEL="llama3.1:8b"
    fi
else
    echo "  No NVIDIA GPU detected - using CPU (will be slow)"
    MODEL="llama3.1:8b"
fi

# Pull model if not present
if ! ollama list | grep -q "${MODEL%%:*}"; then
    echo "  Pulling $MODEL (this may take a while)..."
    ollama pull "$MODEL"
else
    echo "  ✓ Model already available"
fi

# ==================== STEP 4: RUN INFERENCE ====================
echo ""
echo "[STEP 4/5] Running inference..."
echo "  Model: $MODEL"
echo "  Ensemble: $ENSEMBLE votes"
echo "  Input: $INPUT"
echo "  Output: $OUTPUT"
echo ""

# Check input file exists
if [ ! -f "$INPUT" ]; then
    echo "ERROR: Input file not found: $INPUT"
    exit 1
fi

# Count samples
SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('$INPUT'))))")
echo "  Processing $SAMPLE_COUNT samples..."
echo ""

# Clear output if starting fresh
# rm -f "$OUTPUT"  # Uncomment to start fresh each time

START_TIME=$(date +%s)

python3 generative_reasoning_v2.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --ensemble "$ENSEMBLE" \
    --self-correction \
    --train-data data/train.json \
    --aggregation median

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "  ✓ Inference complete in ${DURATION}s"

# ==================== STEP 5: CREATE SUBMISSION ====================
echo ""
echo "[STEP 5/5] Creating submission..."

# Prepare clean submission file
python3 prepare_submission.py "$OUTPUT" predictions/submission.jsonl

# Create zip
cd predictions
rm -f submission.zip
zip submission.zip submission.jsonl
cd ..

# Verify
PRED_COUNT=$(wc -l < predictions/submission.jsonl | tr -d ' ')
echo ""
echo "=============================================="
echo "  DONE! Submission ready"
echo "=============================================="
echo ""
echo "  Predictions: $PRED_COUNT samples"
echo "  Submission file: predictions/submission.zip"
echo ""
echo "  Upload to: https://www.codabench.org/competitions/10877/"
echo ""

# If dev set exists, offer to evaluate
if [ -f "data/dev_ref.jsonl" ] && [ "$INPUT" == "data/dev.json" ]; then
    echo "  Running evaluation on dev set..."
    python3 scoring.py data/dev_ref.jsonl predictions/submission.jsonl output/scores.json
    echo ""
    echo "  Results:"
    cat output/scores.json
    echo ""
fi
