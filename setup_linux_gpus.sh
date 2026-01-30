#!/bin/bash
# ============================================================
# Setup Script for Linux Lab with Dual NVIDIA GPUs
# ============================================================
# Run this once on your Linux lab machine to set up everything
# Usage: chmod +x setup_linux_gpus.sh && ./setup_linux_gpus.sh
# ============================================================

set -e

echo "=================================================="
echo "SemEval High-Accuracy Setup for Dual-GPU Linux"
echo "=================================================="

# Check for NVIDIA GPUs
echo ""
echo "[1/6] Checking NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --list-gpus
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $GPU_COUNT GPU(s)"
else
    echo "WARNING: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    GPU_COUNT=0
fi

# Install Ollama if not present
echo ""
echo "[2/6] Setting up Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed: $(ollama --version)"
fi

# Check system memory
echo ""
echo "[3/6] Checking system resources..."
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "Total RAM: ${TOTAL_MEM}GB"

# Determine which model to use based on VRAM
echo ""
echo "[4/6] Selecting optimal model..."

# Try to get GPU memory
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "GPU Memory (first GPU): ${GPU_MEM}MB"
    
    # Calculate total VRAM across all GPUs
    TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    echo "Total VRAM: ${TOTAL_VRAM}MB"
    
    if [ "$TOTAL_VRAM" -ge 80000 ]; then
        # 80GB+ VRAM - can run 70B with good context
        MODEL="llama3.1:70b"
        echo "Selected model: $MODEL (requires ~40GB+ VRAM)"
    elif [ "$TOTAL_VRAM" -ge 48000 ]; then
        # 48GB+ VRAM - can run 70B quantized
        MODEL="llama3.1:70b-instruct-q4_0"
        echo "Selected model: $MODEL (quantized, requires ~35GB VRAM)"
    elif [ "$TOTAL_VRAM" -ge 16000 ]; then
        # 16GB+ VRAM - use 8B model
        MODEL="llama3.1:8b"
        echo "Selected model: $MODEL (requires ~8GB VRAM)"
    else
        MODEL="llama3.1:8b-instruct-q4_0"
        echo "Selected model: $MODEL (quantized for low VRAM)"
    fi
else
    MODEL="llama3.1:8b"
    echo "Cannot detect VRAM. Defaulting to: $MODEL"
fi

# Pull the model
echo ""
echo "[5/6] Pulling model: $MODEL (this may take a while)..."
ollama pull $MODEL

# Create configuration for dual-GPU setup
echo ""
echo "[6/6] Creating run configuration..."

# Create the high-accuracy run script
cat > run_high_accuracy_v2.sh << 'SCRIPT_EOF'
#!/bin/bash
# ============================================================
# High-Accuracy Run Script for SemEval (v2)
# ============================================================
# Usage: ./run_high_accuracy_v2.sh [dev|test]
# ============================================================

set -e

MODE=${1:-test}
MODEL=${MODEL:-"llama3.1:70b"}
ENSEMBLE=${ENSEMBLE:-9}

echo "=================================================="
echo "High-Accuracy SemEval Run (v2)"
echo "Mode: $MODE | Model: $MODEL | Ensemble: $ENSEMBLE"
echo "=================================================="

# Determine input/output files
if [ "$MODE" == "dev" ]; then
    INPUT="data/dev.json"
    OUTPUT="predictions/dev_v2.jsonl"
    echo "Running on DEV set for validation..."
else
    INPUT="data/test.json"
    OUTPUT="predictions/submission_v2.jsonl"
    echo "Running on TEST set for submission..."
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    ollama serve > ollama_server.log 2>&1 &
    sleep 5
fi

# Verify model is available
echo "Checking model availability..."
if ! ollama list | grep -q "$MODEL"; then
    echo "Model $MODEL not found. Pulling..."
    ollama pull $MODEL
fi

# Run the enhanced reasoning script
echo ""
echo "Starting inference..."
start_time=$(date +%s)

python3 generative_reasoning_v2.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --model "$MODEL" \
    --ensemble $ENSEMBLE \
    --self-correction \
    --train-data data/train.json \
    --aggregation median

end_time=$(date +%s)
duration=$((end_time - start_time))
echo ""
echo "Inference completed in ${duration}s"

# If dev mode, run evaluation
if [ "$MODE" == "dev" ]; then
    echo ""
    echo "Running evaluation..."
    
    # Create simple submission format (remove reasoning)
    python3 -c "
import json
with open('$OUTPUT', 'r') as f_in, open('predictions/submission_v2_eval.jsonl', 'w') as f_out:
    for line in f_in:
        rec = json.loads(line)
        f_out.write(json.dumps({'id': rec['id'], 'prediction': rec['prediction']}) + '\n')
"
    
    python3 scoring.py data/dev_ref.jsonl predictions/submission_v2_eval.jsonl output/scores_v2.json
    
    echo ""
    echo "Results:"
    cat output/scores_v2.json
fi

# For test mode, create submission zip
if [ "$MODE" == "test" ]; then
    echo ""
    echo "Creating submission file..."
    
    # Create clean submission format
    python3 -c "
import json
with open('$OUTPUT', 'r') as f_in, open('predictions/final_submission.jsonl', 'w') as f_out:
    for line in f_in:
        rec = json.loads(line)
        f_out.write(json.dumps({'id': rec['id'], 'prediction': rec['prediction']}) + '\n')
"
    
    cd predictions
    zip -f submission_v2.zip final_submission.jsonl 2>/dev/null || zip submission_v2.zip final_submission.jsonl
    cd ..
    
    echo ""
    echo "=================================================="
    echo "Submission ready: predictions/submission_v2.zip"
    echo "=================================================="
fi
SCRIPT_EOF

chmod +x run_high_accuracy_v2.sh

# Save selected model to config
echo "$MODEL" > .model_config

echo ""
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Selected Model: $MODEL"
echo ""
echo "To run on DEV set (for validation):"
echo "  ./run_high_accuracy_v2.sh dev"
echo ""
echo "To run on TEST set (for submission):"
echo "  ./run_high_accuracy_v2.sh test"
echo ""
echo "To override model:"
echo "  MODEL=llama3.1:8b ./run_high_accuracy_v2.sh dev"
echo ""
echo "To override ensemble size:"
echo "  ENSEMBLE=15 ./run_high_accuracy_v2.sh test"
echo ""
