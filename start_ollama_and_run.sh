#!/bin/bash
# start_ollama_and_run.sh
# Full lifecycle script: Starts Ollama, runs the job, and cleans up.

set -e

# Check if Ollama is already running
if curl -s http://127.0.0.1:11434 > /dev/null; then
    echo "=== Ollama is already running (System Service?) ==="
    OLLAMA_PID=""
else
    echo "=== Starting User-Level Ollama Server... ==="
    # Start user-level ollama server in background
    # Set host explicitly to avoid binding issues
    OLLAMA_HOST=127.0.0.1:11434 ollama serve > ollama_server.log 2>&1 &
    OLLAMA_PID=$!
    echo "Ollama running with PID: $OLLAMA_PID"
    
    echo "=== Waiting for Ollama to be ready... ==="
    MAX_RETRIES=30
    COUNT=0
    URL="http://127.0.0.1:11434"

    until curl -s $URL > /dev/null; do
        echo "Waiting for Ollama... ($COUNT/$MAX_RETRIES)"
        sleep 1
        COUNT=$((COUNT+1))
        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo "ERROR: Ollama failed to start within 30 seconds."
            echo "=== ollama_server.log contents check ==="
            cat ollama_server.log
            echo "========================================"
            exit 1
        fi
    done
fi

echo "SUCCESS: Ollama is reachable."

echo "=== 3. Pulling Model (llama3:70b)... ==="
# This might take a while, but it's necessary. 
# 2 GPUs will be automatically detected by Ollama for inference.
ollama pull llama3:70b

echo "=== 4. Running Prediction Job... ==="
# Using 127.0.0.1 explicitly
./run_high_accuracy.sh --api-url "http://127.0.0.1:11434"

echo "=== 5. Cleanup ==="
if [ -n "$OLLAMA_PID" ]; then
    echo "Stopping user-level Ollama (PID $OLLAMA_PID)..."
    kill $OLLAMA_PID
else
    echo "Leaving system Ollama running."
fi
echo "Done."
