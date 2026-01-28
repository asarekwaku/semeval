#!/bin/bash
# start_ollama_and_run.sh
# Full lifecycle script: Starts Ollama, runs the job, and cleans up.

set -e

echo "=== 1. Starting Ollama Server... ==="
# Start user-level ollama server in background
ollama serve > ollama_server.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama running with PID: $OLLAMA_PID"

echo "=== 2. Waiting for Ollama to be ready... ==="
# Wait up to 30 seconds for the server to start
MAX_RETRIES=30
COUNT=0
URL="http://127.0.0.1:11434"

until curl -s $URL > /dev/null; do
    echo "Waiting for Ollama... ($COUNT/$MAX_RETRIES)"
    sleep 1
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Ollama failed to start within 30 seconds."
        echo "Check ollama_server.log for details."
        kill $OLLAMA_PID
        exit 1
    fi
done
echo "SUCCESS: Ollama is reachable at $URL"

echo "=== 3. Pulling Model (llama3:70b)... ==="
# This might take a while on the first run
ollama pull llama3:70b

echo "=== 4. Running Prediction Job... ==="
# Using 127.0.0.1 explicitly to avoid localhost IPv6 issues
./run_high_accuracy.sh --api-url "$URL"

echo "=== 5. Cleanup ==="
kill $OLLAMA_PID
echo "Done."
