#!/bin/bash
# check_environment.sh
# Run this on the Linux cluster to diagnose issues.

echo "=== System Info ==="
uname -a

echo -e "\n=== Python Check ==="
python3 --version || echo "Python3 not found!"

echo -e "\n=== Ollama Check ==="
if command -v ollama &> /dev/null; then
    echo "Ollama binary found: $(which ollama)"
else
    echo "WARNING: 'ollama' command not found in PATH."
fi

echo -e "\n=== Connectivity Check (localhost:11434) ==="
# Try to connect to Ollama
if curl -s http://localhost:11434 > /dev/null; then
    echo "SUCCESS: Connected to Ollama server."
else
    echo "ERROR: Could not connect to http://localhost:11434"
    echo "Possible causes:"
    echo "1. Ollama is not running. Run 'ollama serve > ollama.log 2>&1 &' to start it."
    echo "2. Ollama is running on a different port."
    echo "3. Firewall blocking localhost (unlikely but possible)."
fi

echo -e "\n=== Process Check ==="
if pgrep -x "ollama" > /dev/null; then
    echo "Ollama process is RUNNING (PID: $(pgrep -x ollama))"
else
    echo "Ollama process is NOT running."
fi

echo -e "\n=== Model Check ==="
if curl -s http://localhost:11434 > /dev/null; then
    echo "Listing available models:"
    ollama list
fi
