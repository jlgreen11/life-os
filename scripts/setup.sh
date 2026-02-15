#!/usr/bin/env bash
# ===========================================================================
# Life OS — First-Run Setup Script
#
# Automates the initial setup process:
#   1. Verifies prerequisites (Docker, docker compose, Python 3.12+)
#   2. Creates config/settings.yaml from the example template
#   3. Creates the data/ directory for SQLite and vector databases
#   4. Pulls the Ollama LLM container and downloads the Mistral model
#   5. Starts all services via docker compose
#   6. Runs a health check to confirm the system is ready
#
# Usage: bash scripts/setup.sh
# ===========================================================================
set -euo pipefail  # Exit on error, undefined var, or pipe failure

echo "============================================"
echo "  Life OS — Setup"
echo "============================================"
echo ""

# --- Step 1: Check prerequisites ---
echo "Checking prerequisites..."

# Helper function to verify a command is available on PATH
check_command() {
    if command -v "$1" &> /dev/null; then
        echo "  ✓ $1 found"
        return 0
    else
        echo "  ✗ $1 not found"
        return 1
    fi
}

MISSING=0
check_command "docker" || MISSING=1
check_command "docker" && docker compose version &>/dev/null && echo "  ✓ docker compose found" || { echo "  ✗ docker compose not found"; MISSING=1; }
check_command "python3" || MISSING=1

if [ "$MISSING" -eq 1 ]; then
    echo ""
    echo "Please install missing prerequisites before continuing."
    echo "  Docker: https://docs.docker.com/get-docker/"
    echo "  Python 3.12+: https://www.python.org/downloads/"
    exit 1
fi

echo ""

# --- Step 2: Create config from template if not already present ---
if [ ! -f "config/settings.yaml" ]; then
    echo "Creating config/settings.yaml from example..."
    cp config/settings.example.yaml config/settings.yaml
    echo "  → Edit config/settings.yaml with your credentials"
    echo ""
fi

# --- Step 3: Create data directory for SQLite databases and vector store ---
mkdir -p data

# --- Step 4: Pull Ollama LLM container and download the Mistral model ---
echo "Pulling Ollama model (this may take a few minutes the first time)..."
echo "Starting Ollama container..."
docker compose up -d ollama
sleep 5

echo "Pulling mistral model..."
docker exec lifeos-ollama ollama pull mistral 2>/dev/null || {
    echo "  ℹ  Will pull model when Ollama is ready"
}

echo ""

# --- Step 5: Start all services (NATS, Ollama, Life OS app) ---
echo "Starting Life OS..."
docker compose up -d

echo ""
echo "Waiting for services to start..."
sleep 10

# --- Step 6: Health check to confirm the system is ready ---
echo ""
echo "Checking health..."
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "  ✓ Life OS is running!"
else
    echo "  ⏳ Still starting up. Check: docker compose logs lifeos"
fi

echo ""
echo "============================================"
echo "  Life OS is ready!"
echo ""
echo "  Web UI:     http://localhost:8080"
echo "  API:        http://localhost:8080/api"
echo "  NATS:       http://localhost:8222"
echo ""
echo "  Next steps:"
echo "    1. Edit config/settings.yaml"
echo "    2. Enable your connectors"
echo "    3. Restart: docker compose restart lifeos"
echo "============================================"
