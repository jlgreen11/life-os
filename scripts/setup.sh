#!/usr/bin/env bash
# ===========================================================================
# Life OS — First-Run Setup Script
# ===========================================================================
set -euo pipefail

echo "============================================"
echo "  Life OS — Setup"
echo "============================================"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

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

# Create config if it doesn't exist
if [ ! -f "config/settings.yaml" ]; then
    echo "Creating config/settings.yaml from example..."
    cp config/settings.example.yaml config/settings.yaml
    echo "  → Edit config/settings.yaml with your credentials"
    echo ""
fi

# Create data directory
mkdir -p data

# Pull Ollama model
echo "Pulling Ollama model (this may take a few minutes the first time)..."
echo "Starting Ollama container..."
docker compose up -d ollama
sleep 5

echo "Pulling mistral model..."
docker exec lifeos-ollama ollama pull mistral 2>/dev/null || {
    echo "  ℹ  Will pull model when Ollama is ready"
}

echo ""

# Start all services
echo "Starting Life OS..."
docker compose up -d

echo ""
echo "Waiting for services to start..."
sleep 10

# Health check
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
