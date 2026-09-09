#!/bin/bash

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo "================================================"
echo "   RoleLens — Setup & Run"
echo "================================================"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "► Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at .venv/"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Install requirements
echo "► Installing requirements..."
pip install --upgrade pip -q
pip install -r "$PROJECT_DIR/assets/requirements.txt" -q
echo "✓ Requirements installed"

# Check .env file
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo ""
    echo "⚠ Warning: .env file not found at project root."
    echo "  Please create it and add your API key:"
    echo "  GROQ_API_KEY=your_key_here"
    echo ""
fi

echo ""
echo "================================================"
echo "   Running ingest pipeline..."
echo "================================================"
python3 "$PROJECT_DIR/src/loaders/ingest.py"

echo ""
echo "================================================"
echo "   Done!"
echo "================================================"
