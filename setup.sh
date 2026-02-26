#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Creating virtual environment..."
uv venv

echo "Installing dependencies..."
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .

echo ""
echo "Done! Activate the environment with:"
echo "  source .venv/bin/activate"
