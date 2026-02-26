#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv &> /dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Creating virtual environment..."
uv venv --clear

echo "Installing dependencies..."
uv pip install -e ".[dev]"

echo ""
echo "Done! Run tools with:"
echo "  uv run nnlc-extract ./data -o output/lateral_data.csv --temporal"
echo ""
echo "Or run the full pipeline:"
echo "  bash prepare_training_data.sh"
