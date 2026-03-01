#!/usr/bin/env bash
set -euo pipefail

REPO=$(find "$HOME" -maxdepth 4 -name "openpilot-nnlc-tools" -type d 2>/dev/null | head -1)
if [[ -z "$REPO" ]]; then
  osascript -e 'display alert "Could not find openpilot-nnlc-tools repo under home directory."'
  exit 1
fi
cd "$REPO"

# Ask for device IP via GUI dialog
DEVICE=$(osascript -e 'Tell application "System Events" to display dialog "Enter comma device IP:" default answer "192.168.1.161"' -e 'text returned of result' 2>/dev/null) || {
  echo "Cancelled."
  exit 0
}

if [[ -z "$DEVICE" ]]; then
  echo "No IP entered. Exiting."
  exit 0
fi

echo "Syncing rlogs from $DEVICE..."
uv run nnlc-sync -d "$DEVICE" -o ./data

echo ""
echo "Sync complete. Data is in ./data/"
echo "(Press any key to close)"
read -n 1 -s
