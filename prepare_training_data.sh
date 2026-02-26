#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults
DEVICE=""
OUTPUT_DIR="./output"
DATA_DIR="./data"
MIN_SCORE=""

usage() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo ""
  echo "Run the full NNLC data preparation pipeline: sync → extract → score → visualize"
  echo ""
  echo "Options:"
  echo "  -d, --device IP      Device IP for rlog sync (skip sync if omitted)"
  echo "  -o, --output DIR     Output directory (default: ./output)"
  echo "  --data-dir DIR       Rlog directory (default: ./data)"
  echo "  --min-score N        Only show routes scoring >= N"
  echo "  -h, --help           Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device)  DEVICE="$2"; shift 2 ;;
    -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --min-score)  MIN_SCORE="$2"; shift 2 ;;
    -h|--help)    usage ;;
    *)            echo "Unknown option: $1"; usage ;;
  esac
done

banner() {
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════"
  echo ""
}

mkdir -p "$OUTPUT_DIR"

# Step 1: Sync rlogs from device
if [[ -n "$DEVICE" ]]; then
  banner "Step 1/4: Syncing rlogs from $DEVICE"
  uv run nnlc-sync -d "$DEVICE" -o "$DATA_DIR"
else
  banner "Step 1/4: Sync skipped (no --device specified)"
  echo "Using existing rlogs in $DATA_DIR"
fi

# Step 2: Extract lateral data with temporal features
banner "Step 2/4: Extracting lateral data"
uv run nnlc-extract "$DATA_DIR" -o "$OUTPUT_DIR/lateral_data.csv" --temporal

# Step 3: Score routes
banner "Step 3/4: Scoring routes"
SCORE_ARGS=("$OUTPUT_DIR/lateral_data.csv")
if [[ -n "$MIN_SCORE" ]]; then
  SCORE_ARGS+=(--min-score "$MIN_SCORE")
fi
uv run nnlc-score "${SCORE_ARGS[@]}"

# Step 4: Visualize coverage
banner "Step 4/4: Visualizing data coverage"
uv run nnlc-visualize "$OUTPUT_DIR/lateral_data.csv" -o "$OUTPUT_DIR/coverage.png"

banner "Pipeline complete!"
echo "Outputs:"
echo "  Data:          $OUTPUT_DIR/lateral_data.csv"
echo "  Coverage plot: $OUTPUT_DIR/coverage.png"
echo ""
echo "Next step: review coverage.png for gaps, then train:"
echo "  bash training/run.sh $OUTPUT_DIR/lateral_data.csv"
