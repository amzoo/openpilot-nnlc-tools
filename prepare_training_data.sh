#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults
DEVICE=""
OUTPUT_DIR="./output"
DATA_DIR="./data"
MIN_SCORE=""
PRUNE="both"

usage() {
  echo "Usage: $(basename "$0") [OPTIONS]"
  echo ""
  echo "Run the full NNLC data preparation pipeline: sync → extract → score → prune routes → visualize → classify & prune"
  echo ""
  echo "Options:"
  echo "  -d, --device IP              Device IP for rlog sync (skip sync if omitted)"
  echo "  -o, --output DIR             Output directory (default: ./output)"
  echo "  --data-dir DIR               Rlog directory (default: ./data)"
  echo "  --min-score N                Exclude routes scoring below N in prune step (default: 0)"
  echo "  --prune mechanical|driver|both|none"
  echo "                               Event types to remove in step 6 (default: both)"
  echo "                               Use 'none' to skip the prune step"
  echo "  -h, --help                   Show this help"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--device)  DEVICE="$2"; shift 2 ;;
    -o|--output)  OUTPUT_DIR="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --min-score)  MIN_SCORE="$2"; shift 2 ;;
    --prune)      PRUNE="$2"; shift 2 ;;
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
  banner "Step 1/6: Syncing rlogs from $DEVICE"
  uv run nnlc-sync -d "$DEVICE" -o "$DATA_DIR"
else
  banner "Step 1/6: Sync skipped (no --device specified)"
  echo "Using existing rlogs in $DATA_DIR"
fi

# Step 2: Extract lateral data with temporal features
banner "Step 2/6: Extracting lateral data"
uv run nnlc-extract "$DATA_DIR" -o "$OUTPUT_DIR/lateral_data.csv" --temporal

# Step 3: Score routes
banner "Step 3/6: Scoring routes"
SCORE_ARGS=("$OUTPUT_DIR/lateral_data.csv")
if [[ -n "$MIN_SCORE" ]]; then
  SCORE_ARGS+=(--min-score "$MIN_SCORE")
fi
uv run nnlc-score "${SCORE_ARGS[@]}"

# Step 4: Prune routes
banner "Step 4/6: Pruning routes"
uv run nnlc-prune-routes "$OUTPUT_DIR/lateral_data.csv" \
    -o "$OUTPUT_DIR/lateral_data_routes_pruned.csv" \
    ${MIN_SCORE:+--min-score "$MIN_SCORE"}

# Step 5: Visualize coverage
banner "Step 5/6: Visualizing data coverage"
uv run nnlc-visualize "$OUTPUT_DIR/lateral_data_routes_pruned.csv" -o "$OUTPUT_DIR/coverage.png"

# Step 6: Classify interventions and prune
if [[ "$PRUNE" != "none" ]]; then
  banner "Step 6/6: Classifying interventions and pruning ($PRUNE)"
  uv run nnlc-interventions "$OUTPUT_DIR/lateral_data_routes_pruned.csv" \
      --prune "$PRUNE" \
      --prune-output "$OUTPUT_DIR/lateral_data_pruned.csv"
  TRAIN_INPUT="$OUTPUT_DIR/lateral_data_pruned.csv"
else
  banner "Step 6/6: Prune skipped (--prune none)"
  TRAIN_INPUT="$OUTPUT_DIR/lateral_data_routes_pruned.csv"
fi

banner "Pipeline complete!"
echo "Outputs:"
echo "  Data (raw):          $OUTPUT_DIR/lateral_data.csv"
echo "  Data (route-pruned): $OUTPUT_DIR/lateral_data_routes_pruned.csv"
if [[ "$PRUNE" != "none" ]]; then
  echo "  Data (pruned):       $OUTPUT_DIR/lateral_data_pruned.csv"
fi
echo "  Coverage plot:       $OUTPUT_DIR/coverage.png"
echo ""
echo "Next step: review coverage.png for gaps, then train:"
echo "  bash training/run.sh $TRAIN_INPUT"
