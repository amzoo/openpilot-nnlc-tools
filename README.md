# openpilot NNLC Training Tools

Tools for training NNLC (Neural Network Lateral Control) models for [openpilot](https://github.com/commaai/openpilot) / [sunnypilot](https://github.com/sunnypilot/sunnypilot).

NNLC replaces the standard torque lateral controller with a per-vehicle neural network that learns the relationship between desired lateral acceleration and steering torque. This produces smoother, more accurate steering — but training a model requires collecting driving data, processing it, and running the Julia training pipeline.

This repo consolidates the scattered, broken tooling into one place.

## Prerequisites

- **Python 3.11+**
- **Julia 1.9+** — for model training (with CUDA or Metal GPU recommended)
- **comma device** — for collecting driving data (comma 3/3X)

## Vehicle Compatibility

NNLC models have been trained for:

| Vehicle | Status | Notes |
|---------|--------|-------|
| Chevy Bolt EUV | Merged | Reference implementation |
| Toyota RAV4 | In progress | Torque controller required |
| Honda/Acura | Blocked | EPS signal filtering needed (opendbc) |
| Hyundai Ioniq 6 | Blocked | Rlog parsing issues |
| Mazda | Blocked | Signal compatibility issues |

## Installation

```bash
git clone https://github.com/amzoo/openpilot-nnlc-tools.git
cd openpilot-nnlc-tools

# Create virtual environment with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or install as editable package
uv pip install -e .
```

Or use the setup script (handles everything including uv installation):
```bash
bash setup.sh
source .venv/bin/activate
```

## Docker

Docker eliminates dependency hell (pycapnp builds, Julia packages, CUDA) and makes the pipeline reproducible.

### Build

```bash
docker compose build
```

### Usage

Two services are provided:
- **`tools`** — Python CLI tools (extract, score, visualize). No GPU needed.
- **`train`** — Julia training with NVIDIA GPU passthrough.

Place your rlogs in `./data/` and outputs go to `./output/`.

```bash
# Extract lateral data
docker compose run --rm tools nnlc-extract /app/data -o /app/output/lateral_data.csv --temporal

# Score routes
docker compose run --rm tools nnlc-score /app/output/lateral_data.csv

# Visualize coverage
docker compose run --rm tools nnlc-visualize /app/output/lateral_data.csv -o /app/output/coverage.png

# Train with GPU (requires nvidia-container-toolkit)
docker compose run --rm train bash training/run.sh /app/output/lateral_data.csv

# Train on CPU (no GPU required)
docker compose run --rm tools bash training/run.sh /app/output/lateral_data.csv --cpu

# Run tests
docker compose run --rm tools pytest tests/ -m "not slow"
```

### GPU Support

GPU training in Docker requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host. The `train` service automatically passes through all NVIDIA GPUs.

**Note:** Metal (Apple Silicon) GPU passthrough is not supported in Docker. Mac users should use the native install for GPU training, or Docker with `--cpu`.

## Quick Start

The full pipeline: **sync → extract → score → visualize → train → deploy**

### 1. Sync rlogs from device

```bash
python -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ~/nnlc-data/

# Dry run first to see what would be synced
python -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ~/nnlc-data/ --dry-run
```

### 2. Extract lateral data

```bash
# Basic extraction
python -m nnlc_tools.extract_lateral_data ~/nnlc-data/ -o lateral_data.csv

# With temporal features (required for training)
python -m nnlc_tools.extract_lateral_data ~/nnlc-data/ -o lateral_data.csv --temporal

# Parquet format (faster for large datasets)
python -m nnlc_tools.extract_lateral_data ~/nnlc-data/ -o lateral_data.parquet --format parquet
```

### 3. Score route quality

```bash
python -m nnlc_tools.score_routes ~/nnlc-data/

# Or score from extracted CSV
python -m nnlc_tools.score_routes lateral_data.csv

# Only show routes scoring 70+
python -m nnlc_tools.score_routes lateral_data.csv --min-score 70
```

### 4. Visualize data coverage

```bash
python -m nnlc_tools.visualize_coverage lateral_data.csv -o coverage.png
```

This generates a 3-panel plot:
- **Speed vs lateral accel heatmap** — shows data density, highlights gaps (<50 samples in red)
- **Lateral accel distribution** — shows balance of left/right turning data
- **Override rate by speed** — shows where the driver is fighting the controller

### 5. Assess coverage and iterate

Review the coverage chart from step 4. If you see red bins (gaps with <50 samples), collect more driving data targeting those conditions before training. Common gaps:
- Low-speed tight turns (city driving)
- High-speed gentle curves (highway)
- One turning direction over the other

### 6. Train model

See [training/README.md](training/README.md) for Julia setup and training instructions.

```bash
# Recommended — handles juliaup PATH automatically
bash training/run.sh /path/to/latmodels/

# Or run Julia directly
cd training/
julia latmodel_temporal.jl /path/to/latmodels/

# Force CPU mode (no GPU required, slower for large datasets)
bash training/run.sh /path/to/latmodels/ --cpu
```

### 7. Deploy model

Copy the output JSON to your openpilot install:

```bash
cp my_car_model.json /path/to/openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/
```

The filename should match your car's fingerprint. See `sunnypilot/selfdrive/controls/lib/nnlc/helpers.py` for the naming convention.

## Driving Tips for Data Collection

Good training data is diverse and clean. Aim for:

- **Disable NNLC while collecting**: Use the stock torque controller during data collection so the torque signal reflects the base controller, not a previous model
- **Disable "lateral on blinker"**: Turn off any blinker-based lateral override settings to avoid noisy data during lane changes
- **Varied speeds**: City streets (5-15 m/s), suburban (15-25 m/s), highway (25-35 m/s)
- **Varied turns**: Gentle curves, tight turns, S-curves, on-ramps/off-ramps
- **Minimal overrides**: Let the controller drive — interventions corrupt the torque signal
- **Both directions**: Left and right turns in equal measure
- **Different road grades**: Flat, uphill, downhill — affects roll compensation
- **Multiple routes**: Don't just drive the same loop repeatedly — aim for 20-30 clean routes across different road types
- **Dry roads**: Wet/icy roads change tire grip and produce non-representative data

**How much data?** Start with 5-10 hours of clean driving across 20-30 routes. Check coverage gaps with `visualize_coverage` and fill them with targeted drives.

**What to avoid:**
- Heavy traffic (lots of standstill/stop-and-go)
- Construction zones (lane changes, overrides)
- Parking lots (low speed, lots of turning at standstill)

## Tool Reference

### sync_rlogs

```
python -m nnlc_tools.sync_rlogs [-h] -d DEVICE -o OUTPUT [-u USER] [-p PATH] [--dry-run] [--no-rsync]

  -d, --device     Device IP address
  -o, --output     Local output directory
  -u, --user       SSH username (default: comma)
  -p, --path       Device rlog path (default: /data/media/0/realdata/)
  --dry-run        Show what would be synced
  --no-rsync       Force SFTP mode
```

### extract_lateral_data

```
python -m nnlc_tools.extract_lateral_data [-h] [-o OUTPUT] [--format {csv,parquet}] [--temporal] input

  input            Directory containing rlog files
  -o, --output     Output file path (default: lateral_data.csv)
  --format         Output format (default: inferred from extension)
  --temporal       Add temporal lag/lead columns for NNLC training
```

### score_routes

```
python -m nnlc_tools.score_routes [-h] [--min-score MIN_SCORE] input

  input            CSV/Parquet file or directory of rlogs
  --min-score      Only show routes with score >= this value
```

Scoring criteria (100 base, deductions):

| Criterion | Penalty |
|-----------|---------|
| Override rate > 10% | -30 |
| Saturated > 5% | -20 |
| Inactive > 20% | -25 |
| Standstill > 30% | -15 |
| Lane change > 10% | -10 |
| Low speed diversity | -10 |
| Low lat-accel diversity | -10 |

### visualize_coverage

```
python -m nnlc_tools.visualize_coverage [-h] [-o OUTPUT] [--gap-threshold GAP_THRESHOLD] input

  input              CSV/Parquet file or directory of rlogs
  -o, --output       Output image path (default: coverage.png)
  --gap-threshold    Highlight bins with fewer samples (default: 50)
```

## Troubleshooting

### Out of memory during extraction

Process fewer rlogs at a time, or use `--format parquet` which is more memory-efficient. The extractor processes segments one at a time, but the final DataFrame concatenation can be large.

### rsync connection refused

The comma device may not have rsync installed. Use `--no-rsync` to fall back to SFTP:
```bash
python -m nnlc_tools.sync_rlogs -d 192.168.1.161 -o ~/nnlc-data/ --no-rsync
```

### No rlog files found

Check that your rlogs are in the expected directory structure:
```
~/nnlc-data/
  2024-01-15--12-30-45/
    0/rlog.zst
    1/rlog.zst
    ...
```

### Julia training on CPU

CPU training works — expect ~8 seconds for 1000 epochs on small datasets. Use `--cpu` to force CPU mode:

```bash
bash training/run.sh /path/to/latmodels/ --cpu
```

GPU (CUDA or Metal) is still recommended for large datasets due to speed. See [training/README.md](training/README.md).

## Source Attribution

This project builds on work from:
- [mmmorks/sunnypilot](https://github.com/mmmorks/sunnypilot) (`staging-merged` @ `8a9f0311`) — Python rlog processing tools
- [ryanomatic/rlog_aggregation](https://github.com/ryanomatic/rlog_aggregation) (`main` @ `26b1ea05`) — Rlog download tool
- [mmmorks/OP_ML_FF](https://github.com/mmmorks/OP_ML_FF) (`master` @ `0116b9e3`, forked from [twilsonco/OP_ML_FF](https://github.com/twilsonco/OP_ML_FF)) — Julia training scripts
