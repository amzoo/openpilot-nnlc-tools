#!/bin/bash
set -e

export PATH="$HOME/.juliaup/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"

DATA_DIR="${1:-./data/}"

echo "Running latmodel_temporal.jl with data from: $DATA_DIR"
julia training/latmodel_temporal.jl "$DATA_DIR"
