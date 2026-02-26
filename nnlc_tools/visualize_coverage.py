#!/usr/bin/env python3
"""Visualize lateral data coverage for NNLC training.

Generates a speed vs lateral acceleration heatmap with gap highlighting,
a lateral acceleration histogram, and override rate by speed.

Usage:
  python -m nnlc_tools.visualize_coverage output.csv -o coverage.png
  python -m nnlc_tools.visualize_coverage output.parquet -o coverage.png
  python -m nnlc_tools.visualize_coverage /path/to/rlogs/ -o coverage.png
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def load_data_for_viz(input_path):
    """Load data from CSV, Parquet, or directory of rlogs."""
    from nnlc_tools.data_io import load_data
    df = load_data(input_path)
    if df is None:
        print(f"ERROR: No data found at {input_path}")
        sys.exit(1)
    return df


def plot_coverage(df, output_path, gap_threshold=50):
    """Generate coverage visualization with 3 subplots."""
    # Determine lateral accel column
    lat_accel_col = None
    for col in ["actual_lateral_accel", "desired_lateral_accel"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lat_accel_col = col
            break

    if lat_accel_col is None:
        print("WARNING: No lateral acceleration data found. Using desired_curvature * v_ego^2.")
        df["_lat_accel"] = df["desired_curvature"] * df["v_ego"] ** 2
        lat_accel_col = "_lat_accel"

    # Filter to active driving only
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    active_df = df[mask].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("NNLC Training Data Coverage", fontsize=14, fontweight="bold")

    # 1. Speed vs Lateral Accel Heatmap
    ax1 = axes[0]
    speed_bins = np.linspace(0, 40, 41)
    lat_bins = np.linspace(-3, 3, 61)

    valid = active_df[["v_ego", lat_accel_col]].dropna()
    h, xedges, yedges = np.histogram2d(
        valid["v_ego"].clip(0, 40),
        valid[lat_accel_col].clip(-3, 3),
        bins=[speed_bins, lat_bins],
    )

    # Highlight gaps
    h_display = h.copy()
    h_display[h_display == 0] = np.nan

    im = ax1.pcolormesh(
        xedges, yedges, h_display.T,
        norm=LogNorm(vmin=1, vmax=max(h.max(), 1)),
        cmap="viridis",
    )

    # Mark gaps in red
    gap_mask = (h > 0) & (h < gap_threshold)
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            if gap_mask[i, j]:
                ax1.add_patch(plt.Rectangle(
                    (xedges[i], yedges[j]),
                    xedges[i + 1] - xedges[i],
                    yedges[j + 1] - yedges[j],
                    linewidth=0.5, edgecolor="red", facecolor="none",
                ))

    fig.colorbar(im, ax=ax1, label="Sample count (log)")
    ax1.set_xlabel("Speed (m/s)")
    ax1.set_ylabel(f"Lateral Accel (m/s²)")
    ax1.set_title("Speed vs Lat Accel\n(red outline = <50 samples)")

    # 2. Lateral Accel Histogram
    ax2 = axes[1]
    lat_valid = active_df[lat_accel_col].dropna()
    ax2.hist(lat_valid.clip(-3, 3), bins=60, color="steelblue", edgecolor="none", alpha=0.8)
    ax2.set_xlabel("Lateral Accel (m/s²)")
    ax2.set_ylabel("Count")
    ax2.set_title("Lateral Accel Distribution")
    ax2.axvline(0, color="gray", linestyle="--", alpha=0.5)

    # Add stats
    stats_text = (
        f"Mean: {lat_valid.mean():.3f}\n"
        f"Std:  {lat_valid.std():.3f}\n"
        f"|>1|: {(lat_valid.abs() > 1).mean():.1%}\n"
        f"|>2|: {(lat_valid.abs() > 2).mean():.1%}"
    )
    ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
             verticalalignment="top", horizontalalignment="right",
             fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # 3. Override Rate by Speed
    ax3 = axes[2]
    if "steering_pressed" in df.columns:
        speed_bins_override = np.arange(0, 42, 2)
        df_with_bin = active_df.copy()
        df_with_bin["speed_bin"] = pd.cut(df_with_bin["v_ego"], bins=speed_bins_override)
        override_by_speed = df_with_bin.groupby("speed_bin", observed=True)["steering_pressed"].mean() * 100

        centers = [(b.left + b.right) / 2 for b in override_by_speed.index]
        ax3.bar(centers, override_by_speed.values, width=1.5, color="coral", edgecolor="none", alpha=0.8)
        ax3.set_xlabel("Speed (m/s)")
        ax3.set_ylabel("Override Rate (%)")
        ax3.set_title("Steering Override by Speed")
        ax3.axhline(10, color="red", linestyle="--", alpha=0.5, label="10% threshold")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No steering_pressed\ndata available",
                 transform=ax3.transAxes, ha="center", va="center")
        ax3.set_title("Steering Override by Speed")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved coverage plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize lateral data coverage for NNLC training.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("-o", "--output", default="coverage.png",
                        help="Output image path (default: coverage.png)")
    parser.add_argument("--gap-threshold", type=int, default=50,
                        help="Highlight bins with fewer than this many samples (default: 50)")
    args = parser.parse_args()

    df = load_data_for_viz(args.input)
    print(f"Loaded {len(df)} rows")

    plot_coverage(df, args.output, args.gap_threshold)


if __name__ == "__main__":
    main()
