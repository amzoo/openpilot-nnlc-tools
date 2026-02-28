#!/usr/bin/env python3
"""Classify steering override events as genuine driver interventions or mechanical disturbances.

steering_pressed=True fires on both intentional driver corrections and mechanical disturbances
(potholes, bumps, curb impacts). This tool segments override events and classifies each one
using three heuristic rules: brief duration, longitudinal shock, and torque oscillation.

Usage:
  python -m nnlc_tools.analyze_interventions output/lateral_data.csv
  python -m nnlc_tools.analyze_interventions output/lateral_data.csv --plot -o output/interventions.png
"""

import argparse
import sys

import numpy as np
import pandas as pd


# Default thresholds
DEFAULT_MIN_DURATION = 0.15    # seconds — rule_brief threshold
DEFAULT_A_EGO_THRESH = 1.5     # m/s² — longitudinal shock threshold
DEFAULT_CONSISTENCY_THRESH = 0.65  # fraction — torque sign consistency threshold
DEFAULT_MIN_SCORE = 2          # rules that must fire to classify as mechanical
DEFAULT_GAP_FRAMES = 5         # frames of allowed gap within one event


def load_data(input_path):
    """Load data from CSV, Parquet, or directory of rlogs."""
    from nnlc_tools.data_io import load_data as _load_data
    df = _load_data(input_path)
    if df is None:
        print(f"ERROR: No data found at {input_path}")
        sys.exit(1)
    return df


def segment_events(df, gap_frames=DEFAULT_GAP_FRAMES):
    """Group consecutive steering_pressed=True rows into events.

    Events separated by <= gap_frames of non-pressed rows are merged.

    Returns a list of dicts with event metadata.
    """
    if "steering_pressed" not in df.columns:
        return []

    sp = df["steering_pressed"].astype(bool).values
    n = len(sp)

    events = []
    i = 0
    while i < n:
        if not sp[i]:
            i += 1
            continue

        # Start of a new event
        start = i
        end = i

        # Extend, merging over short gaps
        j = i + 1
        while j < n:
            if sp[j]:
                end = j
                j += 1
            else:
                # Look ahead to see if there's another pressed region within gap_frames
                gap_end = j
                while gap_end < n and not sp[gap_end] and (gap_end - j) <= gap_frames:
                    gap_end += 1
                if gap_end < n and sp[gap_end]:
                    # Merge: skip the gap
                    j = gap_end
                else:
                    break

        events.append({"start_idx": start, "end_idx": end})
        i = end + 1

    return events


def compute_event_features(df, events):
    """Compute per-event features from the rows each event spans.

    Returns a DataFrame with one row per event.
    """
    rows = []

    for evt_id, evt in enumerate(events):
        s, e = evt["start_idx"], evt["end_idx"]
        chunk = df.iloc[s : e + 1]
        n_frames = len(chunk)
        duration_s = n_frames * 0.01

        # Torque features
        torque_mean_abs = np.nan
        torque_std = np.nan
        torque_sign_consistency = np.nan
        if "steering_torque" in df.columns:
            torque = chunk["steering_torque"].dropna()
            if len(torque) > 0:
                torque_mean_abs = torque.abs().mean()
                torque_std = torque.std() if len(torque) > 1 else 0.0
                modal_sign = 1 if torque.mean() >= 0 else -1
                torque_sign_consistency = (np.sign(torque) == modal_sign).mean()

        # Steering rate
        steering_rate_max_abs = np.nan
        if "steering_rate_deg" in df.columns:
            sr = chunk["steering_rate_deg"].dropna()
            if len(sr) > 0:
                steering_rate_max_abs = sr.abs().max()

        # Longitudinal acceleration
        a_ego_max_abs = np.nan
        if "a_ego" in df.columns:
            a = chunk["a_ego"].dropna()
            if len(a) > 0:
                a_ego_max_abs = a.abs().max()

        # Lateral acceleration std
        lat_accel_std = np.nan
        for col in ["actual_lateral_accel", "desired_lateral_accel"]:
            if col in df.columns:
                la = chunk[col].dropna()
                if len(la) > 1:
                    lat_accel_std = la.std()
                break

        # Speed context
        speed_mean = np.nan
        if "v_ego" in df.columns:
            v = chunk["v_ego"].dropna()
            if len(v) > 0:
                speed_mean = v.mean()

        # Start/end time (use index as proxy if no time column)
        start_time = s * 0.01
        end_time = e * 0.01

        rows.append({
            "event_id": evt_id,
            "start_idx": s,
            "end_idx": e,
            "start_time": start_time,
            "end_time": end_time,
            "duration_s": duration_s,
            "n_frames": n_frames,
            "torque_mean_abs": torque_mean_abs,
            "torque_std": torque_std,
            "torque_sign_consistency": torque_sign_consistency,
            "steering_rate_max_abs": steering_rate_max_abs,
            "a_ego_max_abs": a_ego_max_abs,
            "lat_accel_std": lat_accel_std,
            "speed_mean": speed_mean,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def classify_events(
    events_df,
    min_duration=DEFAULT_MIN_DURATION,
    a_ego_thresh=DEFAULT_A_EGO_THRESH,
    consistency_thresh=DEFAULT_CONSISTENCY_THRESH,
    min_score=DEFAULT_MIN_SCORE,
):
    """Apply heuristic rules and classify each event.

    Adds columns: rule_brief, rule_shock, rule_chaotic, mechanical_score, classification, confidence.
    """
    df = events_df.copy()

    # Rule 1: Brief duration
    df["rule_brief"] = (df["duration_s"] < min_duration).astype(int)

    # Rule 2: Longitudinal shock + brief
    df["rule_shock"] = (
        (df["a_ego_max_abs"] > a_ego_thresh) & (df["duration_s"] < 0.4)
    ).fillna(False).astype(int)

    # Rule 3: Chaotic torque (low sign consistency)
    df["rule_chaotic"] = (
        df["torque_sign_consistency"] < consistency_thresh
    ).fillna(False).astype(int)

    df["mechanical_score"] = df["rule_brief"] + df["rule_shock"] + df["rule_chaotic"]
    df["classification"] = df["mechanical_score"].apply(
        lambda s: "mechanical" if s >= min_score else "genuine"
    )
    # Confidence: fraction of rules fired (for mechanical), inverted for genuine
    df["confidence"] = df.apply(
        lambda r: r["mechanical_score"] / 3 if r["classification"] == "mechanical"
        else 1 - r["mechanical_score"] / 3,
        axis=1,
    )

    return df


def mark_rows(active_df, events_df):
    """Add '_intervention' column to active_df: 'normal', 'genuine', or 'mechanical'.

    Rows that belong to an override event are labelled with the event's classification;
    all other rows are labelled 'normal'.
    """
    labels = pd.array(["normal"] * len(active_df), dtype=object)
    for _, evt in events_df.iterrows():
        labels[int(evt["start_idx"]) : int(evt["end_idx"]) + 1] = evt["classification"]
    result = active_df.copy()
    result["_intervention"] = labels
    return result


def plot_intervention_scatter(df, output_path, max_points=None):
    """Generate lat_accel vs torque scatter by speed bin with intervention points highlighted.

    Three layers per panel:
      - gray (faint)   — normal driving frames
      - steelblue      — frames belonging to genuine driver interventions
      - tomato (large) — frames belonging to mechanical disturbances
    """
    import math

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    MS_TO_MPH = 2.23694

    lat_accel_col = None
    for col in ["actual_lateral_accel", "desired_lateral_accel"]:
        if col in df.columns and df[col].notna().sum() > 0:
            lat_accel_col = col
            break

    if lat_accel_col is None:
        print("WARNING: No lateral acceleration data found for scatter plot.")
        return

    if "torque_output" not in df.columns:
        print("WARNING: No torque_output column. Skipping intervention scatter.")
        return

    valid = df[[lat_accel_col, "torque_output", "v_ego", "_intervention"]].dropna(
        subset=[lat_accel_col, "torque_output", "v_ego"]
    ).copy()
    valid["speed_mph"] = valid["v_ego"] * MS_TO_MPH

    speed_bins = list(range(0, 90, 10))
    n_bins = len(speed_bins)
    ncols = 3
    nrows = math.ceil(n_bins / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle("Lateral Accel vs Torque — Intervention Overlay", fontsize=14, fontweight="bold")

    for i, speed_lo in enumerate(speed_bins):
        speed_hi = speed_lo + 10
        ax = axes[i]

        bin_data = valid[(valid["speed_mph"] >= speed_lo) & (valid["speed_mph"] < speed_hi)]

        normal = bin_data[bin_data["_intervention"] == "normal"]
        genuine = bin_data[bin_data["_intervention"] == "genuine"]
        mechanical = bin_data[bin_data["_intervention"] == "mechanical"]

        # Downsample normal background if requested
        plot_normal = (
            normal.sample(n=max_points, random_state=42)
            if max_points and len(normal) > max_points
            else normal
        )

        if len(plot_normal) > 0:
            ax.scatter(plot_normal[lat_accel_col], plot_normal["torque_output"],
                       c="gray", s=0.5, alpha=0.15, rasterized=True)

        if len(genuine) > 0:
            ax.scatter(genuine[lat_accel_col], genuine["torque_output"],
                       c="steelblue", s=4, alpha=0.6, rasterized=True)

        if len(mechanical) > 0:
            ax.scatter(mechanical[lat_accel_col], mechanical["torque_output"],
                       c="tomato", s=10, alpha=0.9, rasterized=True, zorder=5)

        ax.set_title(
            f"{speed_lo}-{speed_hi} mph  (n={len(bin_data):,}"
            + (f", gen={len(genuine):,}" if len(genuine) > 0 else "")
            + (f", mech={len(mechanical):,}" if len(mechanical) > 0 else "")
            + ")",
            fontsize=9,
        )
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Lat Accel (m/s²)")
        ax.set_ylabel("Torque")
        ax.grid(axis="x", color="0.95")
        ax.grid(axis="y", color="0.95")

    # Legend on first panel
    handles = [
        Patch(facecolor="gray", alpha=0.5, label="Normal driving"),
        Patch(facecolor="steelblue", alpha=0.8, label="Genuine intervention"),
        Patch(facecolor="tomato", alpha=0.9, label="Mechanical disturbance"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8)

    for j in range(n_bins, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved intervention scatter to {output_path}")
    plt.close()


def print_summary(events_df):
    """Print console summary of classification results."""
    total = len(events_df)
    if total == 0:
        print("No override events found.")
        return

    genuine = events_df[events_df["classification"] == "genuine"]
    mechanical = events_df[events_df["classification"] == "mechanical"]
    n_genuine = len(genuine)
    n_mechanical = len(mechanical)

    print(f"\nTotal override events: {total:,}")
    print(f"  Genuine interventions:   {n_genuine:,}  ({n_genuine / total:.1%})")
    print(f"  Mechanical disturbances: {n_mechanical:,} ({n_mechanical / total:.1%})")

    if n_genuine > 0:
        g_dur = genuine["duration_s"].median()
        g_torque = genuine["torque_mean_abs"].dropna().median()
        print(f"\nGenuine:   median duration {g_dur:.2f}s", end="")
        if not np.isnan(g_torque):
            print(f", median |torque| {g_torque:.1f}", end="")
        print()

    if n_mechanical > 0:
        m_dur = mechanical["duration_s"].median()
        m_torque = mechanical["torque_mean_abs"].dropna().median()
        print(f"Mechanical: median duration {m_dur:.2f}s", end="")
        if not np.isnan(m_torque):
            print(f", median |torque| {m_torque:.1f}", end="")
        print()

    print()


def plot_interventions(events_df, output_path, min_duration, a_ego_thresh, consistency_thresh):
    """Generate 1×3 subplot showing each rule's discriminating power."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Driver Intervention Classification — Per-Rule Analysis", fontsize=13, fontweight="bold")

    # Panel 1 — Rule: Brief Duration
    ax1 = axes[0]
    fired = events_df[events_df["rule_brief"] == 1]["duration_s"]
    not_fired = events_df[events_df["rule_brief"] == 0]["duration_s"]

    bins = np.linspace(0, min(events_df["duration_s"].quantile(0.99), 5.0), 50)
    ax1.hist(not_fired.clip(upper=bins[-1]), bins=bins, color="steelblue", alpha=0.7, label="not fired")
    ax1.hist(fired.clip(upper=bins[-1]), bins=bins, color="tomato", alpha=0.7, label="fired")
    ax1.axvline(min_duration, color="black", linestyle="--", linewidth=1.5, label=f"threshold={min_duration}s")
    ax1.set_xlabel("Duration (s)")
    ax1.set_ylabel("Event count")
    ax1.set_title("Rule: Brief Duration")
    ax1.legend(fontsize=8)

    # Panel 2 — Rule: Longitudinal Shock
    ax2 = axes[1]
    has_a_ego = events_df["a_ego_max_abs"].notna()
    plot_df = events_df[has_a_ego]

    if len(plot_df) > 0:
        colors = ["tomato" if r == 1 else "steelblue" for r in plot_df["rule_shock"]]
        ax2.scatter(
            plot_df["duration_s"].clip(upper=2.0),
            plot_df["a_ego_max_abs"].clip(upper=5.0),
            c=colors, s=8, alpha=0.5,
        )
        ax2.axhline(a_ego_thresh, color="black", linestyle="--", linewidth=1.5, label=f"|a_ego|={a_ego_thresh}")
        ax2.axvline(0.4, color="gray", linestyle="--", linewidth=1.0, label="dur=0.4s")
        ax2.set_xlabel("Duration (s)")
        ax2.set_ylabel("|a_ego| max (m/s²)")
        ax2.legend(fontsize=8)
        # Add proxy handles for color legend
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor="tomato", alpha=0.7, label="fired"),
            Patch(facecolor="steelblue", alpha=0.7, label="not fired"),
        ]
        ax2.legend(handles=handles + ax2.get_lines()[:2], fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No a_ego data available",
                 transform=ax2.transAxes, ha="center", va="center")

    ax2.set_title("Rule: Longitudinal Shock (a_ego)")

    # Panel 3 — Rule: Torque Oscillation
    ax3 = axes[2]
    has_consistency = events_df["torque_sign_consistency"].notna()
    plot_df3 = events_df[has_consistency]

    if len(plot_df3) > 0:
        fired3 = plot_df3[plot_df3["rule_chaotic"] == 1]["torque_sign_consistency"]
        not_fired3 = plot_df3[plot_df3["rule_chaotic"] == 0]["torque_sign_consistency"]
        bins3 = np.linspace(0, 1, 41)
        ax3.hist(not_fired3, bins=bins3, color="steelblue", alpha=0.7, label="not fired")
        ax3.hist(fired3, bins=bins3, color="tomato", alpha=0.7, label="fired")
        ax3.axvline(consistency_thresh, color="black", linestyle="--", linewidth=1.5,
                    label=f"threshold={consistency_thresh}")
        ax3.set_xlabel("Torque sign consistency")
        ax3.set_ylabel("Event count")
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No torque data available",
                 transform=ax3.transAxes, ha="center", va="center")

    ax3.set_title("Rule: Torque Oscillation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved interventions plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Classify steering override events as genuine driver interventions or mechanical disturbances.",
    )
    parser.add_argument("input", help="CSV/Parquet file or directory of rlogs")
    parser.add_argument("-o", "--output", default="interventions.png",
                        help="Output plot PNG (default: interventions.png)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate per-rule visualization PNG")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                        help=f"Threshold for rule_brief in seconds (default: {DEFAULT_MIN_DURATION})")
    parser.add_argument("--a-ego-thresh", type=float, default=DEFAULT_A_EGO_THRESH,
                        help=f"|a_ego| threshold for rule_shock in m/s² (default: {DEFAULT_A_EGO_THRESH})")
    parser.add_argument("--consistency-thresh", type=float, default=DEFAULT_CONSISTENCY_THRESH,
                        help=f"Torque sign consistency threshold for rule_chaotic (default: {DEFAULT_CONSISTENCY_THRESH})")
    parser.add_argument("--min-score", type=int, default=DEFAULT_MIN_SCORE,
                        help=f"Mechanical score >= this → mechanical classification (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument("--gap-frames", type=int, default=DEFAULT_GAP_FRAMES,
                        help=f"Frames of gap allowed within one event (default: {DEFAULT_GAP_FRAMES})")
    parser.add_argument("--scatter", action="store_true",
                        help="Generate lat_accel vs torque scatter plot with intervention points highlighted")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max normal-driving points per scatter subplot (random sample, default: all)")
    args = parser.parse_args()

    df = load_data(args.input)
    print(f"Loaded {len(df):,} rows")

    if "steering_pressed" not in df.columns:
        print("ERROR: No steering_pressed column found. Cannot detect override events.")
        sys.exit(1)

    # Filter to active driving (same pattern as visualize_coverage.py:52-57)
    mask = pd.Series(True, index=df.index)
    if "active" in df.columns:
        mask &= df["active"].astype(bool)
    if "standstill" in df.columns:
        mask &= ~df["standstill"].astype(bool)
    active_df = df[mask].reset_index(drop=True)

    n_override_frames = active_df["steering_pressed"].astype(bool).sum()
    print(f"Active frames: {len(active_df):,}  |  Override frames: {n_override_frames:,} ({n_override_frames / max(len(active_df), 1):.1%})")

    events = segment_events(active_df, gap_frames=args.gap_frames)
    print(f"Segmented into {len(events):,} events (gap tolerance: {args.gap_frames} frames)")

    if not events:
        print("No override events found.")
        return

    events_df = compute_event_features(active_df, events)
    events_df = classify_events(
        events_df,
        min_duration=args.min_duration,
        a_ego_thresh=args.a_ego_thresh,
        consistency_thresh=args.consistency_thresh,
        min_score=args.min_score,
    )

    print_summary(events_df)

    if args.plot:
        plot_interventions(
            events_df,
            args.output,
            min_duration=args.min_duration,
            a_ego_thresh=args.a_ego_thresh,
            consistency_thresh=args.consistency_thresh,
        )

    if args.scatter:
        import os
        base, ext = os.path.splitext(args.output)
        scatter_path = f"{base}_scatter{ext or '.png'}"
        labelled_df = mark_rows(active_df, events_df)
        plot_intervention_scatter(labelled_df, scatter_path, max_points=args.max_points)


if __name__ == "__main__":
    main()
