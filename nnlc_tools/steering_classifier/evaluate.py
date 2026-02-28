"""Evaluation metrics for the steering event classifier."""

import numpy as np
import pandas as pd


def classification_report(
    y_true: list,   # "driver" | "mechanical"
    y_pred: list,
    confidences: list = None,
) -> dict:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix.

    Returns a dict with keys: accuracy, precision, recall, f1, confusion_matrix.
    """
    labels = ["driver", "mechanical"]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    n = len(y_true)
    accuracy = float(np.mean(y_true == y_pred))

    results = {"accuracy": accuracy, "n": n, "classes": {}}

    for label in labels:
        tp = int(np.sum((y_pred == label) & (y_true == label)))
        fp = int(np.sum((y_pred == label) & (y_true != label)))
        fn = int(np.sum((y_pred != label) & (y_true == label)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results["classes"][label] = {"precision": prec, "recall": rec, "f1": f1,
                                     "tp": tp, "fp": fp, "fn": fn}

    # Confusion matrix: rows = true, cols = pred
    results["confusion_matrix"] = {
        t: {p: int(np.sum((y_true == t) & (y_pred == p))) for p in labels}
        for t in labels
    }

    return results


def per_speed_band_accuracy(
    y_true: list,
    y_pred: list,
    v_ego_mean: list,
    bands: list = None,
) -> dict:
    """Compute accuracy broken down by mean speed during event.

    bands: list of (low_mps, high_mps) tuples. Default: 0-10, 10-20, 20-30, 30+.
    """
    if bands is None:
        bands = [(0, 10), (10, 20), (20, 30), (30, 999)]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    v = np.array(v_ego_mean)

    out = {}
    for lo, hi in bands:
        mask = (v >= lo) & (v < hi)
        if np.sum(mask) == 0:
            continue
        acc = float(np.mean(y_true[mask] == y_pred[mask]))
        label = f"{lo}-{hi} m/s" if hi < 999 else f"{lo}+ m/s"
        out[label] = {"accuracy": acc, "n": int(np.sum(mask))}

    return out


def latency_histogram(stage_list: list) -> dict:
    """Summarise which cascade stage made each decision.

    stage_list: list of ints (1, 2, or 3).

    Returns counts and fraction for each stage, plus a rough latency estimate
    (Stage 1 ≈ 10 ms, Stage 2 ≈ 50 ms, Stage 3 ≈ 150 ms).
    """
    stage_latency_ms = {1: 10, 2: 50, 3: 150}
    stage_arr = np.array(stage_list)
    n = len(stage_arr)
    out = {}
    for s in [1, 2, 3]:
        count = int(np.sum(stage_arr == s))
        out[f"stage_{s}"] = {
            "count": count,
            "fraction": count / n if n > 0 else 0.0,
            "latency_ms": stage_latency_ms[s],
        }
    return out


def print_report(results: dict, speed_results: dict = None, latency: dict = None):
    """Pretty-print evaluation results to stdout."""
    print(f"\n{'='*55}")
    print(f"  Accuracy: {results['accuracy']:.1%}  (n={results['n']})")
    print(f"{'='*55}")
    print(f"  {'Class':<12} {'Prec':>6} {'Rec':>6} {'F1':>6}  (TP/FP/FN)")
    print(f"  {'-'*50}")
    for label, m in results["classes"].items():
        print(f"  {label:<12} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}"
              f"  ({m['tp']}/{m['fp']}/{m['fn']})")

    cm = results["confusion_matrix"]
    labels = list(cm.keys())
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    header = "  " + " " * 12 + "  ".join(f"{l:>10}" for l in labels)
    print(header)
    for t in labels:
        row = "  " + f"{t:<12}" + "  ".join(f"{cm[t][p]:>10}" for p in labels)
        print(row)

    if speed_results:
        print(f"\n  Per-speed accuracy:")
        for band, m in speed_results.items():
            print(f"    {band:<15} {m['accuracy']:.1%}  (n={m['n']})")

    if latency:
        print(f"\n  Cascade stage distribution:")
        for stage, m in latency.items():
            print(f"    {stage}  {m['fraction']:.1%} ({m['count']})  ~{m['latency_ms']} ms")

    print()
