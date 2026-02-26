#!/usr/bin/env python3
"""One-time script to create test fixture from real driving data.

Samples 1000 rows from data/lateral_data.csv (filtering to active, non-standstill)
and writes to tests/fixtures/test_data.csv for use in integration tests.
"""

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(REPO_ROOT, "data", "lateral_data.csv")
OUTPUT = os.path.join(REPO_ROOT, "tests", "fixtures", "test_data.csv")


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows from {INPUT}")

    # Filter same as training script
    df = df[(df["active"] == True) & (df["standstill"] == False)]
    print(f"After filtering (active=true, standstill=false): {len(df)} rows")

    sample = df.sample(n=1000, random_state=42)
    print(f"Sampled 1000 rows")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    sample.to_csv(OUTPUT, index=False)
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
