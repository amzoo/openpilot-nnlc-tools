"""Shared data loading for NNLC tools."""

import os

import pandas as pd


def load_data(input_path):
    """Load lateral data from CSV, Parquet, or directory of rlogs.

    Returns a DataFrame, or None if no data found.
    """
    if os.path.isfile(input_path):
        if input_path.endswith(".parquet"):
            return pd.read_parquet(input_path)
        return pd.read_csv(input_path)

    if os.path.isdir(input_path):
        from nnlc_tools.extract_lateral_data import find_rlogs, extract_segment, COLUMNS

        rlog_files = find_rlogs(input_path)
        if not rlog_files:
            return None
        all_rows = []
        for path in rlog_files:
            all_rows.extend(extract_segment(path))
        if not all_rows:
            return None
        return pd.DataFrame(all_rows, columns=COLUMNS)

    return None
