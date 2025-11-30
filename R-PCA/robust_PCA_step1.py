# robust_PCA.py
# ------------------------------------------------------------
# Purpose (educational):
#   Step 1 — Column-wise centering (median)
# ------------------------------------------------------------
# This script prepares a detrended returns matrix for RPCA in first clean steps.
# Step 1 subtracts a robust per-column center (median).

import pandas as pd
import numpy as np
from typing import Tuple

# -----------------------------
# 1) Data ingestion
# -----------------------------
# Expect a CSV with detrended daily log-returns (columns = assets, rows = dates).
detrended_returns_df = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
detrended_returns_df.index = pd.to_datetime(detrended_returns_df.index)
detrended_returns_df = detrended_returns_df.select_dtypes(include=[np.number]).astype("float64")


# -----------------------------
# Step 1: Column-wise centering
# -----------------------------
def center_columns_median(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Column-wise centering by subtracting the median of each column.

    Parameters
    ----------
    df : pd.DataFrame
        T x N returns matrix (rows = time, cols = assets).

    Returns
    -------
    centered_df : pd.DataFrame
        The centered matrix where each column has median ~ 0.
    shift_vector : pd.Series
        The per-column medians that were subtracted (for inverse-transform).

    Notes
    -----
    - Median is used as a robust location estimator (less sensitive to outliers).
    """
    # Per-column median (robust location)
    shift_vector = df.median(axis=0)

    # Subtract the median from each column
    centered_df = df - shift_vector

    return centered_df, shift_vector


if __name__ == "__main__":
    # --------------------------------------------------------
    # Step 1: Centering (median)
    # --------------------------------------------------------
    centered_df, shift_vec = center_columns_median(detrended_returns_df)
    centered_df.to_parquet("returns_centered.parquet")
    shift_vec.to_csv("center_shift_vector.csv")

    # Quick sanity check: column medians should be ~0
    print("Sanity — |column median| after centering (should be ~0):")
    print(centered_df.median().abs().describe())

    print("\n[INFO] Step 1 (centering). Files saved:")
    print(" - returns_centered.parquet")
    print(" - center_shift_vector.csv")
