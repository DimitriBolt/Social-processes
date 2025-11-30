# robust_PCA.py
# ------------------------------------------------------------
# Purpose:
#   Step 2 — Column-wise scaling (robust, via MAD*1.4826)
# ------------------------------------------------------------
# This script prepares a detrended returns matrix for RPCA in 2nd clean steps.
# Step 2 divides by a robust per-column scale estimate (MAD * 1.4826), making RPCA penalties comparable across assets.

import pandas as pd
import numpy as np
from typing import Tuple

# -----------------------------
# 1) Data ingestion
# -----------------------------
#  For 2nd step expect a returns centered parquet file from step 1.
centered_df = pd.read_parquet("returns_centered.parquet")


# -----------------------------
# Step 2: Column-wise scaling (robust)
# -----------------------------
def scale_columns_mad(centered_df: pd.DataFrame, mad_consistency: float = 1.4826) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Column-wise robust scaling using MAD (Median Absolute Deviation).

    Parameters
    ----------
    centered_df : pd.DataFrame
        Centered returns matrix (ideally, each column already has median ~ 0).
    mad_consistency : float
        Consistency constant to map MAD to an std-like scale under Normality.
        (MAD * 1.4826 ≈ std if the data are approximately Normal.)

    Returns
    -------
    scaled_df : pd.DataFrame
        Centered-and-scaled matrix (comparable scale across columns).
    scale_vector : pd.Series
        The per-column robust scale used to divide each column.

    Notes
    -----
    - Because columns are centered, MAD can be computed as median(|x|).
    - We multiply MAD by 1.4826 so that the resulting scale is comparable to std.
    - If you prefer classical scaling, replace MAD with per-column std.
    """
    # Median absolute deviation per column (about the column's median ~ 0)
    # Since centered_df is centered, abs deviations are simply |x|.
    mad = centered_df.abs().median(axis=0)

    # Robust scale comparable to std under Normal assumptions
    scale_vector = mad_consistency * mad

    # Divide each column by its robust scale
    scaled_df = centered_df / scale_vector

    return scaled_df, scale_vector


if __name__ == "__main__":
    # --------------------------------------------------------
    # Step 2: Scaling (robust, MAD*1.4826)
    # --------------------------------------------------------
    scaled_df, scale_vec = scale_columns_mad(centered_df, mad_consistency=1.4826)
    scaled_df.to_parquet("returns_centered_scaled.parquet")
    scale_vec.to_csv("scale_vector.csv")

    # Quick sanity check: robust spread per column should be around 1
    # (MAD*1.4826 of the scaled data should be ~1)
    mad_check = (scaled_df.abs().median(axis=0)) * 1.4826
    print("\nSanity — robust spread (MAD*1.4826) after scaling (~1 expected):")
    print(mad_check.describe())

    print("\n[INFO] Step 2 (scaling) completed. Files saved:")
    print(" - returns_centered_scaled.parquet")
    print(" - scale_vector.csv")
