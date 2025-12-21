# plot_singular_values_L0.py
# -------------------------------------------
# Purpose:
#   Read L_original.parquet (low-rank RPCA component in original units),
#   compute its singular values, and save a log-scale plot:
#   -> L0_singular_values_log.png
#
# Usage:
#   Place this script in the same folder where L_original.parquet lives (e.g., R-PCA/),
#   then run:  python plot_singular_values_L0.py
#
# Notes:
#   - The plot is purely illustrative: it shows low effective rank if the curve decays fast.
#   - We do not set custom colors/styles (keep it minimal and reproducible).

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Load L0 from Parquet
# -----------------------------
# Expect file in current directory: ./L_original.parquet
here = Path(__file__).resolve().parent
L0_path = here / "L_original.parquet"

# Read as DataFrame; columns=assets, rows=dates (T x N)
L0_df: pd.DataFrame = pd.read_parquet(L0_path)

# Ensure numeric dtype (simple and explicit for teaching)
L0_df = L0_df.astype("float64")

# -----------------------------
# 2) Compute singular values
# -----------------------------
# Convert to dense ndarray (T x N)
L0_mat: np.ndarray = L0_df.to_numpy()

# Full SVD is fine in teaching context (readability > ultimate speed)
# We only need singular values, so compute_uv=False
sing_vals: np.ndarray = np.linalg.svd(L0_mat, full_matrices=False, compute_uv=False)

# Sort descending for a clean look
sing_vals_sorted: np.ndarray = np.sort(sing_vals)[::-1]

# -----------------------------
# 3) Plot on log-scale
# -----------------------------
plt.figure(figsize=(6, 4))
plt.semilogy(sing_vals_sorted, linewidth=2)  # log-scale on y-axis
plt.title("Singular values of $L_0$ (log-scale)")
plt.xlabel("Index (sorted)")
plt.ylabel("Singular value (log)")
plt.tight_layout()

out_path = here / "L0_singular_values_log.png"
plt.savefig(out_path, dpi=200)
plt.close()

print(f"[OK] Saved plot: {out_path}")
