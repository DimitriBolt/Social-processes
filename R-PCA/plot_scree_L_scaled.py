# plot_scree_L_scaled.py
# ------------------------------------------------------------
# Scree plot of singular values of L (in centered+scaled units).
# Input:  L_scaled.parquet  (from Step 3)
# Output: A single figure with singular values vs. index (log-y).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

L = pd.read_parquet("L_scaled.parquet").astype("float64").values
# Full SVD (educational clarity)
U, s, Vt = np.linalg.svd(L, full_matrices=False)

plt.figure(figsize=(8, 5))
plt.semilogy(np.arange(1, len(s)+1), s, marker="o", linestyle="-")
plt.xlabel("Singular value index")
plt.ylabel("Singular value (log scale)")
plt.title("Scree plot: singular values of L (scaled)")
plt.tight_layout()
plt.show()