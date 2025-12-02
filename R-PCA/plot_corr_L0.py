# plot_corr_L0.py
# ------------------------------------------------------------
# Heatmap of Corr(L_original) i.e., factor correlation structure.
# Input:  L_original.parquet
# Output: One figure (imshow).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

L0 = pd.read_parquet("L_original.parquet").astype("float64")
R_L = L0.corr()

plt.figure(figsize=(7.5, 6))
im = plt.imshow(R_L.values, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
plt.title("Correlation heatmap of L_original")
plt.xlabel("Assets")
plt.ylabel("Assets")
plt.tight_layout()
plt.show()