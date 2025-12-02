# plot_corr_empirical.py
# ------------------------------------------------------------
# Heatmap of empirical Corr(X) from detrended log-returns (original units).
# Input:  ../detrended_returns_df_200.csv
# Output: One figure (imshow).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
X.index = pd.to_datetime(X.index)
X = X.select_dtypes(include=[np.number]).astype("float64")

R_emp = X.corr()

plt.figure(figsize=(7.5, 6))
im = plt.imshow(R_emp.values, aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
plt.title("Empirical correlation heatmap of X")
plt.xlabel("Assets")
plt.ylabel("Assets")
plt.tight_layout()
plt.show()