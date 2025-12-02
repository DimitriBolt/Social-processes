# plot_specific_variance_bar.py
# ------------------------------------------------------------
# Bar chart of specific variances diag(Var(S_original)).
# Input:  S_original.parquet
# Output: One figure (bar chart). Shows top-N largest specific variances.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOP_N = 30  # show top-30 assets by specific variance for readability

S0 = pd.read_parquet("S_original.parquet").astype("float64")
spec_var = S0.var(axis=0)  # per-column variance
spec_var_sorted = spec_var.sort_values(ascending=False).head(TOP_N)

plt.figure(figsize=(10, 5))
plt.bar(range(len(spec_var_sorted)), spec_var_sorted.values)
plt.xticks(range(len(spec_var_sorted)), spec_var_sorted.index, rotation=90)
plt.ylabel("Specific variance (daily units)")
plt.title(f"Top-{TOP_N} specific variances from S_original")
plt.tight_layout()
plt.show()