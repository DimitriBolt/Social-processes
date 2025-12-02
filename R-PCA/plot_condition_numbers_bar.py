# plot_condition_numbers_bar.py
# ------------------------------------------------------------
# Bar chart of condition numbers: empirical Sigma(X), Sigma_L, Sigma_RPCA.
# Inputs:
#   - ../detrended_returns_df_200.csv (to compute empirical Sigma)
#   - Sigma_L.parquet
#   - Sigma_RPCA.parquet
# Output: One figure (bar chart).
# Note: annualization factor cancels in condition number (scale-invariant),
#       but we compute empirical Sigma (annualized) to be consistent.
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Empirical covariance of X (annualized)
X = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
X.index = pd.to_datetime(X.index)
X = X.select_dtypes(include=[np.number]).astype("float64")
Sigma_emp = X.cov().values * 252.0

# RPCA-based covariances (already annualized in these files)
Sigma_L   = pd.read_parquet("Sigma_L.parquet").astype("float64").values
Sigma_RP  = pd.read_parquet("Sigma_RPCA.parquet").astype("float64").values

labels = ["Empirical Σ", "$Σ_L$ (factor-only)", "$Σ_RPCA$ (factor+diag)"]
conds  = [
    float(np.linalg.cond(Sigma_emp)),
    float(np.linalg.cond(Sigma_L)),
    float(np.linalg.cond(Sigma_RP)),
]

plt.figure(figsize=(7, 4.5))
plt.bar(range(3), conds)
plt.xticks(range(3), labels, rotation=10)
plt.ylabel("Condition number")
plt.title("Condition numbers: empirical vs $Σ_L$ vs $Σ_{RPCA}$ (annualized)")
# Optional: log scale if magnitudes are extreme
plt.yscale("log")
plt.tight_layout()
plt.show()