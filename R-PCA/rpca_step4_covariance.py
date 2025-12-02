# rpca_step4_covariance.py
# ------------------------------------------------------------
# Purpose (educational):
#   From RPCA output (L_original), build covariance/correlation in original units.
#   IMPORTANT: We build TWO variants:
#     (A) Factor-only: Sigma_L = Cov(L)
#     (B) RPCA factor + specific diag: Sigma_RPCA = Cov(L) + diag(Var(E))
#         where E = (original returns - shift) - L  ~ specific component
#
#   This (B) variant dramatically improves conditioning for optimization, without RMT.
# ------------------------------------------------------------
# Inputs:
#   - L_original.parquet          : low-rank component in original units (from Step 3)
#   - center_shift_vector.csv     : per-column shift used in Step 1 (for residuals)
#   - detrended_returns_df.csv    : original detrended daily log-returns (rows=dates, cols=assets)
#
# Outputs:
#   - Sigma_L_daily.parquet, Sigma_L.parquet             (factor-only)
#   - Sigma_RPCA_daily.parquet, Sigma_RPCA.parquet       (factor + diag specific risk)
#   - R_L.parquet                                         correlation from L
#   - mu_L_daily.csv, mu_L_annual.csv                     mean returns from L
#
# Additionally, we set markowitz-like names for (B):
#   Sigma_daily = Sigma_RPCA_daily
#   Sigma       = Sigma_RPCA
#
# Notes:
# - Clarity > robustness: assume ideal data.
# - Annualization scalar does not change condition number, but we print both for clarity.

from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

TRADING_DAYS: int = 252

# -----------------------------
# 1) Load data
# -----------------------------
L_df_orig: pd.DataFrame = pd.read_parquet("L_original.parquet")
shift_vec: pd.Series = pd.read_csv("center_shift_vector.csv", index_col=0).iloc[:, 0].astype(float)
X_orig: pd.DataFrame = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
X_orig.index = pd.to_datetime(X_orig.index)

# Align columns just in case (educational: we assume they match)
L_df_orig = L_df_orig.loc[:, X_orig.columns]
shift_vec = shift_vec.loc[X_orig.columns]

# -----------------------------
# 2) Factor-only objects from L
# -----------------------------
Sigma_L_daily: pd.DataFrame = L_df_orig.cov()
Sigma_L: pd.DataFrame = Sigma_L_daily * TRADING_DAYS
R_L: pd.DataFrame = L_df_orig.corr()
mu_L_daily: pd.Series = L_df_orig.mean(axis=0)
mu_L_annual: pd.Series = mu_L_daily * TRADING_DAYS

# -----------------------------
# 3) Add specific risk (diagonal) from residuals
# -----------------------------
# Residuals (specific component) in original units:
# E = (original returns - shift) - L  ~ should approximate S (up to demeaning in cov)
E_df: pd.DataFrame = X_orig.subtract(shift_vec, axis=1) - L_df_orig

# Diagonal specific variance, daily
spec_var_daily: pd.Series = E_df.var(axis=0, ddof=1)

# Build diagonal matrix as DataFrame (to preserve labels)
Spec_daily = pd.DataFrame(np.diag(spec_var_daily.values),
                          index=spec_var_daily.index,
                          columns=spec_var_daily.index)

# RPCA covariance = factor + specific diag (daily and annualized)
Sigma_RPCA_daily: pd.DataFrame = Sigma_L_daily + Spec_daily
Sigma_RPCA: pd.DataFrame = Sigma_RPCA_daily * TRADING_DAYS

# (Optional) tiny ridge to guarantee PD (scale-invariant):
ridge = 1e-6 * float(np.trace(Sigma_RPCA_daily.values)) / Sigma_RPCA_daily.shape[0]
Sigma_RPCA_daily = Sigma_RPCA_daily + ridge * np.eye(Sigma_RPCA_daily.shape[0])
Sigma_RPCA = Sigma_RPCA_daily * TRADING_DAYS  # re-apply annualization after ridge

# -----------------------------
# 4) Mirror markowitz_portfolio.py names for (B)
# -----------------------------
Sigma_daily: pd.DataFrame = Sigma_RPCA_daily.copy()
Sigma: pd.DataFrame = Sigma_RPCA.copy()

# -----------------------------
# 5) Save artifacts
# -----------------------------
Sigma_L_daily.to_parquet("Sigma_L_daily.parquet")
Sigma_L.to_parquet("Sigma_L.parquet")
R_L.to_parquet("R_L.parquet")
mu_L_daily.to_csv("mu_L_daily.csv")
mu_L_annual.to_csv("mu_L_annual.csv")

Sigma_RPCA_daily.to_parquet("Sigma_RPCA_daily.parquet")
Sigma_RPCA.to_parquet("Sigma_RPCA.parquet")

# Also save the 'active' matrices used for portfolio (B)
Sigma_daily.to_parquet("Sigma_daily.parquet")
Sigma.to_parquet("Sigma.parquet")

# -----------------------------
# 6) Diagnostics
# -----------------------------
cond_factor_only = float(np.linalg.cond(Sigma_L.values))
cond_rpca = float(np.linalg.cond(Sigma.values))
print("[INFO] Condition numbers (annualized):")
print(f"  cond(Sigma_L  = Cov(L))         : {cond_factor_only:,.0f}")
print(f"  cond(Sigma_RPCA = Cov(L)+diag)  : {cond_rpca:,.0f}")

print("\n[INFO] Saved files:")
print("  - Sigma_L_daily.parquet / Sigma_L.parquet / R_L.parquet")
print("  - mu_L_daily.csv / mu_L_annual.csv")
print("  - Sigma_RPCA_daily.parquet / Sigma_RPCA.parquet")
print("  - Sigma_daily.parquet / Sigma.parquet  (markowitz-compatible)")

# -----------------------------
# 5) Plot гистограмму эмпирических собственных значений of the covariance matrix
# -----------------------------
from matplotlib import pyplot as plt
eigenvalues_rpca = np.linalg.eigvalsh(Sigma)
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues_rpca, bins=50, edgecolor='black')
plt.title('Гистограмма эмпирических собственных значений ковариационной матрицы Σ')
plt.xlabel('Собственные значения')
plt.ylabel('Частота')
plt.grid(axis='y', alpha=0.75)
plt.show()
# Save the plot
# plt.savefig('eigenvalue_histogram.png')

