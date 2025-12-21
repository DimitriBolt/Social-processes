# rpca_step4_covariance.py
# ------------------------------------------------------------
# Step 4: Build covariance/correlation in the original units from RPCA outputs.
# EDUCATIONAL VERSION: highly commented, straightforward, no try/except.
#
# INPUTS (produced by Step 3):
#   - L_original.parquet : L0   (T x N) low-rank component in original units
#   - S_original.parquet : S0   (T x N) sparse component in original units
#
# ALSO USED for the empirical Σ histogram:
#   - ../detrended_returns_df_200.csv : X (T x N) detrended daily log-returns
#
# OUTPUTS:
#   - Sigma_L_daily.parquet : Cov(L0)                    (N x N, daily units)
#   - Sigma_L.parquet       : Cov(L0) * 252              (N x N, annualized)
#   - R_L.parquet           : Corr(L0)                   (N x N)
#   - mu_L_daily.csv        : mean(L0, axis=0)           (N,   daily units)
#   - mu_L_annual.csv       : mean(L0, axis=0) * 252     (N,   annualized)
#   - Sigma_RPCA_daily.parquet : Cov(L0) + diag(Var(S0)) (N x N, daily units)
#   - Sigma_RPCA.parquet       : [above] * 252           (N x N, annualized)
#
#   - eigen_hist_empirical_sigma.png : Histogram of empirical Σ eigenvalues
#                                      (Σ_emp = Cov(X) * 252), saved to disk.
#                                      NOTE: figure is SAVED, not shown.
#
# WHY add diag(Var(S0))?
#   Σ_L = Cov(L0) captures the factor/common structure but misses idiosyncratic risk.
#   Adding diag(Var(S0)) restores specific (asset-level) variances and usually
#   improves the conditioning dramatically: Σ_RPCA = Σ_L + diag(Var(S0)).
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- 0) Global constants ----
TRADING_DAYS: int = 252  # annualization factor

def main() -> None:
    # --------------------------------------------------------
    # 1) Load RPCA artifacts in the ORIGINAL units (T x N)
    # --------------------------------------------------------
    L0: pd.DataFrame = pd.read_parquet("L_original.parquet").astype("float64")
    S0: pd.DataFrame = pd.read_parquet("S_original.parquet").astype("float64")

    # Shapes (rows=dates=T, cols=assets=N)
    # print(f"L0 shape: {L0.shape}, S0 shape: {S0.shape}")

    # --------------------------------------------------------
    # 2) Factor covariance from L0 (daily) and annualized
    #    Pandas .cov() -> sample covariance (ddof=1) column-wise.
    # --------------------------------------------------------
    Sigma_L_daily: pd.DataFrame = L0.cov()                 # (N x N), daily units
    Sigma_L: pd.DataFrame       = Sigma_L_daily * TRADING_DAYS  # annualized

    # --------------------------------------------------------
    # 3) Specific variances from S0 (daily), and Σ_RPCA (daily & annualized)
    #    E = (X - 1 m^T) - L0 = S0  in our pipeline, so Var(E) = Var(S0).
    # --------------------------------------------------------
    spec_var_daily: pd.Series = S0.var(axis=0)  # length-N series, daily units

    # Start from Σ_L_daily, then add the specific variances on the diagonal:
    Sigma_RPCA_daily: pd.DataFrame = Sigma_L_daily.copy()
    # Add diag(Var(S0)) to the diagonal of Σ_L_daily:
    ii = np.diag_indices_from(Sigma_RPCA_daily.values)
    Sigma_RPCA_daily.values[ii] += spec_var_daily.values

    # Annualized RPCA covariance:
    Sigma_RPCA: pd.DataFrame = Sigma_RPCA_daily * TRADING_DAYS

    # --------------------------------------------------------
    # 4) Correlation of L0 and the means (daily & annualized)
    # --------------------------------------------------------
    R_L: pd.DataFrame       = L0.corr()                    # (N x N)
    mu_L_daily: pd.Series   = L0.mean(axis=0)              # (N,), daily units
    mu_L_annual: pd.Series  = mu_L_daily * TRADING_DAYS    # (N,), annualized

    # --------------------------------------------------------
    # 5) SAVE all matrices/vectors to disk
    # --------------------------------------------------------
    Sigma_L_daily.to_parquet("Sigma_L_daily.parquet")
    Sigma_L.to_parquet("Sigma_L.parquet")
    R_L.to_parquet("R_L.parquet")
    mu_L_daily.to_csv("mu_L_daily.csv")
    mu_L_annual.to_csv("mu_L_annual.csv")
    Sigma_RPCA_daily.to_parquet("Sigma_RPCA_daily.parquet")
    Sigma_RPCA.to_parquet("Sigma_RPCA.parquet")

    # --------------------------------------------------------
    # 6) EXTRA (as in your earlier version): histogram of empirical Σ eigenvalues
    #    We compute Σ_emp from the original returns X and save the figure to disk.
    #    Everything is in ENGLISH; the plot is SAVED, not shown.
    # --------------------------------------------------------
    X: pd.DataFrame = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
    X = X.select_dtypes(include=[np.number]).astype("float64")

    # Empirical (annualized) covariance:
    Sigma_emp: np.ndarray = (X.cov().values * TRADING_DAYS)  # (N x N)

    # Eigenvalues (symmetric PSD expected, use eigvalsh)
    evals_emp: np.ndarray = np.linalg.eigvalsh(Sigma_emp)
    # For log x-scale, restrict to strictly positive values (if any zeros due to numerics):
    evals_emp = evals_emp[evals_emp > 0]

    # Plot & SAVE (no plt.show())
    plt.figure(figsize=(8, 5))
    plt.hist(evals_emp, bins=60, density=True)
    plt.xscale("log")
    plt.xlabel("Eigenvalue (log scale)")
    plt.ylabel("Density (histogram)")
    plt.title("Histogram of empirical eigenvalues of covariance Σ (annualized)")
    plt.tight_layout()
    plt.savefig("eigen_hist_empirical_sigma.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

    # Optional numeric diagnostics (printed to console; harmless for nbconvert --no-input)
    print(f"cond(Σ_emp):   {np.linalg.cond(Sigma_emp):.3e}")
    print(f"cond(Σ_L):     {np.linalg.cond(Sigma_L.values):.3e}")
    print(f"cond(Σ_RPCA):  {np.linalg.cond(Sigma_RPCA.values):.3e}")

if __name__ == "__main__":
    main()
