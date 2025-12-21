# plot_condition_numbers_bar.py
# ------------------------------------------------------------
# Bar chart of condition numbers: empirical Σ, Σ_L, Σ_RPCA (annualized for comparability).
# Inputs :
#   - ../detrended_returns_df_200.csv  (to compute empirical Σ)
#   - Sigma_L.parquet
#   - Sigma_RPCA.parquet
# Output : condition_numbers_bar.png
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADING_DAYS = 252

def main() -> None:
    X = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
    X = X.select_dtypes(include=[np.number]).astype("float64")
    Sigma_emp = X.cov().values * TRADING_DAYS

    Sigma_L  = pd.read_parquet("Sigma_L.parquet").astype("float64").values
    Sigma_RP = pd.read_parquet("Sigma_RPCA.parquet").astype("float64").values

    labels = ["Empirical Σ", "Σ_L (factor-only)", "Σ_RPCA (factor+diag)"]
    conds  = [
        float(np.linalg.cond(Sigma_emp)),
        float(np.linalg.cond(Sigma_L)),
        float(np.linalg.cond(Sigma_RP)),
    ]

    plt.figure(figsize=(7, 4.5))
    plt.bar(range(3), conds)
    plt.xticks(range(3), labels, rotation=10)
    plt.ylabel("Condition number")
    plt.title("Condition numbers: empirical vs Σ_L vs Σ_RPCA (annualized)")
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig("condition_numbers_bar.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
