# plot_corr_empirical.py
# ------------------------------------------------------------
# Heatmap of empirical Corr(X) from detrended log-returns (original units).
# Input : ../detrended_returns_df_200.csv
# Output: corr_empirical.png  (saved to current directory)
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    # 1) Load returns (rows = dates, columns = assets)
    X = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)
    X = X.select_dtypes(include=[np.number]).astype("float64")

    # 2) Correlation matrix
    R_emp = X.corr()

    # 3) Plot (no explicit colors; default style as requested)
    plt.figure(figsize=(7.5, 6))
    im = plt.imshow(R_emp.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.title("Empirical correlation heatmap of X")
    plt.xlabel("Assets")
    plt.ylabel("Assets")
    plt.tight_layout()

    # 4) Save instead of show
    plt.savefig("corr_empirical.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
