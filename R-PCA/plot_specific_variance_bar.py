# plot_specific_variance_bar.py
# ------------------------------------------------------------
# Bar chart of specific variances diag(Var(S_original)) — top-N assets.
# Input : S_original.parquet
# Output: specific_variance_bar_top30.png
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TOP_N = 30

def main() -> None:
    S0 = pd.read_parquet("S_original.parquet").astype("float64")
    spec_var = S0.var(axis=0)  # per-column variance (daily units)
    spec_var_sorted = spec_var.sort_values(ascending=False).head(TOP_N)

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(spec_var_sorted)), spec_var_sorted.values)
    plt.xticks(range(len(spec_var_sorted)), spec_var_sorted.index, rotation=90)
    plt.ylabel("Specific variance (daily units)")
    plt.title(f"Top-{TOP_N} specific variances from S_original")
    plt.tight_layout()

    plt.savefig("specific_variance_bar_top30.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
