# plot_corr_L0.py
# ------------------------------------------------------------
# Heatmap of Corr(L_original) i.e., factor correlation structure in original units.
# Input : L_original.parquet
# Output: corr_L_original.png
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    L0 = pd.read_parquet("L_original.parquet").astype("float64")
    R_L = L0.corr()

    plt.figure(figsize=(7.5, 6))
    im = plt.imshow(R_L.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.title("Correlation heatmap of L_original")
    plt.xlabel("Assets")
    plt.ylabel("Assets")
    plt.tight_layout()

    plt.savefig("corr_L_original.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
