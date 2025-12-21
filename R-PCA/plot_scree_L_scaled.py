# plot_scree_L_scaled.py
# ------------------------------------------------------------
# Scree plot of singular values of L (in centered+scaled units).
# Input : L_scaled.parquet
# Output: scree_L_scaled.png
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main() -> None:
    L = pd.read_parquet("L_scaled.parquet").astype("float64").values
    # Full SVD for didactic clarity
    _, s, _ = np.linalg.svd(L, full_matrices=False)

    plt.figure(figsize=(8, 5))
    plt.semilogy(np.arange(1, len(s)+1), s, marker="o", linestyle="-")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value (log scale)")
    plt.title("Scree plot: singular values of L (scaled)")
    plt.tight_layout()

    plt.savefig("scree_L_scaled.png", dpi=200, bbox_inches="tight")
    # plt.show()
    plt.close()

if __name__ == "__main__":
    main()
