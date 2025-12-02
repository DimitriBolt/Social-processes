# plot_eig_hist_SigmaL_vs_SigmaRPCA.py
# ------------------------------------------------------------
# Compare eigenvalue distributions of Sigma_L and Sigma_RPCA (annualized).
# Inputs:  Sigma_L.parquet, Sigma_RPCA.parquet
# Output:  One figure, overlaid histograms (log-x).
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Sigma_L = pd.read_parquet("Sigma_L.parquet").astype("float64").values
Sigma_RP = pd.read_parquet("Sigma_RPCA.parquet").astype("float64").values

# Symmetric -> use eigvalsh for numerical stability
evals_L  = np.linalg.eigvalsh(Sigma_L)
evals_RP = np.linalg.eigvalsh(Sigma_RP)

# Keep strictly positive for log-x visualization
evals_L  = evals_L[evals_L  > 0]
evals_RP = evals_RP[evals_RP > 0]

plt.figure(figsize=(8, 5))
bins = 50
plt.hist(evals_L,  bins=bins, alpha=0.5, label=r"$\Sigma_L$",  density=True)
plt.hist(evals_RP, bins=bins, alpha=0.5, label=r"$\Sigma_{\mathrm{RPCA}}$", density=True)
plt.xscale("log")
plt.xlabel("Eigenvalue (log scale)")
plt.ylabel("Density (histogram)")
plt.title("Eigenvalue spectra: $\Sigma_L$ vs $\Sigma_{\mathrm{RPCA}}$ (annualized)")
plt.legend()
plt.tight_layout()
plt.show()