# ------------------------------------------------------------
# Purpose: Robust PCA (RPCA) via ADMM on centered+scaled returns
#   Step 3 — RPCA via a simple ADMM loop (clean and readable)
# ------------------------------------------------------------
# This script adds a SINGLE, CLEAR function to perform Robust PCA (RPCA) on the centered-and-scaled returns matrix from Step 1 & 2.
#
# Pipeline reminder:
#   - Step 1 (done earlier): center columns by subtracting the median.
#   - Step 2 (done earlier): scale columns by MAD*1.4826 (robust scale).
#   - Step 3 (here): run RPCA on the centered+scaled matrix:
#         X_cs = L + S
#       where L is low-rank (market/sector structure), S is sparse (outliers).
#
# After RPCA, we also PROVIDE an "inverse transform" back to original units:
#   X  = X_cs * scale + shift
#   L0 = L     * scale              (low-rank in original units)
#   S0 = S     * scale              (sparse in original units)
# (Per-column operations: multiply each column of L and S by its original scale, and, when reconstructing the full X, add the per-column shift vector.)
#
# Notes (by design, for readability over performance/robustness):
#   - We use a plain ADMM loop with full SVD for clarity (not efficient for huge matrices).
#   - Parameters (lambda/mu/tol) are given friendly defaults; tune them later as needed.

import pandas as pd
import numpy as np
from typing import Tuple

# -----------------------------
# 0) Load preprocessed artifacts (from Step 1 & 2)
# -----------------------------
# Files produced by previous steps:
#   - returns_centered_scaled.parquet  (X_cs)
#   - center_shift_vector.csv          (per-column shift used in centering)
#   - scale_vector.csv                 (per-column robust scale used in scaling)
# X_cs: centered + scaled returns
X_cs = pd.read_parquet("returns_centered_scaled.parquet")

# Read shift and scale as 1D Series (pandas >= 2.0 has no 'squeeze' kwarg)
shift_vec = pd.read_csv("center_shift_vector.csv", index_col=0).iloc[:, 0].astype(float)
scale_vec = pd.read_csv("scale_vector.csv", index_col=0).iloc[:, 0].astype(float)

# -----------------------------
# Utility: proximal operators
# -----------------------------
def soft_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    """
    Elementwise soft-thresholding.
    Used for the sparse update: S <- soft_threshold(X - L + U, tau).
    """
    return np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)


def singular_value_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    """
    Singular Value Thresholding (SVT):
    L <- U * shrink(Sigma, tau) * V^T
    where shrink(s) = max(s - tau, 0).
    """
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    s_shrunk = np.maximum(s - tau, 0.0)
    return (U * s_shrunk) @ Vt


# -----------------------------
# Step 3: RPCA via ADMM (clean & readable)
# -----------------------------
def rpca_admm(
    X: np.ndarray,
    lam: float = None,    # sparsity weight; if None, default to 1/sqrt(max(T,N))
    mu: float = 1.0,      # ADMM penalty parameter (augmented Lagrangian)
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust PCA (Principal Component Pursuit) via a simple ADMM loop.
    Decompose X into X = L + S, where L is low-rank and S is sparse.

    Parameters
    ----------
    X : np.ndarray
        2D data array (T x N), typically centered and scaled.
    lam : float
        L1 penalty weight for S. Default heuristic is 1/sqrt(max(T,N)).
    mu : float
        ADMM penalty parameter (controls the SVT and soft-threshold magnitudes).
    max_iter : int
        Maximum number of ADMM iterations.
    tol : float
        Stopping tolerance on relative residual and variable changes.

    Returns
    -------
    L : np.ndarray
        Low-rank matrix (same shape as X).
    S : np.ndarray
        Sparse matrix (same shape as X).

    ADMM updates (high level):
        L_{k+1} = SVT( X - S_k + U_k, 1/mu )
        S_{k+1} = soft_threshold( X - L_{k+1} + U_k, lam/mu )
        U_{k+1} = U_k + ( X - L_{k+1} - S_{k+1} )
    """
    T, N = X.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(T, N))

    # Initialize variables
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    U = np.zeros_like(X)

    # Precompute norms for stopping criteria
    X_fro = np.linalg.norm(X, ord='fro') + 1e-12  # avoid division by zero

    for it in range(1, max_iter + 1):
        # L-update (low-rank via Singular Value Thresholding)
        L = singular_value_threshold(X - S + U, tau=1.0 / mu)

        # S-update (sparse via soft-thresholding)
        S = soft_threshold(X - L + U, tau=lam / mu)

        # Dual variable update
        U = U + (X - L - S)

        # Stopping conditions (simple and readable)
        primal_res = np.linalg.norm(X - L - S, ord='fro') / X_fro
        # monitor relative changes in L and S for stability
        # (optional: could compute norms of differences, omitted for brevity)

        if it % 50 == 0 or it == 1:
            print(f"[ADMM] iter={it:4d} | primal_res={primal_res:.2e}")

        if primal_res < tol:
            print(f"[ADMM] Converged at iter={it} with primal_res={primal_res:.2e}")
            break

    return L, S


if __name__ == "__main__":
    # --------------------------------------------------------
    # 1) Prepare the matrix X_cs for RPCA (numpy view)
    # --------------------------------------------------------
    # X_cs is centered+scaled returns (T x N). We feed it directly into RPCA.
    X_mat = X_cs.values

    # --------------------------------------------------------
    # 2) Run RPCA (ADMM) with educational defaults
    # --------------------------------------------------------
    # - lam default is 1/sqrt(max(T,N)) if omitted.
    # - mu=1.0 is a reasonable starting value for many panels.
    # - tol=1e-4 and max_iter=1000 are standard educational settings.
    L_mat, S_mat = rpca_admm(X_mat, lam=None, mu=1.0, max_iter=1000, tol=1e-4)

    # Wrap results back into DataFrames with original index/columns
    L_df_scaled = pd.DataFrame(L_mat, index=X_cs.index, columns=X_cs.columns)
    S_df_scaled = pd.DataFrame(S_mat, index=X_cs.index, columns=X_cs.columns)

    # Save the RPCA results in the centered+scaled space
    L_df_scaled.to_parquet("L_scaled.parquet")
    S_df_scaled.to_parquet("S_scaled.parquet")

    # --------------------------------------------------------
    # 3) Inverse transform: back to original units
    # --------------------------------------------------------
    # Recall:
    #   X  = X_cs * scale + shift
    #   L0 = L     * scale
    #   S0 = S     * scale
    #
    # Column-wise operations: multiply each column by its scale.
    # We do NOT add the shift to L or S (shift is the baseline for X).
    L_df_orig = L_df_scaled.multiply(scale_vec, axis=1)
    S_df_orig = S_df_scaled.multiply(scale_vec, axis=1)

    # Save low-rank and sparse components in original units
    L_df_orig.to_parquet("L_original.parquet")
    S_df_orig.to_parquet("S_original.parquet")

    # (Optional) Reconstruct X in original units to sanity-check:
    # X_recon = L0 + S0 + shift
    X_recon = L_df_orig.add(S_df_orig, fill_value=0.0).add(shift_vec, axis=1)
    X_recon.to_parquet("X_reconstructed_from_RPCA.parquet")

    # --------------------------------------------------------
    # 4) Quick sanity checks
    # --------------------------------------------------------
    # 4.1) Check decomposition quality in scaled space
    residual_scaled = X_cs - (L_df_scaled + S_df_scaled)
    rel_res = np.linalg.norm(residual_scaled.values, ord='fro') / (
        np.linalg.norm(X_cs.values, ord='fro') + 1e-12
    )
    print("\nSanity — relative residual in scaled space (should be ~ tol or lower):")
    print(f"  rel_res = {rel_res:.3e}")

    # 4.2) Quick look at ranks and sparsity (very rough indicators)
    # Rank estimate: count singular values above a small threshold
    _, svals, _ = np.linalg.svd(L_df_scaled.values, full_matrices=False)
    rank_est = int((svals > 1e-6).sum())
    sparsity = (np.abs(S_df_scaled.values) < 1e-8).mean()  # fraction of (near-)zeros

    print("\nSanity — rough structure measures (scaled space):")
    print(f"  estimated rank(L) ~ {rank_est}")
    print(f"  sparsity(S) ~ {sparsity:.2%}")

    print("\n[INFO] RPCA completed. Files saved:")
    print(" - L_scaled.parquet, S_scaled.parquet")
    print(" - L_original.parquet, S_original.parquet")
    print(" - X_reconstructed_from_RPCA.parquet")
