#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Markowitz minimum-variance portfolio with a target return (Example 6.2 style).

We solve the quadratic program:
    minimize   (1/2) * w^T Σ w
    subject to 1^T w = 1
               r^T w = d

KKT = Karush–Kuhn–Tucker conditions.
For this strictly convex quadratic problem with linear equality constraints,
the KKT system is linear and gives the unique optimum (assuming Σ is SPD):

    [ Σ    r     1 ] [ w            ] = [ 0 ]
    [ r^T  0     0 ] [ λ_return     ]   [ d ]
    [ 1^T  0     0 ] [ λ_budget     ]   [ 1 ]

We then solve this linear system for (w, λ_return, λ_budget).

Notes on annualization:
- If r_daily is the vector of mean daily log-returns and Σ_daily is the daily covariance,
  a standard scaling is:
      r_annual = 252 * r_daily
      Σ_annual = 252 * Σ_daily
- This implies standard deviation scales as sqrt(252).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Data ingestion
# -----------------------------
# Expect a CSV with detrended daily log-returns (columns = assets, rows = dates).
# This file is produced by your own pipeline.
df_ret = pd.read_csv("detrended_returns_df.csv", index_col=0)

# -----------------------------
# 2) Build annualized r and Σ
# -----------------------------
TRADING_DAYS = 252
r_daily = df_ret.mean().to_numpy()       # shape (n,)
Sigma_daily = df_ret.cov().to_numpy()    # shape (n, n)

r = r_daily * TRADING_DAYS
Sigma = Sigma_daily * TRADING_DAYS

# Target annual return d = 10%
d = 0.1

# -----------------------------
# 3) Solve KKT linear system
# -----------------------------
n = Sigma.shape[0]
ones = np.ones(n)

# Block matrix and right-hand side (rhs = right-hand side vector of the linear system).
K = np.block([
    [Sigma,               r.reshape(-1, 1),    ones.reshape(-1, 1)],
    [r.reshape(1, -1),    np.zeros((1, 1)),    np.zeros((1, 1))   ],
    [ones.reshape(1, -1), np.zeros((1, 1)),    np.zeros((1, 1))   ]
])
rhs = np.concatenate([np.zeros(n), np.array([d, 1.0])])

solution = np.linalg.solve(K, rhs)
w = solution[:n]
lambda_return = solution[n]
lambda_budget = solution[n+1]

# -----------------------------
# 4) Report scalar diagnostics (do not print the weight vector)
# -----------------------------
exp_return = float(r @ w)                # should be close to d
variance = float(w @ (Sigma @ w))        # portfolio variance
std_dev = float(np.sqrt(variance))       # portfolio std
cond_number = float(np.linalg.cond(Sigma))  # condition number of Σ

print("=== Minimum-Variance Portfolio (target d = 10% annual) ===")
print(f"Sum of weights (1^T w):       {w.sum():.10f}")
print(f"Target return (d):            {d:.6f}")
print(f"Achieved expected return:     {exp_return:.6f}")
print(f"Portfolio variance (w^TΣw):   {variance:.8f}")
print(f"Portfolio std (sqrt):         {std_dev:.6f}")
print(f"Condition number cond(Σ):     {cond_number:.6e}")

# -----------------------------
# 5) Plot spectrum of the covariance matrix
# -----------------------------
# Compute and plot eigenvalues of Σ (symmetric -> use eigvalsh).
eigs = np.linalg.eigvalsh(Sigma)   # non-decreasing order
eigs_sorted = np.sort(eigs)[::-1]  # descending for plotting aesthetics

plt.figure()
plt.plot(eigs_sorted, marker='o', linestyle='-')
plt.title("Covariance Matrix Spectrum")
plt.xlabel("Index (sorted by magnitude)")
plt.ylabel("Eigenvalue")
plt.grid(True)
plt.tight_layout()
plt.show()
