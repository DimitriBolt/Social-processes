# markowitz_portfolio_R_PCA_step5.py
# ------------------------------------------------------------
# Markowitz minimum-variance portfolio with a target return (Example 6.2 style),
# but using RPCA-based covariance Σ_RPCA instead of empirical Σ.
#
# Key differences vs markowitz_portfolio.py:
#   - Σ is loaded from "Sigma_RPCA.parquet" (annualized) produced at RPCA Step 4.
#   - Plot is saved as "weight_change_l1_norm_timeseries_200_R_PCA.png".
#
# All variable names and the KKT block structure mirror the original script
# to ease side-by-side comparison.
# ------------------------------------------------------------
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker  # for PercentFormatter

# -----------------------------
# 1) Data ingestion
# -----------------------------
# Expect a CSV with detrended daily log-returns (columns = assets, rows = dates).
detrended_returns_df = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)

# -----------------------------
# 2) Build annualized Σ (RPCA version)
# -----------------------------
TRADING_DAYS = 252

# We do NOT compute empirical Σ here.
# Instead, we load the RPCA-based (annualized) covariance saved by rpca_step4_covariance.py.
# Shape: (n, n)
Sigma_RPCA_df = pd.read_parquet("Sigma_RPCA.parquet").astype("float64")
Sigma = Sigma_RPCA_df.to_numpy()  # already annualized

# Optional: print condition number for diagnostics (scale-invariant)
print(f"[INFO] cond(Σ_RPCA): {np.linalg.cond(Sigma):,.0f}")

# Target annual return d = 10%
d = 0.1

# Prepare containers for time series of relative l1 norm of weight changes
dates = []
relative_delta_w_l1_norms = []

n = Sigma.shape[0]
ones = np.ones(n)
w_yesterday = np.zeros(n)  # initial weights (zero for first day)

# -----------------------------
# 3) Loop over dates and solve KKT each day
# -----------------------------
# We reuse the exact KKT structure from the original script:
#   minimize (1/2) w^T Σ w
#   subject to 1^T w = 1
#              r^T w = d
# This yields the linear system:
#   [ Σ    r     1 ] [ w        ] = [ 0 ]
#   [ r^T  0     0 ] [ λ_return ]   [ d ]
#   [ 1^T  0     0 ] [ λ_budget ]   [ 1 ]
for date in detrended_returns_df.index:
    r_daily = detrended_returns_df.loc[date]             # shape (n,)
    r = r_daily.to_numpy() * TRADING_DAYS                # annualized expected returns for that date

    K = np.block([
        [Sigma,               r.reshape(-1, 1),    ones.reshape(-1, 1)],
        [r.reshape(1, -1),    np.zeros((1, 1)),    np.zeros((1, 1))   ],
        [ones.reshape(1, -1), np.zeros((1, 1)),    np.zeros((1, 1))   ]
    ])
    rhs = np.concatenate([np.zeros(n), np.array([d, 1.0])])

    solution = np.linalg.solve(K, rhs)
    w_today = solution[:n]
    # lambda_return = solution[n]
    # lambda_budget = solution[n+1]

    # Relative l1 change of weights vs previous day
    delta_w = w_today - w_yesterday
    delta_w_l1_norm: float = np.linalg.norm(delta_w, 1)
    relative_delta_w_l1_norm = delta_w_l1_norm / np.linalg.norm(w_yesterday, 1) if np.linalg.norm(w_yesterday, 1) > 0 else 0.0

    dates.append(datetime.strptime(date, "%Y-%m-%d").date())
    relative_delta_w_l1_norms.append(relative_delta_w_l1_norm)

    # Sanity checks (same as original)
    assert np.isclose(w_today.sum(), 1.0)
    assert np.isclose(r @ w_today, d)

    w_yesterday = w_today.copy()

# -----------------------------
# 4) Plot time series of relative l1 norm of weight changes
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(dates[1:], relative_delta_w_l1_norms[1:], marker='o', linestyle='--')

# Percent y-axis like the original
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_ylim(bottom=0, top=4.00)  # 0% to 400%

ax.set_title(r'$\frac{||\vec{\Delta w}||_{l^1}}{||\vec{w}||_{l^1}}$ over Time (RPCA Σ)')
ax.set_xlabel('Date')
ax.set_ylabel(r'$\frac{||\vec{\Delta w}||_{l^1}}{||\vec{w}||_{l^1}}$ (%)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()

# Save with RPCA-specific filename
plt.savefig('weight_change_l1_norm_timeseries_200_R_PCA.png', format='png')
plt.show()

# -----------------------------
# End of markowitz_portfolio_R_PCA_step5.py
# -----------------------------
