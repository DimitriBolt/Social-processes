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
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # 1. Импортируем ticker


# -----------------------------
# 1) Data ingestion
# -----------------------------
# Expect a CSV with detrended daily log-returns (columns = assets, rows = dates).
detrended_returns_df = pd.read_csv("../detrended_returns_df_200.csv", index_col=0)

# -----------------------------
# 2) Build annualized Σ
# -----------------------------
TRADING_DAYS = 252
Sigma_daily = detrended_returns_df.cov()    # shape (n, n)
Sigma = Sigma_daily.to_numpy() * TRADING_DAYS

# Target annual return d = 10%
d = 0.1
# Подготовка списков для сбора данных
dates = []
relative_delta_w_l1_norms = []

n = Sigma.shape[0]
ones = np.ones(n)
w_yesterday = np.zeros(n) # initial weights (zero for first day)



# Loop over each date in the return vector index
for date in detrended_returns_df.index:
    r_daily = detrended_returns_df.loc[date]
    # Build annualized r for the current date
    r = r_daily.to_numpy() * TRADING_DAYS

    # -----------------------------
    # 3) Solve KKT linear system
    # -----------------------------
    # Block matrix and right-hand side (rhs = right-hand side vector of the linear system).
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
    delta_w = w_today - w_yesterday
    delta_w_l1_norm: float = np.linalg.norm(delta_w, 1)
    x1: float = np.sum(np.abs(delta_w_l1_norm)) #Проверка правильности вычисления нормы
    norm2 = np.linalg.norm(w_yesterday, 1)

    relative_delta_w_l1_norm = delta_w_l1_norm / np.linalg.norm(w_yesterday, 1) if np.linalg.norm(w_yesterday, 1) > 0 else 0.0
    # Накопить результаты, сохранять в Series, где индекс - дата, значение - норма изменения весов
    dates.append(datetime.strptime(date, "%Y-%m-%d").date())
    relative_delta_w_l1_norms.append(relative_delta_w_l1_norm)

    # print(f"Date: {date}, relative l1 norm of weight change: {relative_delta_w_l1_norm:.0%}, condition number of KKT: {np.linalg.cond(K):,.0f}")

    # Sanity checks
    # Sum of weights should be 1
    assert np.isclose(w_today.sum(), 1.0)
    # Achieved expected return should be close to target d
    assert np.isclose(r @ w_today, d)
    w_yesterday = w_today.copy()

# -----------------------------
# 4) Condition number of Σ
# -----------------------------
# Condition number gives a sense of numerical stability of the optimization.
# High condition number -> ill-conditioned -> small changes in inputs can lead to large changes in outputs
cond_number = float(np.linalg.cond(Sigma))  # condition number of Σ
print(f"Condition number cond(Σ):     {cond_number:,.0f}")

# -----------------------------
# 5) Plot гистограмму эмпирических собственных значений of the covariance matrix
# -----------------------------
eigenvalues = np.linalg.eigvalsh(Sigma)
plt.figure(figsize=(10, 6))
plt.hist(eigenvalues, bins=50, edgecolor='black')
plt.title('Histogram of the empirical eigenvalues of the covariance matrix Σ')
plt.xlabel('eigenvalues')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()
# Save the plot
# plt.savefig('eigenvalue_histogram.png')

# -----------------------------
# 6) Plot time series of relative l1 norm of weight changes
# -----------------------------
# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(12, 6))

# Используем ax.plot вместо plt.plot
ax.plot(dates[1:], relative_delta_w_l1_norms[1:], marker='o', linestyle='--')

# 3. Создаем и применяем PercentFormatter
# Xmax=1.0 указывает, что значение 1.0 в ваших данных должно отображаться как 100%.
formatter = mticker.PercentFormatter(xmax=1.0)
ax.yaxis.set_major_formatter(formatter)
ax.set_ylim(bottom=0, top=4.00)  # Устанавливаем пределы оси Y от 0% до 400%
# ax.set_ylim(bottom=0, top=0.50)  # Устанавливаем пределы оси Y от 0% до 50% - Только для Степанова!


# Настройка графика
ax.set_title(r'$\frac{||\vec{Δw}||_{l^1}}{||\vec{w}||_{l^1}}$ over Time')
ax.set_xlabel('Date')
ax.set_ylabel(r'$\frac{||\vec{Δw}||_{l^1}}{||\vec{w}||_{l^1}}$ (%)') # Обновляем метку оси для ясности
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
# Save the plot
# plt.savefig('weight_change_l1_norm_timeseries_200.png', format='png') # Не забыть поменять Пределы оси Y до 400%
plt.savefig('weight_change_l1_norm_timeseries_200.png', format='png')

plt.show()
# -----------------------------
# End of markowitz_portfolio.py
# -----------------------------
