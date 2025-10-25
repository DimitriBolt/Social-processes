"""
Задача (Example 6.2, секция 6.1: Линейно-ограниченная квадратичная оптимизация):
    minimize   (1/2) * w^T Σ w
    subject to 1^T w = 1
               r^T w = d

Где:
- Σ: ковариационная матрица доходностей активов
- r: вектор ожидаемых доходностей
- d: целевой ожидаемый доход портфеля (здесь d = 0.1 = 10% годовых)
- w: веса портфеля

Решение через KKT (как в книге):
    [ Σ   r   1 ] [ w       ] = [ 0 ]
    [ r^T 0   0 ] [ λ_return]   [ d ]
    [ 1^T 0   0 ] [ λ_budget]   [ 1 ]
Решаем линейную систему на (w, λ_return, λ_budget).
"""

import numpy as np
import glob
import pandas as pd
from pandas import DataFrame
from log_returns import Log_returns

if __name__ == '__main__':
    # -----------------------------
    # 1) Подготовка данных
    # -----------------------------
    # open csv file nasdaq_screener_*.csv to pandas dataframe from current directory
    csv_path = glob.glob("./nasdaq_screener_*.csv")[0]
    df = pd.read_csv(csv_path)
    symbols: list[str] = df["Symbol"].astype(str).dropna().reset_index(drop=True).tolist()

    log_returns: Log_returns = Log_returns(symbols_local=symbols,
                                           start_date_local='2025-01-01',
                                           end_date_local='2025-10-01')
    detrended_returns_df: DataFrame = log_returns.get_log_returns()
    detrended_returns_df = detrended_returns_df.dropna(axis=0, how='any')


    # Calculate the covariance matrix of the detrended returns
    covariance_matrix = detrended_returns_df.cov()

    # Вектор ожидаемых доходностей r — средние по столбцам (по активам)
    # (для дневных лог-доходностей это средние дневные; далее мы их годуем)
    r_daily = detrended_returns_df.mean().to_numpy()            # shape (n_assets,)
    Sigma_daily = covariance_matrix.to_numpy()         # shape (n_assets, n_assets)

    # -----------------------------
    # 2) Приведение к годовым величинам
    # -----------------------------
    TRADING_DAYS = 252
    r = r_daily * TRADING_DAYS             # годовые ожидания
    Sigma = Sigma_daily * TRADING_DAYS     # годовая ковариация (дисперсия масштабир. ~ T)

    # Целевой доход: 10% годовых
    d = 0.1

    # -----------------------------
    # 3) Сборка и решение KKT-системы
    # -----------------------------
    n = Sigma.shape[0]
    ones = np.ones(n)

    # Блоковая матрица и правые части (как в примере 6.2)
    K = np.block([
        [Sigma,               r.reshape(-1, 1),    ones.reshape(-1, 1)],
        [r.reshape(1, -1),    np.zeros((1, 1)),    np.zeros((1, 1))   ],
        [ones.reshape(1, -1), np.zeros((1, 1)),    np.zeros((1, 1))   ]
    ])
    rhs = np.concatenate([np.zeros(n), np.array([d, 1.0])])

    solution = np.linalg.solve(K, rhs)
    w = solution[:n]                # искомые веса
    lambda_return = solution[n]     # множитель Лагранжа при r^T w = d
    lambda_budget = solution[n+1]   # множитель Лагранжа при 1^T w = 1

    # -----------------------------
    # 4) Проверки и вывод
    # -----------------------------
    exp_return = float(r @ w)                   # должно быть близко к d (=0.1)
    variance = float(w @ (Sigma @ w))           # w^T Σ w
    std_dev = float(np.sqrt(variance))          # σ портфеля

    # Красивый вывод с именами активов
    asset_names = list(detrended_returns_df.columns)
    weights_series = pd.Series(w, index=asset_names, name="weight")

    print("=== Минимальная дисперсия при целевом доходе d = 10% годовых (Example 6.2) ===")
    print(f"Сумма весов 1^T w         = {weights_series.sum():.10f}")
    print(f"Ожидаемый доход r^T w     = {exp_return:.6f}")
    print(f"Дисперсия портфеля w^TΣw  = {variance:.8f}")
    print(f"Стандартное отклонение σ  = {std_dev:.6f}\n")

    print("Веса портфеля (w):")
    for name, wi in weights_series.items():
        print(f"{name:>20s}: {wi:+.8f}")

