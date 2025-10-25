import glob
import os
import pickle
from os.path import expanduser, join
from sklearn.linear_model import LinearRegression

import numpy as np
import yfinance as yf

import pandas
import pandas as pd
from numpy import ndarray, array
from pandas import DataFrame, Series


# Заменять минус в имени на подчёркивание

class Log_returns:
    #  Initializer выполняется перед! основной программой.
    #  Private Instance or static Class attribute. Переменные должны начинаться с двух подчеркиваний.
    # Initial static valus
    __arr: ndarray = array([[100, 200, 300], [400, 500, 600]])

    # Здесь не реализован (а должен был) Constructor WithOut parameters
    def __init__(self, symbols_local: list[str], start_date_local: str, end_date_local: str) -> None:
        self.start_date = start_date_local
        self.symbols: list[str] = symbols_local
        self.end_date: str = end_date_local
        pass

    # Methods
    def __private_method(self, assets: list[str], start: str, end: str) -> DataFrame:
        delisted: list[str] = ['STRD', 'STRF', 'STRC', 'BRKRP', 'STRK', 'SNDK', 'CRCL', 'MCHPP', 'AGNCZ', 'SAIL', 'GLXY', 'CRWV'] + ['BRK/A', 'BRK/B'] + ['EMP']
        failed_downloads = delisted
        assets_without_failed: list[str] = [asset for asset in assets if asset not in failed_downloads]
        # assets_without_failed: list = ['AAPL', 'MSFT', 'GOOGL']
        data: Series = yf.download(tickers=assets_without_failed[0:50], start=start, end=end, timeout=20, auto_adjust=True)['Close']
        log_prices = np.log(data)
        X = np.arange(len(log_prices)).reshape(-1, 1)  # Time index as the independent variable
        y = log_prices  # Logarithmic prices as the dependent variable

        # Fit linear regression: time → all columns
        model = LinearRegression().fit(X, y)

        trend = model.predict(X)
        detrended_log_prices = log_prices - trend
        # Compute returns as differences between current row and previous row of detrended logarithmic prices
        detrended_returns = detrended_log_prices.diff().dropna()

        with open(join(expanduser("~"), "Documents", 'detrended_returns.pickle'), 'wb') as f:
            pickle.dump(detrended_returns, f)
        return detrended_returns

    # Accessor( = getter) methods
    def get_log_returns(self) -> DataFrame:
        # It is used for return outside protected attributes of the object only
        return self.__private_method(self.symbols, self.start_date, self.end_date)


if __name__ == '__main__':
    # open csv file nasdaq_screener_*.csv to pandas dataframe from current directory
    csv_path = glob.glob("./nasdaq_screener_*.csv")[0]
    df = pd.read_csv(csv_path)
    symbols: list[str] = df["Symbol"].astype(str).dropna().reset_index(drop=True).tolist()

    log_returns: Log_returns = Log_returns(symbols_local=symbols,
                                           start_date_local='2025-01-01',
                                           end_date_local='2025-10-01')
    detrended_returns_df: DataFrame = log_returns.get_log_returns()
    detrended_returns_df = detrended_returns_df.dropna(axis=0, how='any')
    # save pickle file detrended_returns_df to current directory
    with open('detrended_returns_df.pickle', 'wb') as f:
        pickle.dump(detrended_returns_df, f)
    # Save to csv file detrended_returns_df to current directory
    detrended_returns_df.to_csv('detrended_returns_df.csv', index=True)

    # Calculate the covariance matrix of the detrended returns
    covariance_matrix = detrended_returns_df.cov()

    # Correlation matrix
    correlation_matrix = detrended_returns_df.corr()



pass  # Press Ctrl+8 to toggle the breakpoint.
