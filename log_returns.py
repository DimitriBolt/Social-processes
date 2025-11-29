import glob
import pickle
from sklearn.linear_model import LinearRegression

import numpy as np
import yfinance as yf

import pandas
import pandas as pd
from numpy import ndarray, array
from pandas import DataFrame, Series


class LogReturns:
    # Initializer выполняется перед! основной программой.
    # Private Instance or static Class attribute. Переменные должны начинаться с двух подчеркиваний.
    # Initial static values

    # Здесь не реализован (а должен был) Constructor WithOut parameters
    def __init__(self, symbols_local: list[str], start_date_local: str, end_date_local: str) -> None:
        self.start_date = start_date_local
        self.symbols: list[str] = symbols_local
        self.end_date: str = end_date_local
        pass

    # Methods
    def __private_method(self, assets: list[str], start: str, end: str) -> DataFrame:
        delisted: list[str] = ['STRD', 'STRF', 'STRC', 'BRKRP', 'STRK', 'SNDK', 'CRCL', 'MCHPP', 'AGNCZ', 'SAIL', 'GLXY', 'CRWV'] + ['BRK/A', 'BRK/B'] + ['EMP']
        # bad defined
        delisted: list[str] = delisted + ['ALAB', 'ASTS', 'AUR', 'BE', 'CRDO', 'FTAI', 'HIMS', 'IONQ', 'MP', 'OKLO', 'RGTI', 'RKLB', 'SATS', 'SMCI', 'SMR', 'TEM', 'TTD'] + ['AFRM', 'APP', 'CNC', 'COHR', 'COIN', 'CVNA', 'HOOD', 'JOBY', 'LITE', 'MDB', 'MRVL', 'MSTR', 'PLTR', 'RDDT', 'SYM', 'U', 'VRT', 'VST', 'W']
        failed_downloads = delisted
        assets_without_failed: list[str] = [asset for asset in assets if asset not in failed_downloads]
        # assets_without_failed: list = ['AAPL', 'MSFT', 'GOOGL']
        # n = 100
        # data: Series = yf.download(tickers=assets_without_failed[0:n], start=start, end=end, timeout=20, auto_adjust=True)['Close']
        data: Series = yf.download(tickers=assets_without_failed, start=start, end=end, timeout=20, auto_adjust=True)['Close']
        # Save to pickle file data to current directory
        with open('downloaded_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        print("Downloaded data saved to downloaded_data.pkl")
        # Save to csv file data to current directory
        data.to_csv('downloaded_data.csv', index=True)
        print("Downloaded data saved to downloaded_data.csv")

        log_prices = np.log(data)
        X = np.arange(len(log_prices)).reshape(-1, 1)  # Time index as the independent variable
        y = log_prices  # Logarithmic prices as the dependent variable

        # Fit linear regression: time → all columns
        model = LinearRegression().fit(X, y)

        trend = model.predict(X)
        detrended_log_prices = log_prices - trend
        # Compute returns as differences between current row and previous row of detrended logarithmic prices
        detrended_returns = detrended_log_prices.diff().dropna()
        return detrended_returns


    # Accessor( = getter) methods
    def get_log_returns(self) -> DataFrame:
        # It is used for return outside protected attributes of the object only
        return self.__private_method(self.symbols, self.start_date, self.end_date)


if __name__ == '__main__':
    # open csv file nasdaq_screener_*.csv to pandas dataframe from current directory
    # csv_path = glob.glob("./nasdaq_screener_*.csv")[-1]
    # csv_path = glob.glob("./nasdaq_screener_*.csv")[-2] # nasdaq_screener_100.csv
    csv_path = glob.glob("./nasdaq_screener_*.csv")[-1] # nasdaq_screener_200.csv


    df = pd.read_csv(csv_path)
    symbols: list[str] = df["Symbol"].astype(str).dropna().reset_index(drop=True).tolist()

    log_returns: LogReturns = LogReturns(symbols_local=symbols,
                                         start_date_local='2025-01-01',
                                         end_date_local='2025-10-01')
    detrended_returns_df: DataFrame = log_returns.get_log_returns()

    # Для каждого дня в detrended_returns_df и для каждого столбца в detrended_returns_df посчитать среднее значение по окну за предыдущие 30 дней
    # detrended_returns_df: DataFrame = detrended_returns_df.rolling(window=45).mean().dropna()

    # Save to csv file detrended_returns_df to current directory
    detrended_returns_df.to_csv('detrended_returns_df.csv', index=True)
    # Save to pickle file detrended_returns_df to current directory
    with open('detrended_returns_df.pkl', 'wb') as f:
        pickle.dump(detrended_returns_df, f)
    print("Detrended returns saved to detrended_returns_df.csv and detrended_returns_df.pkl")