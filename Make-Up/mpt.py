import numpy as np
import pandas as pd
import yfinance as yf
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

# Define the assets and the period for data collection
assets: list = ['AAPL', 'GOOGL', 'MSFT']
start_date = '2024-11-15'
end_date = '2024-11-29'

# Fetch historical adjusted closing prices
data: Series = yf.download(assets, start=start_date, end=end_date, auto_adjust=True)['Close']

# Initialize dictionaries to store detrended log-prices and returns
detrended_log_prices: dict = {}

detrended_returns: dict = {}

# Detrend each asset's logarithmic price series using linear regression
for asset in assets:
    # Compute logarithmic prices
    log_prices = np.log(data[asset].values).reshape(-1, 1)

    # Prepare data for linear regression
    X = np.arange(len(log_prices)).reshape(-1, 1) # Time index as the independent variable
    y = log_prices # Logarithmic prices as the dependent variable

    # Fit the linear regression model
    model = LinearRegression().fit(X, y)
    trend = model.predict(X)

    # Calculate the detrended logarithmic prices (residuals)
    detrended_log_prices[asset]: dict = log_prices - trend

    # Compute returns as differences between detrended logarithmic prices
    detrended_returns[asset]: dict = detrended_log_prices[asset][1:] - detrended_log_prices[asset][:-1]
    pass

# Convert the detrended returns dictionary to a DataFrame
detrended_returns_df = pd.DataFrame({asset: detrended_returns[asset].flatten() for asset in assets})

## Calculate the expected return vector (mean of detrended returns)
expected_returns = detrended_returns_df.mean()

# Calculate the covariance matrix of the detrended returns
covariance_matrix = detrended_returns_df.cov()

# Annualize the returns and covariance matrix
trading_days = 252 # Number of trading days in a year
annualized_returns = trading_days * expected_returns
annualized_covariance_matrix = (trading_days ** 2) * covariance_matrix

# Correlation matrix
correlation_matrix = detrended_returns_df.corr()

# Print results for verification
print("Annualized Return Vector:")
print(annualized_returns)
print("\nAnnualized Covariance Matrix:")
print(annualized_covariance_matrix)

# Round for printing in LaTeX
annualized_returns = np.round(annualized_returns, 6)
annualized_covariance_matrix = np.round(annualized_covariance_matrix,6)
correlation_matrix = np.round(correlation_matrix, 6)