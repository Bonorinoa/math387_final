import pandas as pd
import statsmodels.api as sm
import yfinance as yf
import matplotlib.pyplot as plt

vix_data = yf.download('^VIX', start='2020-01-01', end='2023-11-01')

# Plot and display the data
vix_data['Adj Close'].plot(title="VIX", figsize=(12, 3))

plt.show()