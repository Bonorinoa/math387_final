import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Fetching VIX data
vix = yf.download('^VIX', start='2010-01-01', end='2023-01-01')['Close']

# Defining thresholds for volatility regimes
low_threshold = vix.quantile(0.33)  # Lower 33% of VIX values
high_threshold = vix.quantile(0.66)  # Higher 33% of VIX values

# Classifying into regimes
vix_regimes = pd.cut(vix, bins=[0, low_threshold, high_threshold, np.inf], 
                     labels=['Low', 'Medium', 'High'])

# Visualizing the regimes
plt.figure(figsize=(12, 6))
plt.plot(vix, label='VIX')
plt.fill_between(vix.index, 0, vix, where=vix_regimes=='Low', color='green', alpha=0.3, label='Low Volatility')
plt.fill_between(vix.index, 0, vix, where=vix_regimes=='Medium', color='orange', alpha=0.3, label='Medium Volatility')
plt.fill_between(vix.index, 0, vix, where=vix_regimes=='High', color='red', alpha=0.3, label='High Volatility')
plt.legend()
plt.title('VIX with Volatility Regimes')
plt.xlabel('Date')
plt.ylabel('VIX')
plt.show()

# Fetching additional financial data
sp500 = yf.download('^GSPC', start='2010-01-01', end='2023-01-01')['Close']
treasury_yield = yf.download('^TNX', start='2010-01-01', end='2023-01-01')['Close']
gold_prices = yf.download('GC=F', start='2010-01-01', end='2023-01-01')['Close']

# Combining all data into a single DataFrame
financial_data = pd.DataFrame({
    'VIX': vix,
    'SP500': sp500,
    'Treasury_Yield': treasury_yield,
    'Gold_Prices': gold_prices
})

# Filling any missing values
financial_data = financial_data.fillna(method='ffill')

# Checking the head of the DataFrame
print(financial_data.head())

import seaborn as sns

# Calculating the correlation matrix
correlation_matrix = financial_data.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Financial Indicators')
plt.show()

# Normalizing data for comparison
normalized_data = (financial_data - financial_data.min()) / (financial_data.max() - financial_data.min())

# Plotting
plt.figure(figsize=(14, 8))
for column in normalized_data.columns:
    plt.plot(normalized_data.index, normalized_data[column], label=column)
plt.legend()
plt.title('Normalized Financial Indicators Over Time')
plt.xlabel('Year')
plt.ylabel('Normalized Value')
plt.show()
