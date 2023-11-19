from matplotlib.pylab import LinAlgError
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred

fred = Fred(api_key='aeca492c1d6b24a773fe1fb915779b96')

# Define start and end dates
start_date = '2010-01-01'
end_date = '2023-01-01'

# Fetching financial market data
vix_data = yf.download('^VIX', start=start_date, end=end_date)['Close']
sp500_data = yf.download('^GSPC', start=start_date, end=end_date)['Close']
treasury_yield_data = yf.download('^TNX', start=start_date, end=end_date)['Close']
gold_prices_data = yf.download('GC=F', start=start_date, end=end_date)['Close']
oil_prices_data = yf.download('CL=F', start=start_date, end=end_date)['Close']
usd_index_data = yf.download('DX-Y.NYB', start=start_date, end=end_date)['Close']

# Fetching economic data from FRED
cpi_data = fred.get_series('CPIAUCSL', observation_start=start_date, observation_end=end_date)
unemployment_data = fred.get_series('UNRATE', observation_start=start_date, observation_end=end_date)
gdp_data = fred.get_series('GDP', observation_start=start_date, observation_end=end_date)

# Combining all data into a single DataFrame
financial_indicators = pd.DataFrame({
    'VIX': vix_data,
    'SP500': sp500_data,
    'Treasury_Yield': treasury_yield_data,
    'Gold_Prices': gold_prices_data,
    'Oil_Prices': oil_prices_data,
    'USD_Index': usd_index_data,
    # Add additional economic indicators here after fetching
    "CPI": cpi_data,
    "UnempRate": unemployment_data,
    "GDP": gdp_data
})

# Fill missing values
financial_indicators.fillna(method='ffill', inplace=True)

# Check the DataFrame
print(financial_indicators.head())

# Calculating the correlation matrix
correlation_matrix = financial_indicators.corr()
print(correlation_matrix)

# Plotting the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Financial Indicators')
plt.show()

# modeling