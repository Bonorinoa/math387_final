import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Fetch S&P 500 historical data
sp500 = yf.download('^GSPC', start='2000-01-01', end='2023-01-01')

# Preprocess data: Calculate log returns
sp500['Log_Returns'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
sp500.dropna(inplace=True)

# Display the first few rows of the processed data
print(sp500.head())

plt.plot(sp500['Log_Returns'])
#plt.show()

# Setting up the Markov Switching Model
model = sm.tsa.MarkovRegression(sp500['Log_Returns'], k_regimes=2, trend='c', switching_variance=True)
results = model.fit()

# Print the summary of the model's results

#print(results.summary())
# Extract regime probabilities
regime_probabilities = results.smoothed_marginal_probabilities

# Plotting
plt.figure(figsize=(12,6))
plt.plot(regime_probabilities[0], label='Low Volatility Regime')
plt.plot(regime_probabilities[1], label='High Volatility Regime')
plt.title('Regime Probabilities over Time')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
#plt.show()

## Predicting state probabilities
# Assuming 'results' is your fitted MarkovSwitching model
filtered_probs = results.filtered_marginal_probabilities.iloc[-1]
transition_matrix = results.regime_transition

print("Filtered Probabilities of the Last Observation:")
print(filtered_probs)
print("\nTransition Matrix:")
print(transition_matrix)

# we use the trnasition matrix and filtered probabilities for prediction
# Reshape the transition matrix from 3D to 2D
reshaped_transition_matrix = transition_matrix.reshape(2, 2)

# Predict the next state probabilities
next_state_probs = np.dot(reshaped_transition_matrix, filtered_probs)

print("Predicted State Probabilities for the Next Period:")
print(next_state_probs)

# we can use these predicted probabilities for assessing market risk, protfolio analysis, long-term investment strategies,...