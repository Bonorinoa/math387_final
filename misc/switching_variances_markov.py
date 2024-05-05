# Get the federal funds rate data
from statsmodels.tsa.regime_switching.tests.test_markov_regression import areturns
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

dta_areturns = pd.Series(
    areturns, index=pd.date_range("2004-05-04", "2014-5-03", freq="W")
)

# Plot and display the data
dta_areturns.plot(title="S&P 500 returns", figsize=(12, 3))
plt.show()

# the model is y_t = mu_{s_t} + y_{t-1}*\beta_{s_t} + \epsilon_t, where s_t is the regime at time t and epsilon_t is a Gaussian white noise process with variance sigma_{s_t}^2.
# The variance of the error term is allowed to switch between two regimes.
# The following method implements MLE to estimate the parameters of the model.
# p_{00}, p_{10}, mu_0, mu_1, beta_0, beta_1, sigma2_0, sigma2_1

# The application is to absolute returns on stocks, where the data can be found at https://www.stata-press.com/data/r14/snp500.

# Fit the model
mod_areturns = sm.tsa.MarkovRegression(
    dta_areturns.iloc[1:],
    k_regimes=2,
    exog=dta_areturns.iloc[:-1],
    switching_variance=True,
)
res_areturns = mod_areturns.fit()

print(res_areturns.summary())

# The first regime is a low-variance regime and the second regime is a high-variance regime.
# Below we plot the probabilities of being in the low-variance regime.
# Between 2008 and 2012 there does not appear to be a clear indication of one regime guiding the economy.

res_areturns.smoothed_marginal_probabilities[0].plot(
    title="Probability of being in a low-variance regime", figsize=(12, 3)
)

plt.show()
