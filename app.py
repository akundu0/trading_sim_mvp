import yfinance as yf
import pandas as pd
import numpy as np
import mibian
import matplotlib.pyplot as plt

TICKER = "AAPL"
PERIOD = "1mo"
INTERVAL = "1d"

print("Fetching AAPL data...")
data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=False, group_by='column')

if isinstance(data.columns, pd.MultiIndex):
    data.columns = ['_'.join([str(c) for c in col if c]).strip() for col in data.columns.values]

# Calculate SMAs
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()

cash = 10000
shares = 0
position = None  # "long" or None

print("\nStarting simulation...\n")

for i in range(len(data)):
    sma5 = float(data['SMA_5'].iloc[i])
    sma10 = float(data['SMA_10'].iloc[i])
    close_price = float(data['Close'].iloc[i])
    date = data.index[i].date()

    if np.isnan(sma5) or np.isnan(sma10):
        continue

    # Simple crossover strategy
    if sma5 > sma10 and position is None:
        shares = cash / close_price
        cash = 0
        position = "long"
        print(f"[{date}] BUY at ${close_price:.2f}")

    elif sma5 < sma10 and position == "long":
        cash = shares * close_price
        shares = 0
        position = None
        print(f"[{date}] SELL at ${close_price:.2f}")

# Close final position if open
if position == "long":
    final_price = float(data['Close'].iloc[-1])
    cash = shares * final_price
    shares = 0
    print(f"[{data.index[-1].date()}] FINAL SELL at ${final_price:.2f}")


print("\nSimulation complete.")
print(f"Final Portfolio Value: ${cash:.2f}")

# # # After simulation, calculating Sharpe Ratio
data['Portfolio_Value'] = np.nan
portfolio_value = cash
if position == "long":
    portfolio_value = shares * final_price
else:
    portfolio_value = cash
data.iloc[-1, data.columns.get_loc('Portfolio_Value')] = portfolio_value

# Example: calculate daily portfolio returns
data['Portfolio_Return'] = data['Portfolio_Value'].pct_change()

# Remove NaNs
returns = data['Portfolio_Return'].dropna()

# Risk-free rate (0 for simplicity, adjust if needed)
risk_free_rate = 0.0

# Sharpe Ratio: (mean return - risk-free rate) / std deviation of returns
sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()

print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

### Options pricing

# Parameters: underlying price, strike price, interest rate (%), days until expiration, volatility (%)
c = mibian.BS([final_price, 150, 2, 30], volatility=20)

print(f"Call Price: {c.callPrice:.2f}")
print(f"Put Price: {c.putPrice:.2f}")

### Monte Carlo Forecast

# Ensure final_price is a float
final_price = float(data['Close'].iloc[-1])

# Prepare Monte Carlo params
returns = data['Close'].pct_change().dropna()
mu = float(returns.mean())
sigma = float(returns.std())
last_price = final_price

plt.figure(figsize=(10, 6))

num_simulations = 100
num_days = 30

for _ in range(num_simulations):
    price_series = [last_price]
    for _ in range(num_days):
        price_series.append(price_series[-1] * (1 + np.random.normal(mu, sigma)))
    plt.plot(range(len(price_series)), price_series, linewidth=1)

plt.title("Monte Carlo Simulation")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
