# app_ui.py — dynamic, fixed version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mibian

st.set_page_config(layout="wide", page_title="Trading + Monte Carlo + Options Dashboard")
st.title("📈 Trading Simulator")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "AAPL").upper()
    sma_short = st.slider("Short SMA window", 3, 50, 5)
    sma_long = st.slider("Long SMA window", 5, 200, 20)
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    initial_capital = st.number_input("Initial Capital ($)", value=10000, step=1000)
    show_mc = st.checkbox("Show Monte Carlo", value=True)
    mc_runs = st.number_input("Monte Carlo runs", min_value=10, max_value=1000, value=200, step=10)
    mc_days = st.number_input("Monte Carlo horizon (days)", min_value=10, max_value=252, value=252, step=10)
    st.markdown("---")
    st.header("Options (mibian)")
    option_strike = st.number_input("Strike Price ($)", value=0.0, step=1.0, help="0 = ATM")
    option_days = st.number_input("Days to expiration", min_value=1, max_value=365, value=30)
    rf_rate = st.number_input("Risk-free rate (%)", value=2.0, step=0.1)

# -------------------------
# Fetch only raw price data (cached)
# -------------------------
@st.cache_data
def fetch_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if df.empty:
        return df
    # Flatten multiindex if it somehow appears
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c not in (None, "")]).strip() for col in df.columns]
    return df

price_data = fetch_prices(ticker, start_date, end_date)
if price_data is None or price_data.empty:
    st.error("No price data found for that ticker/date range.")
    st.stop()

# make a working copy (so cached raw remains unchanged)
data = price_data.copy()

# detect close column robustly
close_col = None
for c in data.columns:
    if "close" in str(c).lower():
        close_col = c
        break
if close_col is None:
    st.error(f"Could not find a Close column. Columns: {list(data.columns)}")
    st.stop()

# ensure numeric and drop NaNs
data[close_col] = pd.to_numeric(data[close_col], errors="coerce")
data.dropna(subset=[close_col], inplace=True)

# -------------------------
# Indicators (recomputed on every change)
# -------------------------
short = int(sma_short)
long = int(sma_long)
if short >= long:
    st.warning("Short SMA should be smaller than Long SMA. Adjusting short < long.")
    short = max(3, long - 1)

data["SMA_short"] = data[close_col].rolling(window=short).mean()
data["SMA_long"] = data[close_col].rolling(window=long).mean()

# Signals: 1 when short > long, 0 otherwise
data["Signal"] = (data["SMA_short"] > data["SMA_long"]).astype(int)
data["Trade"] = data["Signal"].diff().fillna(0).astype(int)  # 1 => buy, -1 => sell

# -------------------------
# Backtest (row-wise — safe and clear)
# -------------------------
cash = float(initial_capital)
position_shares = 0.0
portfolio_values = []
trade_log = []
last_buy_price = None
trade_profits = []

# Iterate rows (row is a Series — scalars are safe)
for idx, row in data.iterrows():
    price = float(row[close_col])
    trade_signal = int(row["Trade"])  # scalar int

    if trade_signal == 1:  # buy
        if position_shares == 0 and cash > 0:
            position_shares = cash / price
            cash = 0.0
            last_buy_price = price
            trade_log.append({"Date": idx, "Action": "BUY", "Price": price})
    elif trade_signal == -1:  # sell
        if position_shares > 0:
            cash = position_shares * price
            # record profit for the trade
            if last_buy_price is not None:
                profit_pct = (price - last_buy_price) / last_buy_price
                trade_profits.append(profit_pct)
                last_buy_price = None
            trade_log.append({"Date": idx, "Action": "SELL", "Price": price})
    # update current portfolio value
    portfolio_values.append(cash + position_shares * price)

# If still holding at the end, compute final value (we already appended last value)
portfolio_series = pd.Series(portfolio_values, index=data.index)
data["Portfolio"] = portfolio_series

# -------------------------
# Performance metrics
# -------------------------
final_value = float(portfolio_series.iloc[-1])
total_return_pct = (final_value - initial_capital) / initial_capital * 100
daily_rets = portfolio_series.pct_change().dropna()
sharpe = (np.sqrt(252) * daily_rets.mean() / daily_rets.std()) if daily_rets.std() != 0 else 0.0

# max drawdown
cum = (1 + daily_rets).cumprod()
rolling_max = cum.cummax()
drawdown = (cum - rolling_max) / rolling_max
max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

num_trades = len(trade_profits)
win_rate = float((np.sum(np.array(trade_profits) > 0) / num_trades * 100) if num_trades > 0 else 0.0)

# -------------------------
# Monte Carlo
# -------------------------
def monte_carlo_paths(last_price, mu, sigma, days, runs):
    runs_data = []
    for _ in range(int(runs)):
        path = [last_price]
        for _ in range(int(days)):
            shock = np.random.normal(loc=(mu - 0.5 * sigma ** 2), scale=sigma)
            path.append(path[-1] * np.exp(shock))
        runs_data.append(path)
    return runs_data

last_price = float(data[close_col].iloc[-1])
daily_log_returns = np.log(1 + data[close_col].pct_change().dropna())
mu = float(daily_log_returns.mean()) if not daily_log_returns.empty else 0.0
sigma = float(daily_log_returns.std()) if not daily_log_returns.empty else 0.0
mc_results = monte_carlo_paths(last_price, mu, sigma, int(mc_days), int(mc_runs)) if show_mc else []

# -------------------------
# Options pricing (mibian)
# -------------------------
if option_strike == 0.0:
    strike = round(last_price, 2)
else:
    strike = float(option_strike)

annual_vol_pct = float(sigma * np.sqrt(252) * 100)
try:
    option_bs = mibian.BS([last_price, strike, float(rf_rate), int(option_days)], volatility=annual_vol_pct)
    call_price = option_bs.callPrice
    put_price = option_bs.putPrice
except Exception as e:
    call_price = None
    put_price = None
    st.warning(f"Options pricing failed: {e}")

# -------------------------
# UI: display metrics & charts
# -------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Final Portfolio", f"${final_value:,.2f}")
col2.metric("Total Return", f"{total_return_pct:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
col4.metric("Max Drawdown", f"{max_drawdown:.2%}")

# Price chart with SMA and trade markers
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[close_col], mode="lines", name="Close", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_short"], mode="lines", name=f"SMA {short}", line=dict(color="orange")))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_long"], mode="lines", name=f"SMA {long}", line=dict(color="green")))

trades_df = pd.DataFrame(trade_log)
if not trades_df.empty:
    buys = trades_df[trades_df["Action"] == "BUY"]
    sells = trades_df[trades_df["Action"] == "SELL"]
    if not buys.empty:
        fig.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"], mode="markers",
                                 marker=dict(symbol="triangle-up", color="green", size=10), name="Buy"))
    if not sells.empty:
        fig.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"], mode="markers",
                                 marker=dict(symbol="triangle-down", color="red", size=10), name="Sell"))

fig.update_layout(template="plotly_white", height=500, title=f"{ticker} — SMA crossover ({short}/{long})")
st.plotly_chart(fig, use_container_width=True)

# Portfolio equity curve
st.subheader("Portfolio Equity Curve")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data.index, y=data["Portfolio"], mode="lines", name="Portfolio"))
fig2.update_layout(template="plotly_white", height=350)
st.plotly_chart(fig2, use_container_width=True)

# Monte Carlo plot
if show_mc and mc_results:
    st.subheader(f"Monte Carlo ({mc_runs} runs, {mc_days} days)")
    fig_mc = go.Figure()
    for path in mc_results:
        fig_mc.add_trace(go.Scatter(x=list(range(len(path))), y=path, mode="lines", line=dict(width=1), opacity=0.08, showlegend=False))
    mc_array = np.array(mc_results)
    median_path = np.median(mc_array, axis=0)
    fig_mc.add_trace(go.Scatter(x=list(range(len(median_path))), y=median_path, mode="lines", line=dict(width=2, color="black"), name="Median"))
    fig_mc.update_layout(template="plotly_white", height=400, xaxis_title="Days", yaxis_title="Price")
    st.plotly_chart(fig_mc, use_container_width=True)

# Options
st.subheader("Options (Black-Scholes approx via mibian)")
st.write(f"Underlying: ${last_price:.2f} | Strike: ${strike:.2f} | Days: {int(option_days)} | Vol (ann %): {annual_vol_pct:.2f}")
if call_price is not None:
    st.metric("Call Price (approx)", f"${call_price:.2f}")
    st.metric("Put Price (approx)", f"${put_price:.2f}")
else:
    st.write("Options pricing unavailable for these inputs.")

# Trade log & stats
st.subheader("Trade Log & Stats")
st.write(f"Number of completed trades: {num_trades}")
st.write(f"Win rate (by trade): {win_rate:.1f}%")
if trades_df.empty:
    st.write("No trades occurred for this strategy in the given period.")
else:
    st.dataframe(trades_df)
