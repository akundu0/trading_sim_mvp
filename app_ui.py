# app_ui.py — Trading Simulator with Backtest & Trade Markers
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📈 Trading Simulator — SMA Crossover Backtest")

ticker = st.text_input("Ticker", "AAPL").upper()
start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

@st.cache_data
def get_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, group_by="column", auto_adjust=False)

data = get_data(ticker, start_date, end_date)

# --- Flatten columns if MultiIndex ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = ["_".join([str(c) for c in col if c]) for col in data.columns]

# --- Get Close column ---
close_col = [c for c in data.columns if "close" in c.lower()][0]
data["SMA_5"] = data[close_col].rolling(window=5).mean()
data["SMA_10"] = data[close_col].rolling(window=10).mean()

# --- Generate signals ---
data["Signal"] = 0
data.loc[data["SMA_5"] > data["SMA_10"], "Signal"] = 1
data.loc[data["SMA_5"] < data["SMA_10"], "Signal"] = -1
data["Position"] = data["Signal"].diff()

# --- Backtest ---
cash = 10000
position = 0
trade_log = []
portfolio_values = []

for date, row in data.iterrows():
    price = row[close_col]

    # Buy
    if row["Position"] == 2 or row["Position"] == 1:  # crossover up
        position = cash / price
        cash = 0
        trade_log.append({"Date": date, "Action": "BUY", "Price": price, "Value": position * price})

    # Sell
    elif row["Position"] == -2 or row["Position"] == -1:  # crossover down
        cash = position * price
        position = 0
        trade_log.append({"Date": date, "Action": "SELL", "Price": price, "Value": cash})

    portfolio_values.append(cash + position * price)

data["Portfolio"] = portfolio_values
trades_df = pd.DataFrame(trade_log)

# --- Metrics ---
total_return = (data["Portfolio"].iloc[-1] - 10000) / 10000 * 100
returns = data["Portfolio"].pct_change().dropna()
sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() != 0 else 0
win_rate = (
    (trades_df[trades_df["Action"] == "SELL"]["Value"].diff().dropna() > 0).mean() * 100
    if not trades_df.empty else 0
)

st.metric("💰 Final Portfolio Value", f"${data['Portfolio'].iloc[-1]:,.2f}")
st.metric("📈 Total Return", f"{total_return:.2f}%")
st.metric("⚡ Sharpe Ratio", f"{sharpe:.2f}")
st.metric("🏆 Win Rate", f"{win_rate:.1f}%")

# --- Interactive chart with trade markers ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index, y=data[close_col],
    mode="lines", name="Close", line=dict(color="blue")
))
fig.add_trace(go.Scatter(
    x=data.index, y=data["SMA_5"],
    mode="lines", name="SMA 5", line=dict(color="orange")
))
fig.add_trace(go.Scatter(
    x=data.index, y=data["SMA_10"],
    mode="lines", name="SMA 10", line=dict(color="green")
))

# Buy markers
buys = trades_df[trades_df["Action"] == "BUY"]
fig.add_trace(go.Scatter(
    x=buys["Date"], y=buys["Price"], mode="markers",
    marker=dict(symbol="triangle-up", color="green", size=10),
    name="Buy Signal"
))

# Sell markers
sells = trades_df[trades_df["Action"] == "SELL"]
fig.add_trace(go.Scatter(
    x=sells["Date"], y=sells["Price"], mode="markers",
    marker=dict(symbol="triangle-down", color="red", size=10),
    name="Sell Signal"
))

fig.update_layout(title=f"{ticker} — SMA Crossover Strategy",
                  xaxis_title="Date", yaxis_title="Price",
                  template="plotly_white")

st.plotly_chart(fig, use_container_width=True)

# --- Trade Log Table ---
st.subheader("Trade Log")
st.dataframe(trades_df)
