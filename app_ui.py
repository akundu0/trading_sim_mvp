"""Streamlit UI — thin presentation layer that delegates to src/ modules."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data.fetcher import fetch_prices, detect_close_column
from src.strategies.sma_crossover import add_sma_signals, run_backtest
from src.risk.monte_carlo import compute_log_return_params, simulate as mc_simulate
from src.risk.options import price_options
from src.metrics.performance import compute_metrics

# -------------------------
# Page config
# -------------------------
st.set_page_config(layout="wide", page_title="Trading Simulator — Backtest & Risk Dashboard")
st.title("Trading Simulator")
st.caption("SMA crossover backtesting  ·  Monte Carlo risk analysis  ·  Black-Scholes options pricing")

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Strategy")
    ticker = st.text_input("Ticker", "AAPL").upper()
    sma_short = st.slider("Short SMA window", 3, 50, 5)
    sma_long = st.slider("Long SMA window", 5, 200, 20)
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))
    initial_capital = st.number_input("Initial Capital ($)", value=10_000, step=1_000)
    txn_cost = st.number_input(
        "Transaction Cost (%)", value=0.1, step=0.05, min_value=0.0, max_value=5.0,
        help="Round-trip cost applied on each buy/sell (e.g. 0.1 = 0.1%)",
    )

    st.markdown("---")
    st.header("Monte Carlo")
    show_mc = st.checkbox("Show Monte Carlo", value=True)
    mc_runs = st.number_input("Simulation runs", min_value=10, max_value=1_000, value=200, step=10)
    mc_days = st.number_input("Horizon (trading days)", min_value=10, max_value=252, value=252, step=10)

    st.markdown("---")
    st.header("Options Pricing")
    option_strike = st.number_input("Strike Price ($)", value=0.0, step=1.0, help="0 = ATM")
    option_days = st.number_input("Days to expiration", min_value=1, max_value=365, value=30)
    rf_rate = st.number_input("Risk-free rate (%)", value=2.0, step=0.1)

# -------------------------
# Data fetch (cached)
# -------------------------
@st.cache_data(show_spinner="Fetching market data...")
def _fetch(tick: str, s, e):
    return fetch_prices(tick, s, e)

try:
    price_data = _fetch(ticker, start_date, end_date)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

data = price_data.copy()

try:
    close_col = detect_close_column(data)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# -------------------------
# Strategy signals & backtest
# -------------------------
short = int(sma_short)
long_ = int(sma_long)
if short >= long_:
    st.warning("Short SMA must be less than Long SMA — auto-adjusting.")
    short = max(3, long_ - 1)

data = add_sma_signals(data, close_col, short_window=short, long_window=long_)
result = run_backtest(data, close_col, initial_capital=float(initial_capital), transaction_cost_pct=txn_cost / 100)

data["Portfolio"] = result.portfolio_series
data["Benchmark"] = result.benchmark_series

# -------------------------
# Metrics
# -------------------------
trade_profits = [t.profit_pct for t in result.trades if t.profit_pct is not None]
metrics = compute_metrics(result.portfolio_series, trade_profits, float(initial_capital))

# -------------------------
# Monte Carlo
# -------------------------
last_price = float(data[close_col].iloc[-1])
mu, sigma = compute_log_return_params(data[close_col])
mc_result = mc_simulate(last_price, mu, sigma, days=int(mc_days), runs=int(mc_runs)) if show_mc else None

# -------------------------
# Options pricing
# -------------------------
opt = price_options(
    underlying=last_price,
    strike=float(option_strike),
    risk_free_rate_pct=float(rf_rate),
    days_to_expiry=int(option_days),
    daily_sigma=sigma,
)

# ═══════════════════════════════════
# UI LAYOUT
# ═══════════════════════════════════

# --- KPI row ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Final Portfolio", f"${metrics.final_value:,.2f}")
col2.metric("Total Return", f"{metrics.total_return_pct:+.2f}%")
col3.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.2f}")
col4.metric("Max Drawdown", f"{metrics.max_drawdown:.2%}")
col5.metric("Win Rate", f"{metrics.win_rate:.1f}%")

# --- Price chart ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[close_col], mode="lines", name="Close", line=dict(color="#1f77b4")))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_short"], mode="lines", name=f"SMA {short}", line=dict(color="#ff7f0e", dash="dot")))
fig.add_trace(go.Scatter(x=data.index, y=data["SMA_long"], mode="lines", name=f"SMA {long_}", line=dict(color="#2ca02c", dash="dot")))

trades_df = pd.DataFrame([{"Date": t.date, "Action": t.action, "Price": t.price} for t in result.trades])
if not trades_df.empty:
    buys = trades_df[trades_df["Action"] == "BUY"]
    sells = trades_df[trades_df["Action"] == "SELL"]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys["Date"], y=buys["Price"], mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=12), name="Buy",
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells["Date"], y=sells["Price"], mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12), name="Sell",
        ))

fig.update_layout(template="plotly_white", height=500, title=f"{ticker} — SMA Crossover ({short}/{long_})")
st.plotly_chart(fig, use_container_width=True)

# --- Equity curve vs benchmark ---
st.subheader("Equity Curve vs. Buy-and-Hold Benchmark")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data.index, y=data["Portfolio"], mode="lines", name="Strategy", line=dict(color="#1f77b4")))
fig2.add_trace(go.Scatter(x=data.index, y=data["Benchmark"], mode="lines", name="Buy & Hold", line=dict(color="#d62728", dash="dash")))
fig2.update_layout(template="plotly_white", height=350, yaxis_title="Portfolio Value ($)")
st.plotly_chart(fig2, use_container_width=True)

# --- Monte Carlo ---
if show_mc and mc_result is not None:
    st.subheader(f"Monte Carlo Simulation ({mc_runs} paths, {mc_days}-day horizon)")

    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
    risk_col1.metric("VaR (95%)", f"{mc_result.var_95:+.2%}")
    risk_col2.metric("CVaR (95%)", f"{mc_result.cvar_95:+.2%}")
    risk_col3.metric("5th Pctl Price", f"${mc_result.percentile_5:,.2f}")
    risk_col4.metric("95th Pctl Price", f"${mc_result.percentile_95:,.2f}")

    fig_mc = go.Figure()
    # Plot a subset of paths for performance (max 200 traces)
    paths_to_plot = mc_result.paths[:min(len(mc_result.paths), 200)]
    for path in paths_to_plot:
        fig_mc.add_trace(go.Scatter(
            x=list(range(len(path))), y=path.tolist(), mode="lines",
            line=dict(width=0.8, color="rgba(31,119,180,0.08)"), showlegend=False,
        ))
    fig_mc.add_trace(go.Scatter(
        x=list(range(len(mc_result.median_path))), y=mc_result.median_path.tolist(),
        mode="lines", line=dict(width=2.5, color="black"), name="Median",
    ))
    fig_mc.update_layout(template="plotly_white", height=420, xaxis_title="Trading Days", yaxis_title="Price ($)")
    st.plotly_chart(fig_mc, use_container_width=True)

# --- Options ---
st.subheader("Options Pricing (Black-Scholes)")
if opt is not None:
    opt_c1, opt_c2, opt_c3 = st.columns(3)
    opt_c1.metric("Call Price", f"${opt.call_price:.2f}")
    opt_c2.metric("Put Price", f"${opt.put_price:.2f}")
    opt_c3.metric("Annualized Vol", f"{opt.annual_vol_pct:.1f}%")
    st.caption(
        f"Underlying: ${last_price:.2f}  ·  Strike: ${opt.strike:.2f}  ·  "
        f"Days to expiry: {opt.days_to_expiry}  ·  Risk-free rate: {opt.risk_free_rate_pct:.1f}%"
    )
else:
    st.warning("Options pricing unavailable for these inputs.")

# --- Trade log ---
st.subheader("Trade Log")
tcol1, tcol2, tcol3 = st.columns(3)
tcol1.metric("Completed Trades", metrics.num_trades)
tcol2.metric("Win Rate", f"{metrics.win_rate:.1f}%")
tcol3.metric("Avg Trade Return", f"{metrics.avg_trade_return:+.2%}" if metrics.num_trades > 0 else "N/A")

if trades_df.empty:
    st.info("No trades were triggered in this date range with the current SMA settings.")
else:
    st.dataframe(trades_df, use_container_width=True)
