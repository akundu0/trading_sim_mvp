# Trading Simulator

A quantitative backtesting and risk analysis platform that combines **SMA crossover strategy execution**, **Monte Carlo price forecasting**, and **Black-Scholes options pricing** into an interactive dashboard powered by real-time market data.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit)
![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **SMA Crossover Backtesting** — Configurable short/long moving average windows with buy/sell signal generation and trade execution
- **Buy-and-Hold Benchmark** — Strategy equity curve plotted against a passive benchmark for direct comparison
- **Transaction Cost Modeling** — Adjustable round-trip cost per trade for realistic P&L
- **Monte Carlo Simulation (GBM)** — Geometric Brownian Motion price forecasting with VaR and CVaR risk metrics
- **Black-Scholes Options Pricing** — European call/put valuation via the mibian library with annualized volatility
- **Performance Metrics** — Sharpe Ratio, Max Drawdown, Win Rate, and per-trade return statistics
- **Interactive Dashboard** — Streamlit UI with Plotly charts; all parameters adjustable in real time
- **CLI Interface** — Full-featured command-line tool with `argparse` for scripted/headless analysis

## Architecture

```
trading_sim_mvp/
├── app_ui.py                    # Streamlit dashboard (thin presentation layer)
├── app.py                       # CLI entry point
├── src/
│   ├── data/
│   │   └── fetcher.py           # Market data fetching, validation, schema checks
│   ├── strategies/
│   │   └── sma_crossover.py     # SMA signal generation + backtest engine
│   ├── risk/
│   │   ├── monte_carlo.py       # GBM Monte Carlo simulation with VaR/CVaR
│   │   └── options.py           # Black-Scholes pricing wrapper
│   └── metrics/
│       └── performance.py       # Sharpe, drawdown, win rate, trade stats
├── tests/
│   ├── test_data_fetcher.py     # Data validation & schema tests
│   ├── test_strategies.py       # Strategy signal & backtest tests
│   ├── test_monte_carlo.py      # MC simulation correctness & risk metric tests
│   ├── test_metrics.py          # Performance calculation tests
│   └── test_options.py          # Options pricing tests
├── .github/workflows/ci.yml     # GitHub Actions CI (Python 3.11 & 3.12)
├── requirements.txt
└── .gitignore
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ |
| **Data** | yfinance (Yahoo Finance API), pandas |
| **Quantitative** | NumPy, SciPy, mibian (Black-Scholes) |
| **Visualization** | Plotly, Streamlit |
| **Testing** | pytest (38 unit tests) |
| **CI/CD** | GitHub Actions |

## Quick Start

### Prerequisites

- Python 3.11 or higher
- pip

### Installation

```bash
git clone https://github.com/<your-username>/trading_sim_mvp.git
cd trading_sim_mvp
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app_ui.py
```

### Run via CLI

```bash
# Default: AAPL, last 5 years, $10k capital
python app.py

# Custom parameters
python app.py --ticker MSFT --sma-short 10 --sma-long 50 --capital 50000

# Full options
python app.py --ticker TSLA --start 2022-01-01 --end 2024-01-01 \
              --capital 25000 --txn-cost 0.002 --mc-runs 500 --mc-days 126
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Dashboard

<img width="1437" height="771" alt="Trading Simulator Dashboard" src="https://github.com/user-attachments/assets/a7f2daa5-ac17-472c-9ce6-e293900287f9" />

### Demo

![Trading Simulator Demo](https://github.com/user-attachments/assets/845e57e2-9237-41f7-94e2-871caa1ba3a3)

## Key Metrics Explained

| Metric | Description |
|--------|-------------|
| **Sharpe Ratio** | Risk-adjusted return (annualized). Higher = better return per unit of risk. |
| **Max Drawdown** | Largest peak-to-trough decline. Measures worst-case loss scenario. |
| **VaR (95%)** | Value at Risk — the 5th percentile return from Monte Carlo paths. |
| **CVaR (95%)** | Conditional VaR — expected loss beyond the VaR threshold (tail risk). |
| **Win Rate** | Percentage of trades that were profitable. |

## Roadmap

- [ ] Additional strategies (RSI, MACD, Bollinger Bands)
- [ ] Multi-asset portfolio backtesting
- [ ] Database persistence for trade history (PostgreSQL/SQLite)
- [ ] Options Greeks visualization (Delta, Gamma, Theta, Vega)
- [ ] Websocket-based real-time price streaming
- [ ] Strategy optimization via grid search / Bayesian methods
