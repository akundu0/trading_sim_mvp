#!/usr/bin/env python3
"""CLI entry point for the Trading Simulator.

Usage examples:
    python app.py                         # defaults: AAPL, last 5 years
    python app.py --ticker MSFT --sma-short 10 --sma-long 50
    python app.py --ticker TSLA --start 2022-01-01 --end 2024-01-01 --capital 50000
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from src.data.fetcher import fetch_prices, detect_close_column
from src.strategies.sma_crossover import add_sma_signals, run_backtest
from src.risk.monte_carlo import compute_log_return_params, simulate as mc_simulate
from src.risk.options import price_options
from src.metrics.performance import compute_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an SMA-crossover backtest with Monte Carlo risk analysis."
    )
    parser.add_argument("--ticker", default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", default=str(date.today() - timedelta(days=5 * 365)),
                        help="Start date YYYY-MM-DD (default: 5 years ago)")
    parser.add_argument("--end", default=str(date.today()),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--sma-short", type=int, default=5, help="Short SMA window (default: 5)")
    parser.add_argument("--sma-long", type=int, default=20, help="Long SMA window (default: 20)")
    parser.add_argument("--capital", type=float, default=10_000, help="Initial capital (default: 10000)")
    parser.add_argument("--txn-cost", type=float, default=0.001,
                        help="Transaction cost as fraction (default: 0.001 = 0.1%%)")
    parser.add_argument("--mc-runs", type=int, default=200, help="Monte Carlo paths (default: 200)")
    parser.add_argument("--mc-days", type=int, default=252, help="Monte Carlo horizon in days (default: 252)")
    parser.add_argument("--strike", type=float, default=0.0, help="Option strike; 0 = ATM (default: 0)")
    parser.add_argument("--rf-rate", type=float, default=2.0, help="Risk-free rate %% (default: 2.0)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Data ---
    print(f"Fetching {args.ticker} data ({args.start} → {args.end})...")
    try:
        data = fetch_prices(args.ticker, args.start, args.end)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    close_col = detect_close_column(data)

    # --- Strategy ---
    data = add_sma_signals(data, close_col, short_window=args.sma_short, long_window=args.sma_long)
    result = run_backtest(data, close_col, initial_capital=args.capital, transaction_cost_pct=args.txn_cost)

    # --- Metrics ---
    trade_profits = [t.profit_pct for t in result.trades if t.profit_pct is not None]
    metrics = compute_metrics(result.portfolio_series, trade_profits, args.capital)

    # --- Monte Carlo ---
    last_price = float(data[close_col].iloc[-1])
    mu, sigma = compute_log_return_params(data[close_col])
    mc = mc_simulate(last_price, mu, sigma, days=args.mc_days, runs=args.mc_runs)

    # --- Options ---
    opt = price_options(last_price, args.strike, args.rf_rate, 30, sigma)

    # --- Report ---
    print(f"\n{'=' * 50}")
    print(f"  {args.ticker} — SMA Crossover ({args.sma_short}/{args.sma_long})")
    print(f"{'=' * 50}")
    print(f"  Period:          {args.start} → {args.end}")
    print(f"  Initial Capital: ${args.capital:,.2f}")
    print(f"  Final Value:     ${metrics.final_value:,.2f}")
    print(f"  Total Return:    {metrics.total_return_pct:+.2f}%")
    print(f"  Sharpe Ratio:    {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:    {metrics.max_drawdown:.2%}")
    print(f"  Trades:          {metrics.num_trades}")
    print(f"  Win Rate:        {metrics.win_rate:.1f}%")
    print(f"  Avg Trade Ret:   {metrics.avg_trade_return:+.2%}")

    print(f"\n{'─' * 50}")
    print(f"  Monte Carlo ({args.mc_runs} runs, {args.mc_days} days)")
    print(f"{'─' * 50}")
    print(f"  VaR  (95%):       {mc.var_95:+.2%}")
    print(f"  CVaR (95%):       {mc.cvar_95:+.2%}")
    print(f"  5th pctl price:   ${mc.percentile_5:,.2f}")
    print(f"  95th pctl price:  ${mc.percentile_95:,.2f}")

    if opt:
        print(f"\n{'─' * 50}")
        print(f"  Options (Black-Scholes)")
        print(f"{'─' * 50}")
        print(f"  Call Price:  ${opt.call_price:.2f}")
        print(f"  Put Price:   ${opt.put_price:.2f}")
        print(f"  Ann. Vol:    {opt.annual_vol_pct:.1f}%")

    print(f"\n{'─' * 50}")
    print("  Trade Log")
    print(f"{'─' * 50}")
    if result.trades:
        for t in result.trades:
            pnl = f"  ({t.profit_pct:+.2%})" if t.profit_pct is not None else ""
            print(f"  [{t.date.date()}] {t.action:4s} @ ${t.price:.2f}{pnl}")
    else:
        print("  No trades triggered.")

    print()


if __name__ == "__main__":
    main()
