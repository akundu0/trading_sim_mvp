"""SMA crossover trading strategy with configurable transaction costs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single executed trade."""

    date: pd.Timestamp
    action: str          # "BUY" or "SELL"
    price: float
    shares: float = 0.0
    profit_pct: Optional[float] = None  # Populated on SELL


@dataclass
class BacktestResult:
    """Container for all outputs of a backtest run."""

    portfolio_series: pd.Series
    trades: List[Trade]
    final_value: float
    total_return_pct: float
    benchmark_series: pd.Series  # Buy-and-hold comparison


def add_sma_signals(
    df: pd.DataFrame,
    close_col: str,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.DataFrame:
    """Compute SMA indicators and generate crossover signals.

    Adds the following columns to *df* (in-place):

    - ``SMA_short`` / ``SMA_long`` — rolling means
    - ``Signal`` — 1 when short SMA > long SMA, else 0
    - ``Trade`` — +1 on buy crossover, −1 on sell crossover

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the column specified by *close_col*.
    close_col : str
        Name of the close-price column.
    short_window, long_window : int
        Rolling-window sizes.  *short_window* **must** be less than *long_window*.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with signal columns appended.
    """
    if short_window >= long_window:
        logger.warning(
            "short_window (%d) >= long_window (%d); clamping short to %d",
            short_window,
            long_window,
            long_window - 1,
        )
        short_window = max(3, long_window - 1)

    df["SMA_short"] = df[close_col].rolling(window=short_window).mean()
    df["SMA_long"] = df[close_col].rolling(window=long_window).mean()
    df["Signal"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
    df["Trade"] = df["Signal"].diff().fillna(0).astype(int)
    return df


def run_backtest(
    df: pd.DataFrame,
    close_col: str,
    initial_capital: float = 10_000.0,
    transaction_cost_pct: float = 0.0,
) -> BacktestResult:
    """Execute the SMA-crossover backtest on a prepared DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have ``Trade`` and *close_col* columns (see
        :func:`add_sma_signals`).
    close_col : str
        Name of the close-price column.
    initial_capital : float
        Starting cash.
    transaction_cost_pct : float
        Round-trip cost as a fraction (e.g. 0.001 = 0.1 %).

    Returns
    -------
    BacktestResult
    """
    cash: float = float(initial_capital)
    shares: float = 0.0
    last_buy_price: Optional[float] = None
    trades: List[Trade] = []
    portfolio_values: List[float] = []

    # --- benchmark: buy-and-hold ---
    bh_shares = initial_capital / float(df[close_col].iloc[0])
    benchmark_values: List[float] = []

    for idx, row in df.iterrows():
        price = float(row[close_col])
        signal = int(row["Trade"])

        # Buy
        if signal == 1 and shares == 0 and cash > 0:
            cost = cash * transaction_cost_pct
            effective_cash = cash - cost
            shares = effective_cash / price
            cash = 0.0
            last_buy_price = price
            trades.append(Trade(date=idx, action="BUY", price=price, shares=shares))

        # Sell
        elif signal == -1 and shares > 0:
            proceeds = shares * price
            cost = proceeds * transaction_cost_pct
            cash = proceeds - cost
            profit_pct = (price - last_buy_price) / last_buy_price if last_buy_price else 0.0
            trades.append(
                Trade(date=idx, action="SELL", price=price, shares=shares, profit_pct=profit_pct)
            )
            shares = 0.0
            last_buy_price = None

        portfolio_values.append(cash + shares * price)
        benchmark_values.append(bh_shares * price)

    final_value = portfolio_values[-1] if portfolio_values else initial_capital
    total_return_pct = (final_value - initial_capital) / initial_capital * 100

    return BacktestResult(
        portfolio_series=pd.Series(portfolio_values, index=df.index),
        trades=trades,
        final_value=final_value,
        total_return_pct=total_return_pct,
        benchmark_series=pd.Series(benchmark_values, index=df.index),
    )
