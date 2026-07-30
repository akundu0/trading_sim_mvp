"""Portfolio performance metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """Summary statistics for a backtest run."""

    final_value: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    win_rate: float          # Percentage (0–100)
    avg_trade_return: float  # Mean per-trade return


def compute_metrics(
    portfolio_series: pd.Series,
    trade_profits: list[float],
    initial_capital: float,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Derive key performance indicators from a backtest equity curve.

    Parameters
    ----------
    portfolio_series : pd.Series
        Daily portfolio values indexed by date.
    trade_profits : list[float]
        Per-trade profit percentages (e.g. ``0.05`` for a 5 % gain).
    initial_capital : float
        Starting capital.
    risk_free_rate : float
        Annualized risk-free rate as a decimal (default ``0.0``).

    Returns
    -------
    PerformanceMetrics
    """
    final_value = float(portfolio_series.iloc[-1])
    total_return_pct = (final_value - initial_capital) / initial_capital * 100

    # --- Sharpe Ratio (annualized) ---
    daily_returns = portfolio_series.pct_change().dropna()
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = daily_returns - daily_rf
    sharpe = (
        float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / excess.std())
        if excess.std() != 0
        else 0.0
    )

    # --- Maximum Drawdown ---
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    # --- Trade statistics ---
    num_trades = len(trade_profits)
    if num_trades > 0:
        profits = np.array(trade_profits)
        win_rate = float(np.sum(profits > 0) / num_trades * 100)
        avg_trade_return = float(profits.mean())
    else:
        win_rate = 0.0
        avg_trade_return = 0.0

    return PerformanceMetrics(
        final_value=final_value,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_trade_return=avg_trade_return,
    )
