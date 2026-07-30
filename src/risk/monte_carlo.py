"""Monte Carlo price simulation using Geometric Brownian Motion (GBM)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


@dataclass
class MonteCarloResult:
    """Container for Monte Carlo simulation outputs."""

    paths: np.ndarray          # shape (runs, days+1)
    median_path: np.ndarray    # shape (days+1,)
    mean_final: float
    var_95: float              # 5th-percentile loss from current price
    cvar_95: float             # expected loss beyond VaR (Conditional VaR)
    percentile_5: float        # 5th-percentile final price
    percentile_95: float       # 95th-percentile final price


def compute_log_return_params(close_series: pd.Series) -> tuple[float, float]:
    """Compute daily log-return mean (mu) and std (sigma).

    Parameters
    ----------
    close_series : pd.Series
        Series of close prices.

    Returns
    -------
    tuple[float, float]
        ``(mu, sigma)`` of daily log returns.
    """
    log_returns = np.log(1 + close_series.pct_change().dropna())
    mu = float(log_returns.mean()) if not log_returns.empty else 0.0
    sigma = float(log_returns.std()) if not log_returns.empty else 0.0
    return mu, sigma


def simulate(
    last_price: float,
    mu: float,
    sigma: float,
    days: int = 252,
    runs: int = 200,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation using Geometric Brownian Motion.

    Each path is generated as:

        S(t+1) = S(t) * exp((mu - 0.5*sigma^2) + sigma * Z)

    where *Z ~ N(0, 1)*.

    Parameters
    ----------
    last_price : float
        Starting price for all paths.
    mu, sigma : float
        Daily log-return mean and standard deviation.
    days : int
        Forecast horizon in trading days.
    runs : int
        Number of simulation paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    MonteCarloResult
    """
    rng = np.random.default_rng(seed)
    drift = mu - 0.5 * sigma**2
    shocks = rng.normal(loc=drift, scale=sigma, size=(runs, days))

    # Cumulative product via log-space for numerical stability
    log_returns = np.cumsum(shocks, axis=1)
    price_paths = last_price * np.exp(
        np.concatenate([np.zeros((runs, 1)), log_returns], axis=1)
    )

    final_prices = price_paths[:, -1]
    median_path = np.median(price_paths, axis=0)

    # Risk metrics
    returns = (final_prices - last_price) / last_price
    var_95 = float(np.percentile(returns, 5))          # 5th percentile return
    cvar_95 = float(returns[returns <= var_95].mean()) if np.any(returns <= var_95) else var_95

    return MonteCarloResult(
        paths=price_paths,
        median_path=median_path,
        mean_final=float(final_prices.mean()),
        var_95=var_95,
        cvar_95=cvar_95,
        percentile_5=float(np.percentile(final_prices, 5)),
        percentile_95=float(np.percentile(final_prices, 95)),
    )
