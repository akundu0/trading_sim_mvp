"""Data fetching and validation utilities for market data."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
TRADING_DAYS_PER_YEAR = 252


def fetch_prices(
    ticker: str,
    start: date | str,
    end: date | str,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance and return a validated DataFrame.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. ``"AAPL"``).
    start, end : date or str
        Date range for the historical data.
    auto_adjust : bool, optional
        Whether to use adjusted prices (default ``False``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``DatetimeIndex`` with at least the columns
        ``Open``, ``High``, ``Low``, ``Close``, and ``Volume``.

    Raises
    ------
    ValueError
        If the downloaded data is empty or fails validation.
    """
    logger.info("Fetching %s data from %s to %s", ticker, start, end)
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=auto_adjust)
    except Exception as exc:
        raise ValueError(
            f"Failed to download data for '{ticker}': {exc}"
        ) from exc

    if df is None or df.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}' in range {start}–{end}.")

    df = _flatten_columns(df)
    _validate_schema(df, ticker)
    df = _clean(df)
    return df


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a ``MultiIndex`` column header (e.g. from grouped download)."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(c) for c in col if c not in (None, "")).strip()
            for col in df.columns
        ]
    return df


def _validate_schema(df: pd.DataFrame, ticker: str) -> None:
    """Ensure the DataFrame contains the expected OHLCV columns."""
    columns_lower = {c.lower() for c in df.columns}
    missing = {r for r in REQUIRED_COLUMNS if r.lower() not in columns_lower}
    if missing:
        raise ValueError(
            f"Data for '{ticker}' is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric types and forward-fill small gaps."""
    close_col = _detect_close_column(df)
    df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    df.dropna(subset=[close_col], inplace=True)
    return df


def detect_close_column(df: pd.DataFrame) -> str:
    """Public wrapper — find the 'Close' column regardless of casing."""
    return _detect_close_column(df)


def _detect_close_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if "close" in str(c).lower():
            return c
    raise ValueError(f"Could not find a 'Close' column. Available: {list(df.columns)}")
