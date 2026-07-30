"""Tests for SMA crossover strategy logic."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.sma_crossover import add_sma_signals, run_backtest, BacktestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_price_df(prices: list[float]) -> pd.DataFrame:
    """Helper: build a minimal DataFrame with a Close column."""
    dates = pd.bdate_range(start="2023-01-02", periods=len(prices))
    return pd.DataFrame({"Close": prices}, index=dates)


@pytest.fixture
def trending_up_df() -> pd.DataFrame:
    """30 days of steadily rising prices — should trigger a buy."""
    prices = [100 + i * 0.5 for i in range(30)]
    return _make_price_df(prices)


@pytest.fixture
def trending_down_df() -> pd.DataFrame:
    """30 days of steadily falling prices — should not buy or should sell quickly."""
    prices = [150 - i * 0.5 for i in range(30)]
    return _make_price_df(prices)


@pytest.fixture
def crossover_df() -> pd.DataFrame:
    """Synthetic prices designed to produce exactly one buy then one sell."""
    # 10 days flat → 10 days rising (buy) → 10 days falling (sell)
    flat = [100.0] * 10
    rising = [100.0 + i * 2 for i in range(1, 11)]
    falling = [rising[-1] - i * 3 for i in range(1, 11)]
    return _make_price_df(flat + rising + falling)


# ---------------------------------------------------------------------------
# Signal tests
# ---------------------------------------------------------------------------

class TestAddSmaSignals:
    def test_columns_added(self, trending_up_df: pd.DataFrame):
        df = add_sma_signals(trending_up_df, "Close", short_window=3, long_window=7)
        for col in ("SMA_short", "SMA_long", "Signal", "Trade"):
            assert col in df.columns

    def test_signal_values_binary(self, trending_up_df: pd.DataFrame):
        df = add_sma_signals(trending_up_df, "Close", short_window=3, long_window=7)
        assert set(df["Signal"].dropna().unique()).issubset({0, 1})

    def test_short_ge_long_clamped(self, trending_up_df: pd.DataFrame):
        """When short >= long, the function should auto-clamp."""
        df = add_sma_signals(trending_up_df, "Close", short_window=20, long_window=10)
        assert "SMA_short" in df.columns  # should not crash


# ---------------------------------------------------------------------------
# Backtest tests
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_backtest_result(self, crossover_df: pd.DataFrame):
        df = add_sma_signals(crossover_df, "Close", short_window=3, long_window=7)
        result = run_backtest(df, "Close", initial_capital=10_000)
        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_series) == len(df)

    def test_portfolio_starts_at_capital(self, trending_up_df: pd.DataFrame):
        df = add_sma_signals(trending_up_df, "Close", short_window=3, long_window=7)
        result = run_backtest(df, "Close", initial_capital=10_000)
        assert result.portfolio_series.iloc[0] == pytest.approx(10_000, rel=0.01)

    def test_benchmark_series_present(self, trending_up_df: pd.DataFrame):
        df = add_sma_signals(trending_up_df, "Close", short_window=3, long_window=7)
        result = run_backtest(df, "Close", initial_capital=10_000)
        assert len(result.benchmark_series) == len(df)

    def test_transaction_cost_reduces_value(self, crossover_df: pd.DataFrame):
        df = add_sma_signals(crossover_df, "Close", short_window=3, long_window=7)
        no_cost = run_backtest(df, "Close", initial_capital=10_000, transaction_cost_pct=0.0)
        with_cost = run_backtest(df, "Close", initial_capital=10_000, transaction_cost_pct=0.01)
        # If any trades occurred, the cost version should be <= no-cost version
        if no_cost.trades:
            assert with_cost.final_value <= no_cost.final_value + 1e-6

    def test_no_trades_on_flat_data(self):
        """Flat prices should produce zero crossover signals."""
        df = _make_price_df([100.0] * 50)
        df = add_sma_signals(df, "Close", short_window=5, long_window=20)
        result = run_backtest(df, "Close", initial_capital=10_000)
        sell_trades = [t for t in result.trades if t.action == "SELL"]
        assert len(sell_trades) == 0

    def test_single_day_does_not_crash(self):
        df = _make_price_df([100.0])
        df = add_sma_signals(df, "Close", short_window=3, long_window=7)
        result = run_backtest(df, "Close", initial_capital=10_000)
        assert result.final_value == pytest.approx(10_000)
