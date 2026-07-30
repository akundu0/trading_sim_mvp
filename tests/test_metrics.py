"""Tests for performance metrics module."""

import numpy as np
import pandas as pd
import pytest

from src.metrics.performance import compute_metrics, PerformanceMetrics


def _make_equity_curve(values: list[float]) -> pd.Series:
    """Helper: create a portfolio Series from a list of values."""
    dates = pd.bdate_range(start="2023-01-02", periods=len(values))
    return pd.Series(values, index=dates)


class TestComputeMetrics:
    def test_basic_return(self):
        curve = _make_equity_curve([10_000, 10_100, 10_200, 10_300])
        m = compute_metrics(curve, trade_profits=[], initial_capital=10_000)
        assert isinstance(m, PerformanceMetrics)
        assert m.total_return_pct == pytest.approx(3.0, rel=0.01)
        assert m.final_value == pytest.approx(10_300)

    def test_sharpe_positive_for_steady_gains(self):
        values = [10_000 + i * 10 for i in range(100)]
        curve = _make_equity_curve(values)
        m = compute_metrics(curve, [], 10_000)
        assert m.sharpe_ratio > 0

    def test_max_drawdown_negative(self):
        values = [10_000, 11_000, 9_000, 10_500]
        curve = _make_equity_curve(values)
        m = compute_metrics(curve, [], 10_000)
        assert m.max_drawdown < 0

    def test_win_rate_calculation(self):
        profits = [0.05, -0.02, 0.10, 0.03, -0.01]
        curve = _make_equity_curve([10_000] * 10)
        m = compute_metrics(curve, profits, 10_000)
        assert m.num_trades == 5
        assert m.win_rate == pytest.approx(60.0)

    def test_no_trades(self):
        curve = _make_equity_curve([10_000] * 5)
        m = compute_metrics(curve, [], 10_000)
        assert m.num_trades == 0
        assert m.win_rate == 0.0
        assert m.avg_trade_return == 0.0

    def test_all_losses(self):
        profits = [-0.05, -0.10, -0.03]
        curve = _make_equity_curve([10_000, 9_500, 8_550, 8_293.5])
        m = compute_metrics(curve, profits, 10_000)
        assert m.win_rate == 0.0
        assert m.avg_trade_return < 0

    def test_flat_equity(self):
        """Flat equity should yield zero Sharpe (no variance)."""
        curve = _make_equity_curve([10_000.0] * 20)
        m = compute_metrics(curve, [], 10_000)
        assert m.sharpe_ratio == 0.0
        assert m.total_return_pct == pytest.approx(0.0)
