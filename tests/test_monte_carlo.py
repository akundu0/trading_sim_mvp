"""Tests for Monte Carlo simulation module."""

import numpy as np
import pandas as pd
import pytest

from src.risk.monte_carlo import simulate, compute_log_return_params, MonteCarloResult


class TestComputeLogReturnParams:
    def test_basic(self):
        prices = pd.Series([100, 102, 101, 105, 107])
        mu, sigma = compute_log_return_params(prices)
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma >= 0

    def test_constant_prices(self):
        prices = pd.Series([100.0] * 20)
        mu, sigma = compute_log_return_params(prices)
        assert mu == 0.0
        assert sigma == 0.0

    def test_single_price(self):
        prices = pd.Series([100.0])
        mu, sigma = compute_log_return_params(prices)
        assert mu == 0.0
        assert sigma == 0.0


class TestSimulate:
    def test_output_shape(self):
        result = simulate(last_price=100, mu=0.001, sigma=0.02, days=30, runs=50, seed=42)
        assert isinstance(result, MonteCarloResult)
        assert result.paths.shape == (50, 31)  # runs × (days + 1)
        assert result.median_path.shape == (31,)

    def test_all_paths_start_at_last_price(self):
        result = simulate(last_price=150, mu=0.0, sigma=0.01, days=10, runs=100, seed=42)
        np.testing.assert_allclose(result.paths[:, 0], 150.0)

    def test_deterministic_with_seed(self):
        r1 = simulate(100, 0.001, 0.02, 30, 50, seed=123)
        r2 = simulate(100, 0.001, 0.02, 30, 50, seed=123)
        np.testing.assert_array_equal(r1.paths, r2.paths)

    def test_var_is_negative_or_small(self):
        """VaR at 95% should generally be negative or near-zero for volatile data."""
        result = simulate(100, 0.0, 0.05, 252, 1000, seed=0)
        assert result.var_95 <= 0.20  # unlikely to be hugely positive

    def test_cvar_le_var(self):
        """CVaR (expected shortfall) should be at most equal to VaR (both are tail measures)."""
        result = simulate(100, 0.0, 0.03, 252, 500, seed=7)
        assert result.cvar_95 <= result.var_95 + 1e-9

    def test_percentiles_ordered(self):
        result = simulate(100, 0.001, 0.02, 60, 500, seed=99)
        assert result.percentile_5 <= result.percentile_95

    def test_zero_sigma(self):
        """With zero volatility and positive drift, all paths should rise identically."""
        result = simulate(100, 0.01, 0.0, 10, 5, seed=42)
        # All final prices should be identical
        finals = result.paths[:, -1]
        np.testing.assert_allclose(finals, finals[0])
