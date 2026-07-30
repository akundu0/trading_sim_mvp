"""Tests for Black-Scholes options pricing module."""

import pytest

from src.risk.options import price_options, OptionQuote


class TestPriceOptions:
    def test_basic_call_put(self):
        opt = price_options(
            underlying=150.0,
            strike=150.0,
            risk_free_rate_pct=2.0,
            days_to_expiry=30,
            daily_sigma=0.02,
        )
        assert opt is not None
        assert isinstance(opt, OptionQuote)
        assert opt.call_price > 0
        assert opt.put_price > 0

    def test_atm_when_strike_zero(self):
        opt = price_options(
            underlying=200.0,
            strike=0.0,
            risk_free_rate_pct=2.0,
            days_to_expiry=30,
            daily_sigma=0.015,
        )
        assert opt is not None
        assert opt.strike == 200.0

    def test_deep_itm_call(self):
        opt = price_options(
            underlying=200.0,
            strike=100.0,
            risk_free_rate_pct=2.0,
            days_to_expiry=30,
            daily_sigma=0.02,
        )
        assert opt is not None
        # Deep ITM call should be worth at least the intrinsic value (roughly)
        assert opt.call_price >= 90.0

    def test_annual_vol_positive(self):
        opt = price_options(
            underlying=100.0,
            strike=100.0,
            risk_free_rate_pct=2.0,
            days_to_expiry=30,
            daily_sigma=0.025,
        )
        assert opt is not None
        assert opt.annual_vol_pct > 0

    def test_longer_expiry_higher_premium(self):
        short = price_options(100.0, 100.0, 2.0, 10, 0.02)
        long = price_options(100.0, 100.0, 2.0, 90, 0.02)
        assert short is not None and long is not None
        assert long.call_price >= short.call_price
