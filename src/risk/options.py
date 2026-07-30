"""Black-Scholes options pricing via the mibian library."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import mibian
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """Computed option prices and Greeks."""

    call_price: float
    put_price: float
    strike: float
    days_to_expiry: int
    annual_vol_pct: float
    risk_free_rate_pct: float


def price_options(
    underlying: float,
    strike: float,
    risk_free_rate_pct: float,
    days_to_expiry: int,
    daily_sigma: float,
) -> Optional[OptionQuote]:
    """Compute European call/put prices using Black-Scholes.

    Parameters
    ----------
    underlying : float
        Current price of the underlying asset.
    strike : float
        Option strike price.  Pass ``0`` for at-the-money (ATM), which will
        be resolved to *underlying*.
    risk_free_rate_pct : float
        Annualized risk-free interest rate **in percent** (e.g. ``2.0``).
    days_to_expiry : int
        Calendar days until option expiration.
    daily_sigma : float
        Daily standard deviation of log returns (will be annualized
        internally).

    Returns
    -------
    OptionQuote or None
        ``None`` if pricing fails (e.g. invalid inputs).
    """
    if strike <= 0:
        strike = round(underlying, 2)

    annual_vol_pct = float(daily_sigma * np.sqrt(252) * 100)

    try:
        bs = mibian.BS(
            [underlying, strike, risk_free_rate_pct, days_to_expiry],
            volatility=annual_vol_pct,
        )
        return OptionQuote(
            call_price=bs.callPrice,
            put_price=bs.putPrice,
            strike=strike,
            days_to_expiry=days_to_expiry,
            annual_vol_pct=annual_vol_pct,
            risk_free_rate_pct=risk_free_rate_pct,
        )
    except Exception:
        logger.exception("Black-Scholes pricing failed")
        return None
