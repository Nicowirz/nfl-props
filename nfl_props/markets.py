"""Log-normal yardage probability, devig, and odds-format helpers."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import norm

from .model import OFFSET


def prob_over(mu: float, sigma: float, line: float) -> float:
    """P(yards > line) under the fitted log-normal distribution for this player-game."""
    z = (np.log(line + OFFSET) - mu) / sigma
    return float(1 - norm.cdf(z))


def mean_yards(mu: float, sigma: float) -> float:
    return float(np.exp(mu + sigma ** 2 / 2) - OFFSET)


def median_yards(mu: float) -> float:
    return float(np.exp(mu) - OFFSET)


def devig(odds: Sequence[float]) -> list[float]:
    """Remove the bookmaker margin by proportional normalisation."""
    raw = np.array([1.0 / o for o in odds], dtype=float)
    return list(raw / raw.sum())


def fair_odds(p: float) -> float:
    return float("inf") if p <= 0 else 1.0 / p


def to_american(decimal: float) -> str:
    if decimal >= 2:
        return f"+{round((decimal - 1) * 100):d}"
    return f"-{round(100 / (decimal - 1)):d}"


def parse_odds(text: str | float) -> float:
    """Accept decimal ('1.91') or American ('+150', '-110', 150, -110) odds; return decimal.

    A bare number of 100 or more (or -100 or less) is taken as American, since no decimal
    price in this market is that long.
    """
    s = str(text).strip()
    n = float(s)
    if s.startswith(("+", "-")) or abs(n) >= 100:
        if n > 0:
            return 1 + n / 100
        return 1 + 100 / abs(n)
    if n <= 1:
        raise ValueError(f"odds {text!r} are not a valid decimal or American price")
    return n
