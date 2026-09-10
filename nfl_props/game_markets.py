"""Margin/total probability math for game outcomes. Generic odds helpers (devig,
parse_odds, to_american, fair_odds) live in markets.py and are imported directly by
consumers of this module -- they're not yardage-specific, so nothing here re-exports them.
"""
from __future__ import annotations

from scipy.stats import norm


def _prob_over(mu: float, sigma: float, line: float) -> float:
    z = (line - mu) / sigma
    return float(1 - norm.cdf(z))


def prob_margin_over(mu: float, sigma: float, line: float) -> float:
    """P(home_score - away_score > line) under the fitted Normal margin distribution.

    `line` uses nflverse's spread_line convention: positive means the home team was
    favored by that many points. Home covers the spread iff margin > line.
    """
    return _prob_over(mu, sigma, line)


def prob_total_over(mu: float, sigma: float, line: float) -> float:
    """P(home_score + away_score > line) under the fitted Normal total distribution."""
    return _prob_over(mu, sigma, line)


def moneyline_prob(mu: float, sigma: float) -> float:
    """P(home team wins) = P(margin > 0)."""
    return prob_margin_over(mu, sigma, 0.0)
