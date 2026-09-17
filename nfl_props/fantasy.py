"""PPR fantasy-points scoring and Monte Carlo combination of the yardage (log-normal)
and receptions/touchdown (quasi-Poisson) per-player-game distributions into a single
fantasy-points distribution, used to compare two or more players for a start/sit
decision.

Stat components are drawn independently of each other and of other players -- a real
game correlates e.g. rec_yds and receptions (more catches usually means more yards),
but modeling that joint structure is out of scope for v1; see the design spec's
independence caveat (docs/superpowers/specs/2026-09-16-fantasy-start-sit-saas-design.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import OFFSET

PPR_SCORING = {
    "pass_yds": 0.04,   # 1 pt per 25 pass yards
    "rush_yds": 0.1,    # 1 pt per 10 rush yards
    "rec_yds": 0.1,     # 1 pt per 10 rec yards
    "receptions": 1.0,  # PPR: 1 pt per reception
    "pass_td": 4.0,
    "rush_td": 6.0,
    "rec_td": 6.0,
}

N_SAMPLES = 20_000


@dataclass
class PlayerProjection:
    player_id: str
    samples: np.ndarray

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def stdev(self) -> float:
        return float(np.std(self.samples))


def simulate_player(player_id: str, yardage_dists: dict[str, tuple[float, float]],
                    rate_dists: dict[str, tuple[float, float]],
                    n_samples: int = N_SAMPLES, seed: int | None = None) -> PlayerProjection:
    """Simulate `n_samples` fantasy-point draws for one player-game.

    `yardage_dists`: {stat: (mu, sigma)} for any of "pass_yds"/"rush_yds"/"rec_yds" the
    player is projected for (log-normal, as returned by model.predicted_distribution) --
    omit a stat entirely if the player isn't projected for it.
    `rate_dists`: {stat: (lambda, dispersion)} for any of "receptions"/"pass_td"/
    "rush_td"/"rec_td" the player is projected for (quasi-Poisson, as returned by
    rate_model.predicted_rate).
    """
    rng = np.random.default_rng(seed)
    points = np.zeros(n_samples)
    for stat, (mu, sigma) in yardage_dists.items():
        yards = np.maximum(0.0, np.exp(rng.normal(mu, sigma, n_samples)) - OFFSET)
        points += yards * PPR_SCORING[stat]
    for stat, (lam, disp) in rate_dists.items():
        points += _sample_quasi_poisson(rng, lam, disp, n_samples) * PPR_SCORING[stat]
    return PlayerProjection(player_id=player_id, samples=points)


def _sample_quasi_poisson(rng: np.random.Generator, lam: float, dispersion: float,
                          n_samples: int) -> np.ndarray:
    """Draw counts with mean `lam` and variance `dispersion * lam` (dispersion >= 1.0).
    dispersion == 1.0 reduces to a plain Poisson draw. dispersion > 1.0 is modeled as a
    negative binomial with that same mean/variance (r = lam / (dispersion - 1),
    p = r / (r + lam) -- the standard NB2 reparameterization), matching how
    rate_backtest.py scores the same quasi-Poisson fit.
    """
    if dispersion <= 1.0 + 1e-9:
        return rng.poisson(lam, n_samples).astype(float)
    r = lam / (dispersion - 1.0)
    p = r / (r + lam)
    return rng.negative_binomial(r, p, n_samples).astype(float)


def prob_a_over_b(a: PlayerProjection, b: PlayerProjection) -> float:
    """P(a's fantasy points > b's), estimated from independent Monte Carlo draws."""
    return float(np.mean(a.samples > b.samples))


def rank_players(projections: list[PlayerProjection]) -> pd.DataFrame:
    rows = [{"player_id": p.player_id, "mean": p.mean, "stdev": p.stdev} for p in projections]
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
