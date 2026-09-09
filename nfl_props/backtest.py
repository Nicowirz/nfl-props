"""Walk-forward evaluation: refit every week, project each player's actual game, score
average negative log-likelihood and calibration against a season-to-date rolling-average
baseline. No free historical prop-line data exists, so this validates model *calibration*,
not real market edge -- see README.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import model

EPS = 1e-6


def walk_forward(stats: pd.DataFrame, stat: str, start: date, halflife_days: float = 180.0,
                 reg: float = 5.0, min_games: int = 200) -> pd.DataFrame:
    """Predict every qualifying game on/after `start`, refitting whenever the (season, week)
    of the game under test changes. Iterates ALL qualifying games chronologically (not just
    the test window) so the season-to-date baseline has true prior-season history, not just
    history accumulated since `start`.
    """
    qcol, qmin = model.QUALIFY_COLUMN[stat], model.QUALIFY_MIN[stat]
    qualifying = stats[stats[qcol] >= qmin].sort_values("date")
    rows = []
    ratings = None
    fit_week = None
    season_totals: dict[str, list[float]] = {}
    last_season = None
    for _, g in qualifying.iterrows():
        if g["season"] != last_season:
            season_totals = {}
            last_season = g["season"]
        y = float(np.log(g[model.STAT_COLUMN[stat]] + model.OFFSET))
        if g["date"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = model.fit(stats, stat, as_of=g["date"].date(), halflife_days=halflife_days,
                                    reg=reg, min_games=min_games)
                fit_week = week_key
            mu, sigma = model.predicted_distribution(ratings, g["player_id"], g["position_group"],
                                                      g["opponent_team"], bool(g["home"]))
            prior = season_totals.get(g["player_id"], [])
            base_mu = float(np.mean(prior)) if prior else ratings.intercept
            base_sigma = ratings.sigma.get(g["position_group"], ratings.sigma_global)
            rows.append({
                "date": g["date"], "player_id": g["player_id"], "position_group": g["position_group"],
                "actual_yards": g[model.STAT_COLUMN[stat]], "y": y,
                "model_mu": mu, "model_sigma": sigma, "base_mu": base_mu, "base_sigma": base_sigma,
            })
        season_totals.setdefault(g["player_id"], []).append(y)
    return pd.DataFrame(rows)


def _nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, EPS)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def summarize(df: pd.DataFrame) -> dict:
    """Average negative log-likelihood of the model vs. the season-to-date baseline (lower is better)."""
    y = df["y"].to_numpy()
    return {
        "n": int(len(df)),
        "nll_model": _nll(y, df["model_mu"].to_numpy(), df["model_sigma"].to_numpy()),
        "nll_baseline": _nll(y, df["base_mu"].to_numpy(), df["base_sigma"].to_numpy()),
    }


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Probability integral transform: CDF(actual) under the model's own distribution.
    Well-calibrated predictions give PIT values uniform on [0, 1] -- each fixed-width bucket
    should hold ~1/bins of predictions.
    """
    z = (df["y"] - df["model_mu"]) / df["model_sigma"].clip(lower=EPS)
    pit = pd.Series(norm.cdf(z), index=df.index)
    edges = np.linspace(0, 1, bins + 1)
    q = pd.cut(pit, edges, include_lowest=True)
    g = pit.groupby(q, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_pit": g.mean()})
