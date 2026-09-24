"""Walk-forward evaluation for Yards | Reception (yards_model.py): refit every week,
predict the log-yardage distribution for each COMPLETED catch, score average negative
log-likelihood against a per-player season-to-date-average log-yards baseline (falling
back to the fitted model's own position-group base rate for a player with no prior
catches that season) -- mirrors backtest.py's baseline pattern and honesty framing
exactly, adapted for a per-catch row instead of a per-player-game row. Reuses
backtest.py's own log-normal NLL/PIT-calibration formulas (same math, not reimported --
matches this codebase's own convention of each backtest module owning its scoring
functions locally, e.g. rate_backtest.py/catch_backtest.py both do this too rather than
cross-importing a sibling's private helpers).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import model, yards_model

EPS = 1e-6


def walk_forward(targets: pd.DataFrame, start: date, halflife_days: float = 180.0,
                 reg: float = 5.0, min_catches: int = 200,
                 position_split_defense: bool = False) -> pd.DataFrame:
    """`targets` must already be filtered to complete == 1 by the caller -- same
    convention as yards_model.fit_yards_per_catch.
    """
    relevant = targets[targets["position_group"].isin(yards_model.RELEVANT_POSITIONS)].sort_values("date")
    rows = []
    ratings = None
    fit_week = None
    season_totals: dict[str, list[float]] = {}
    last_season = None
    for _, g in relevant.iterrows():
        if g["season"] != last_season:
            season_totals = {}
            last_season = g["season"]
        y = float(model._safe_log_yards(g["receiving_yards"]))
        if g["date"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = yards_model.fit_yards_per_catch(targets, as_of=g["date"].date(),
                                                          halflife_days=halflife_days, reg=reg,
                                                          min_catches=min_catches,
                                                          position_split_defense=position_split_defense)
                fit_week = week_key
            mu, sigma = yards_model.predicted_yards_distribution(ratings, g["player_id"],
                                                                  g["position_group"],
                                                                  g["opponent_team"], bool(g["home"]))
            prior = season_totals.get(g["player_id"], [])
            base_mu = (float(np.mean(prior)) if prior
                      else ratings.position_intercept.get(g["position_group"], ratings.intercept_fallback))
            base_sigma = ratings.sigma.get(g["position_group"], ratings.sigma_global)
            rows.append({
                "date": g["date"], "player_id": g["player_id"], "position_group": g["position_group"],
                "y": y, "model_mu": mu, "model_sigma": sigma, "base_mu": base_mu, "base_sigma": base_sigma,
            })
        season_totals.setdefault(g["player_id"], []).append(y)
    return pd.DataFrame(rows)


def _nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, EPS)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def summarize(df: pd.DataFrame) -> dict:
    y = df["y"].to_numpy()
    return {
        "n": int(len(df)),
        "nll_model": _nll(y, df["model_mu"].to_numpy(), df["model_sigma"].to_numpy()),
        "nll_baseline": _nll(y, df["base_mu"].to_numpy(), df["base_sigma"].to_numpy()),
    }


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    z = (df["y"] - df["model_mu"]) / df["model_sigma"].clip(lower=EPS)
    pit = pd.Series(norm.cdf(z), index=df.index)
    edges = np.linspace(0, 1, bins + 1)
    q = pd.cut(pit, edges, include_lowest=True)
    g = pit.groupby(q, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_pit": g.mean()})
