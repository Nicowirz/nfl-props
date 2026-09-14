"""Walk-forward evaluation for the margin and totals models: refit every week, project
each completed game, score average negative log-likelihood and calibration against a
naive baseline. Margin's baseline is home-field-only (ignores team identity); totals'
baseline is the league-average total so far. No free historical closing-line data exists
for a specific book, so this validates model *calibration*, not proven market edge -- see
README.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import game_model

EPS = 1e-6


def walk_forward_margin(games: pd.DataFrame, start: date, halflife_days: float = 365.0,
                        reg: float = game_model.MARGIN_REG,
                        min_games: int = game_model.MIN_MARGIN_GAMES) -> pd.DataFrame:
    """Predict every completed game on/after `start`, refitting whenever the (season, week)
    of the game under test changes. Baseline: the league's expanding-average margin so far
    (ignores which specific teams are playing).
    """
    df = games.dropna(subset=["home_score", "away_score"]).sort_values("gameday")
    rows = []
    ratings = None
    fit_week = None
    league_margins: list[float] = []
    for _, g in df.iterrows():
        y = float(g["home_score"] - g["away_score"])
        if g["gameday"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = game_model.fit_margin(games, as_of=g["gameday"].date(),
                                                halflife_days=halflife_days, reg=reg, min_games=min_games)
                fit_week = week_key
            mu, sigma = game_model.predicted_margin(ratings, g["home_team"], g["away_team"],
                                                    neutral=bool(g["neutral"]))
            base_mu = float(np.mean(league_margins)) if league_margins else ratings.home_field
            base_sigma = ratings.sigma
            rows.append({
                "gameday": g["gameday"], "home_team": g["home_team"], "away_team": g["away_team"],
                "actual_margin": y, "model_mu": mu, "model_sigma": sigma,
                "base_mu": base_mu, "base_sigma": base_sigma,
            })
        league_margins.append(y)
    return pd.DataFrame(rows)


def walk_forward_total(games: pd.DataFrame, start: date, halflife_days: float = 365.0,
                       reg: float = game_model.TOTAL_REG,
                       min_games: int = game_model.MIN_TOTAL_GAMES) -> pd.DataFrame:
    """Predict every completed game's total on/after `start`, refitting on (season, week)
    change. Baseline: the league's expanding-average total so far.
    """
    df = games.dropna(subset=["home_score", "away_score"]).sort_values("gameday")
    rows = []
    ratings = None
    fit_week = None
    league_totals: list[float] = []
    for _, g in df.iterrows():
        y = float(g["home_score"] + g["away_score"])
        if g["gameday"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = game_model.fit_score(games, as_of=g["gameday"].date(),
                                               halflife_days=halflife_days, reg=reg, min_games=min_games)
                fit_week = week_key
            mu, sigma = game_model.predicted_total(ratings, g["home_team"], g["away_team"])
            base_mu = float(np.mean(league_totals)) if league_totals else ratings.intercept * 2
            base_sigma = float(ratings.sigma * np.sqrt(2))
            rows.append({
                "gameday": g["gameday"], "home_team": g["home_team"], "away_team": g["away_team"],
                "actual_total": y, "model_mu": mu, "model_sigma": sigma,
                "base_mu": base_mu, "base_sigma": base_sigma,
                "league_avg_total": ratings.intercept * 2,
            })
        league_totals.append(y)
    return pd.DataFrame(rows)


def _nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, EPS)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def summarize(df: pd.DataFrame, actual_col: str) -> dict:
    """Average negative log-likelihood of the model vs. the naive baseline (lower is better)."""
    y = df[actual_col].to_numpy(dtype=float)
    return {
        "n": int(len(df)),
        "nll_model": _nll(y, df["model_mu"].to_numpy(), df["model_sigma"].to_numpy()),
        "nll_baseline": _nll(y, df["base_mu"].to_numpy(), df["base_sigma"].to_numpy()),
    }


def calibration(df: pd.DataFrame, actual_col: str, bins: int = 10) -> pd.DataFrame:
    """Probability integral transform: CDF(actual) under the model's own distribution.
    Well-calibrated predictions give PIT values uniform on [0, 1].
    """
    z = (df[actual_col] - df["model_mu"]) / df["model_sigma"].clip(lower=EPS)
    pit = pd.Series(norm.cdf(z), index=df.index)
    edges = np.linspace(0, 1, bins + 1)
    q = pd.cut(pit, edges, include_lowest=True)
    g = pit.groupby(q, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_pit": g.mean()})
