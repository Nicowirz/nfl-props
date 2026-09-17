"""Walk-forward evaluation for the reception/touchdown rate models (rate_model.py):
refit every week, project each player's actual game, score average negative
log-likelihood under the fitted quasi-Poisson distribution against a season-to-date
rolling-average baseline. Mirrors backtest.py's structure and honesty framing for the
yardage model -- see its module docstring.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

from . import model, rate_model

EPS = 1e-6


def walk_forward(stats: pd.DataFrame, stat: str, start: date, halflife_days: float = 180.0,
                 reg: float = 5.0, min_games: int = 200) -> pd.DataFrame:
    if stat in rate_model.SHARE_STATS and "game_id" in stats.columns:
        stats = model.add_trailing_share(stats, stat, halflife_days=halflife_days)
    relevant = stats[stats["position_group"].isin(rate_model.RELEVANT_POSITIONS[stat])].sort_values("date")
    rows = []
    ratings = None
    fit_week = None
    season_totals: dict[str, list[float]] = {}
    last_season = None
    for _, g in relevant.iterrows():
        if g["season"] != last_season:
            season_totals = {}
            last_season = g["season"]
        y = float(g[rate_model.STAT_COLUMN[stat]])
        if g["date"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = rate_model.fit_poisson(stats, stat, as_of=g["date"].date(),
                                                 halflife_days=halflife_days, reg=reg, min_games=min_games)
                fit_week = week_key
            lam, disp = rate_model.predicted_rate(ratings, g["player_id"], g["position_group"],
                                                  g["opponent_team"], bool(g["home"]),
                                                  trailing_share=g.get("trailing_share"))
            prior = season_totals.get(g["player_id"], [])
            group_rate = float(np.exp(ratings.position_intercept.get(g["position_group"],
                                                                      ratings.intercept_fallback)))
            base_lam = (sum(prior) + group_rate) / (len(prior) + 1)
            rows.append({
                "date": g["date"], "player_id": g["player_id"], "position_group": g["position_group"],
                "actual": y, "model_lambda": lam, "model_dispersion": disp, "base_lambda": base_lam,
            })
        season_totals.setdefault(g["player_id"], []).append(y)
    return pd.DataFrame(rows)


def _nll(y: np.ndarray, lam: np.ndarray, dispersion: np.ndarray) -> float:
    lam = np.maximum(lam, EPS)
    nll = np.zeros(len(y))
    poisson_mask = dispersion <= 1.0 + 1e-9
    if poisson_mask.any():
        nll[poisson_mask] = -poisson.logpmf(y[poisson_mask], lam[poisson_mask])
    nb_mask = ~poisson_mask
    if nb_mask.any():
        r, p = rate_model.nb_params(lam[nb_mask], dispersion[nb_mask])
        nll[nb_mask] = -nbinom.logpmf(y[nb_mask], r, p)
    return float(np.mean(nll))


def summarize(df: pd.DataFrame) -> dict:
    y = df["actual"].to_numpy()
    return {
        "n": int(len(df)),
        "nll_model": _nll(y, df["model_lambda"].to_numpy(), df["model_dispersion"].to_numpy()),
        "nll_baseline": _nll(y, df["base_lambda"].to_numpy(), np.ones(len(df))),
    }


def calibration(df: pd.DataFrame, bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Randomized PIT for discrete counts: for actual y under CDF F, draws
    U ~ Uniform(0,1) and computes F(y-1) + U*(F(y)-F(y-1)) -- the standard way to get a
    continuous, uniform-if-well-calibrated PIT from a discrete distribution (plain F(y)
    is systematically biased high for discrete data).
    """
    rng = np.random.default_rng(seed)
    y = df["actual"].to_numpy()
    lam = np.maximum(df["model_lambda"].to_numpy(), EPS)
    dispersion = df["model_dispersion"].to_numpy()
    poisson_mask = dispersion <= 1.0 + 1e-9
    f_lo = np.zeros(len(y))
    f_hi = np.zeros(len(y))
    if poisson_mask.any():
        f_lo[poisson_mask] = poisson.cdf(y[poisson_mask] - 1, lam[poisson_mask])
        f_hi[poisson_mask] = poisson.cdf(y[poisson_mask], lam[poisson_mask])
    nb_mask = ~poisson_mask
    if nb_mask.any():
        r, p = rate_model.nb_params(lam[nb_mask], dispersion[nb_mask])
        f_lo[nb_mask] = nbinom.cdf(y[nb_mask] - 1, r, p)
        f_hi[nb_mask] = nbinom.cdf(y[nb_mask], r, p)
    u = rng.uniform(size=len(y))
    pit = f_lo + u * (f_hi - f_lo)
    edges = np.linspace(0, 1, bins + 1)
    q = pd.cut(pit, edges, include_lowest=True)
    g = pd.Series(pit).groupby(q, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_pit": g.mean()})
