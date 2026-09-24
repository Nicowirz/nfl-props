"""Walk-forward evaluation for CatchRate | Targets (catch_model.py): refit every week,
predict each target's catch probability, score average log-loss against a per-player
recency-weighted trailing completion-rate baseline (blended with the fitted model's own
position-group base rate when a player has no prior targets that season) -- mirrors
rate_backtest.py's season-to-date-average baseline pattern and backtest.py's honesty
framing (see its module docstring), adapted for a per-target Bernoulli row instead of a
per-player-game count row.

Calibration here is a standard binary reliability table (predicted-probability decile vs.
actual catch rate in that decile), not the randomized-PIT bucketing rate_backtest.py uses
for count data. PIT's randomization exists specifically to correct discretization bias
for distributions with many integer support points; a Bernoulli outcome has only two
(0, 1), so a reliability diagram is the standard, more legible tool here -- it directly
answers "when the model says 70%, does it happen about 70% of the time".
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import catch_model

EPS = 1e-6


def walk_forward(targets: pd.DataFrame, start: date, halflife_days: float = 180.0,
                 reg: float = 5.0, min_targets: int = 200) -> pd.DataFrame:
    relevant = targets[targets["position_group"].isin(catch_model.RELEVANT_POSITIONS)].sort_values("date")
    rows = []
    ratings = None
    fit_week = None
    season_totals: dict[str, list[float]] = {}
    last_season = None
    for _, g in relevant.iterrows():
        if g["season"] != last_season:
            season_totals = {}
            last_season = g["season"]
        y = float(g["complete"])
        if g["date"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = catch_model.fit_catch_rate(targets, as_of=g["date"].date(),
                                                     halflife_days=halflife_days, reg=reg,
                                                     min_targets=min_targets)
                fit_week = week_key
            p_model = catch_model.predicted_catch_rate(ratings, g["player_id"], g["position_group"],
                                                        g["opponent_team"], bool(g["home"]),
                                                        air_yards=float(g["air_yards"]))
            prior = season_totals.get(g["player_id"], [])
            group_p = 1.0 / (1.0 + np.exp(-ratings.position_intercept.get(
                g["position_group"], ratings.intercept_fallback)))
            base_p = (sum(prior) + group_p) / (len(prior) + 1)
            rows.append({
                "date": g["date"], "player_id": g["player_id"], "position_group": g["position_group"],
                "actual": y, "model_p": p_model, "base_p": base_p,
            })
        season_totals.setdefault(g["player_id"], []).append(y)
    return pd.DataFrame(rows)


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def summarize(df: pd.DataFrame) -> dict:
    y = df["actual"].to_numpy()
    return {
        "n": int(len(df)),
        "nll_model": _log_loss(y, df["model_p"].to_numpy()),
        "nll_baseline": _log_loss(y, df["base_p"].to_numpy()),
    }


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    p = df["model_p"].to_numpy()
    y = df["actual"].to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    q = pd.cut(p, edges, include_lowest=True)
    g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_predicted": g["p"].mean(), "actual_rate": g["y"].mean()})
