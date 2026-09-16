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

from . import game_model, model, pace

EPS = 1e-6


def walk_forward(stats: pd.DataFrame, stat: str, start: date, halflife_days: float = 180.0,
                 reg: float = 5.0, min_games: int = 200, pace_adjust: bool = False,
                 games: pd.DataFrame | None = None, sensitivity: dict[str, float] | None = None,
                 game_halflife_days: float = 365.0, game_reg: float = game_model.TOTAL_REG,
                 game_min_games: int = game_model.MIN_TOTAL_GAMES, wide_sigma: bool = False) -> pd.DataFrame:
    """Predict every relevant-position game on/after `start` (not just games that clear
    QUALIFY_MIN -- see model.fit's docstring for why; scoring on the same population the
    model is now fit on keeps this a fair, consistent comparison), refitting whenever the
    (season, week) of the game under test changes. Iterates ALL relevant-position games
    chronologically (not just the test window) so the season-to-date baseline has true
    prior-season history, not just history accumulated since `start`.

    pace_adjust=True additionally fits game_model.fit_score() at the same (season, week)
    cadence as the player refit and applies pace.pace_adjust() to model_mu using that
    game's predicted total -- requires `games` (e.g. data.load_games()'s output). Used to
    validate the game-pace adjustment's effect on calibration before it ships as the
    default in predict/parlay/best-bet; see
    docs/superpowers/specs/2026-09-14-nfl-props-game-pace-adjustment-design.md. The
    `team`/`opponent_team`/`home` fields in the output are always present (not gated on
    pace_adjust) so calibration code can derive each row's game for a join without a
    separate lookup.

    wide_sigma=True additionally passes `stat=stat` into model.predicted_distribution(),
    activating model.WIDE_SIGMA_STATS' low-sample sigma widening for stats in that set
    (currently just pass_yds) -- see model.py's comment above WIDE_SIGMA_STATS for why
    this defaults to False and stays a diagnostic-only flag rather than the default
    backtest behavior.
    """
    if pace_adjust and games is None:
        raise ValueError("pace_adjust=True requires games (e.g. data.load_games())")
    if stat in model.SHARE_STATS and "game_id" in stats.columns:
        stats = model.add_trailing_share(stats, stat, halflife_days=halflife_days)
    relevant = stats[stats["position_group"].isin(model.RELEVANT_POSITIONS[stat])].sort_values("date")
    rows = []
    ratings = None
    score_ratings = None
    fit_week = None
    season_totals: dict[str, list[float]] = {}
    last_season = None
    for _, g in relevant.iterrows():
        if g["season"] != last_season:
            season_totals = {}
            last_season = g["season"]
        y = float(model._safe_log_yards(g[model.STAT_COLUMN[stat]]))
        if g["date"] >= pd.Timestamp(start):
            week_key = (g["season"], g["week"])
            if week_key != fit_week:
                ratings = model.fit(stats, stat, as_of=g["date"].date(), halflife_days=halflife_days,
                                    reg=reg, min_games=min_games)
                if pace_adjust:
                    score_ratings = game_model.fit_score(games, as_of=g["date"].date(),
                                                          halflife_days=game_halflife_days,
                                                          reg=game_reg, min_games=game_min_games)
                fit_week = week_key
            mu, sigma = model.predicted_distribution(ratings, g["player_id"], g["position_group"],
                                                      g["opponent_team"], bool(g["home"]),
                                                      trailing_share=g.get("trailing_share"),
                                                      stat=stat if wide_sigma else "")
            if pace_adjust:
                home_team = g["team"] if g["home"] else g["opponent_team"]
                away_team = g["opponent_team"] if g["home"] else g["team"]
                total_mu, _ = game_model.predicted_total(score_ratings, home_team, away_team)
                league_avg_total = score_ratings.intercept * 2
                mu = pace.pace_adjust(mu, total_mu, league_avg_total, stat, sensitivity=sensitivity)
            prior = season_totals.get(g["player_id"], [])
            base_mu = (float(np.mean(prior)) if prior
                      else ratings.position_intercept.get(g["position_group"], ratings.intercept_fallback))
            base_sigma = ratings.sigma.get(g["position_group"], ratings.sigma_global)
            rows.append({
                "date": g["date"], "player_id": g["player_id"], "position_group": g["position_group"],
                "team": g["team"], "opponent_team": g["opponent_team"], "home": bool(g["home"]),
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
