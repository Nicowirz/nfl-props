"""Walk-forward evaluation for the full Monte Carlo rec_yds composition (spec section
1.3): refit all three Stage 1 sub-models (rate_model, catch_model, yards_model) every
week, draw >=10,000 Monte Carlo samples per in-window player-game via
composition.simulate_rec_yds(), moment-match a log-normal to the MC sample in
log(receiving_yards + OFFSET) space, and score average negative log-likelihood + PIT
calibration of the REAL observed receiving_yards under that fitted log-normal -- against
a per-player season-to-date-average log-yards baseline (falling back to the fitted
yards_model's own position-group base rate for a player with no prior catches that
season), mirroring backtest.py's/yards_backtest.py's baseline pattern and honesty
framing exactly. Reuses backtest.py's own log-normal NLL/PIT-calibration formulas (same
math, not reimported -- matches this codebase's own convention of each backtest module
owning its scoring functions locally, e.g. yards_backtest.py/rate_backtest.py/
catch_backtest.py all do this too rather than cross-importing a sibling's private
helpers).

A player-game with no prior target history (trailing_air_yards == NaN, see
model.add_trailing_air_yards's docstring) is skipped entirely -- there is no safe
fallback for aDOT (see composition.py's module docstring), matching the exact skip
convention already used and honestly recorded in the point-estimate composition's own
BASELINE.md entry.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import catch_model, composition, model, rate_model, yards_model

EPS = 1e-6


def walk_forward(stats: pd.DataFrame, targets: pd.DataFrame, start: date,
                 halflife_days: float = 180.0, reg: float = 5.0,
                 min_games: int = 200, min_targets: int = 200, min_catches: int = 200,
                 n_draws: int = 10_000, seed: int | None = None) -> pd.DataFrame:
    """`stats` is the weekly box-score DataFrame (data.load_player_stats()'s output, one
    row per player-game -- the test-population driver, same convention backtest.py's own
    walk_forward() uses). `targets` is the target-level DataFrame (data.load_targets()'s
    output, NOT pre-filtered to complete == 1 -- catch_model.fit_catch_rate needs
    incomplete rows too; the completed-only subset for yards_model.fit_yards_per_catch is
    derived internally).
    """
    catches = targets[targets["complete"] == 1.0].copy()
    stats_share = model.add_trailing_share(stats, "targets", halflife_days=halflife_days)
    targets_ay = model.add_trailing_air_yards(targets, halflife_days=halflife_days)
    game_ay = (targets_ay.sort_values(["player_id", "game_id"], kind="stable")
              .drop_duplicates(["player_id", "game_id"], keep="first")
              [["player_id", "game_id", "trailing_air_yards"]])

    relevant = (stats_share[stats_share["position_group"].isin(rate_model.RELEVANT_POSITIONS["targets"])]
               .sort_values("date").merge(game_ay, on=["player_id", "game_id"], how="left"))

    rows = []
    rate_r = catch_r = yards_r = None
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
                as_of = g["date"].date()
                rate_r = rate_model.fit_poisson(stats_share, "targets", as_of=as_of,
                                                halflife_days=halflife_days, reg=reg,
                                                min_games=min_games)
                catch_r = catch_model.fit_catch_rate(targets, as_of=as_of,
                                                     halflife_days=halflife_days, reg=reg,
                                                     min_targets=min_targets)
                yards_r = yards_model.fit_yards_per_catch(catches, as_of=as_of,
                                                           halflife_days=halflife_days, reg=reg,
                                                           min_catches=min_catches)
                fit_week = week_key
            ay = g["trailing_air_yards"]
            if not pd.isna(ay):
                mc_sample = composition.simulate_rec_yds(
                    rate_r, catch_r, yards_r, g["player_id"], g["position_group"],
                    g["opponent_team"], bool(g["home"]), trailing_air_yards=float(ay),
                    trailing_share=g.get("trailing_share"), n_draws=n_draws, seed=seed)
                model_mu, model_sigma = _moment_match_lognormal(mc_sample)
                prior = season_totals.get(g["player_id"], [])
                base_mu = (float(np.mean(prior)) if prior
                          else yards_r.position_intercept.get(g["position_group"],
                                                              yards_r.intercept_fallback))
                base_sigma = yards_r.sigma.get(g["position_group"], yards_r.sigma_global)
                rows.append({
                    "date": g["date"], "player_id": g["player_id"],
                    "position_group": g["position_group"], "y": y,
                    "model_mu": model_mu, "model_sigma": model_sigma,
                    "base_mu": base_mu, "base_sigma": base_sigma,
                })
        season_totals.setdefault(g["player_id"], []).append(y)
    return pd.DataFrame(rows)


def _moment_match_lognormal(mc_sample: np.ndarray) -> tuple[float, float]:
    """Fit a log-normal to the MC sample by matching moments in log(yards + OFFSET)
    space: mu_hat/sigma_hat are the sample mean/std of that transformed sample. No
    density assumption is needed for this step -- it is a direct empirical-moment fit,
    not a maximum-likelihood claim about the MC sample's true shape.
    """
    log_shifted = np.log(mc_sample + model.OFFSET)
    mu_hat = float(log_shifted.mean())
    sigma_hat = max(float(log_shifted.std(ddof=1)), EPS)
    return mu_hat, sigma_hat


def _nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, EPS)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def summarize(df: pd.DataFrame) -> dict:
    """Average negative log-likelihood of the model vs. the season-to-date baseline
    (lower is better)."""
    y = df["y"].to_numpy()
    return {
        "n": int(len(df)),
        "nll_model": _nll(y, df["model_mu"].to_numpy(), df["model_sigma"].to_numpy()),
        "nll_baseline": _nll(y, df["base_mu"].to_numpy(), df["base_sigma"].to_numpy()),
    }


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Probability integral transform: CDF(actual) under the model's own (moment-matched)
    distribution. Well-calibrated predictions give PIT values uniform on [0, 1].
    """
    z = (df["y"] - df["model_mu"]) / df["model_sigma"].clip(lower=EPS)
    pit = pd.Series(norm.cdf(z), index=df.index)
    edges = np.linspace(0, 1, bins + 1)
    q = pd.cut(pit, edges, include_lowest=True)
    g = pit.groupby(q, observed=True)
    return pd.DataFrame({"n": g.size(), "mean_pit": g.mean()})
