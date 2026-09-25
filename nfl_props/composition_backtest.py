"""Walk-forward evaluation for the full Monte Carlo rec_yds composition (spec section
1.3): refit all three Stage 1 sub-models (rate_model, catch_model, yards_model) every
week, draw >=10,000 Monte Carlo samples per in-window player-game via
composition.simulate_rec_yds(), moment-match a log-normal to the MC sample in
log(receiving_yards + OFFSET) space, and score average negative log-likelihood + PIT
calibration of the REAL observed receiving_yards under that fitted log-normal -- against
a per-player season-to-date-average log-yards baseline (falling back, for a player's
first scored game of the season, to a GAME-level direct model's position-group base rate
-- model.fit(stats, "rec_yds", ...), fit weekly purely to source this fallback; NOT
yards_model's per-catch position_intercept/sigma, a units mismatch since `y` here is
log(receiving_yards + OFFSET) at the game level, not the per-catch level), mirroring
backtest.py's/yards_backtest.py's baseline pattern and honesty framing exactly. Season
totals accumulate once per relevant player-game processed by this loop (including
zero-yard games), not once per catch, so "first scored game of the season" is the real
trigger for this fallback, not "no prior catches." Reuses backtest.py's own log-normal
NLL/PIT-calibration formulas (same math, not reimported -- matches this codebase's own
convention of each backtest module owning its scoring functions locally, e.g.
yards_backtest.py/rate_backtest.py/catch_backtest.py all do this too rather than
cross-importing a sibling's private helpers).

Each scored row's `composition.simulate_rec_yds()` call uses a distinct, reproducible
per-row seed (`seed + row_counter` when `seed` is not None; `None` otherwise), not one
shared `seed` value for every row -- so Monte Carlo sampling error is independent across
rows instead of being perfectly correlated across the entire scored population.

A player-game with literally zero targets recorded in that specific game has no
matching row in the targets table at all, so the left-merge against game_ay produces
NaN for trailing_air_yards -- this row is skipped entirely rather than erroring or
using a wrong fallback (add_trailing_air_yards itself never returns NaN on its own
output -- see its docstring -- so this is purely a join-miss, not a new-player/
no-prior-history scenario), matching the exact skip convention already used and
honestly recorded in the point-estimate composition's own BASELINE.md entry.
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
    rate_r = catch_r = yards_r = rec_r = None
    fit_week = None
    season_totals: dict[str, list[float]] = {}
    last_season = None
    row_counter = 0
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
                # Fit the existing, unchanged game-level direct model purely to source a
                # GAME-level baseline fallback (position_intercept/intercept_fallback/sigma/
                # sigma_global). yards_r's own position_intercept/sigma are fit in PER-CATCH
                # log-yards space (yards_model.fit_yards_per_catch), not per-game log-yards
                # space -- `y` here is log(receiving_yards + OFFSET) at the GAME level, so
                # falling back to yards_r's per-catch parameters was a units mismatch. This
                # mirrors backtest.py's own walk_forward() baseline computation exactly.
                rec_r = model.fit(stats, "rec_yds", as_of=as_of, halflife_days=halflife_days,
                                  reg=reg, min_games=min_games)
                fit_week = week_key
            ay = g["trailing_air_yards"]
            if not pd.isna(ay):
                # Each scored row gets a distinct but reproducible seed derived from the
                # base seed and the row's position among scored rows (seed + row_counter),
                # so per-row Monte Carlo error is independent across rows instead of being
                # perfectly correlated (every row previously reused the exact same seed).
                row_seed = None if seed is None else seed + row_counter
                row_counter += 1
                mc_sample = composition.simulate_rec_yds(
                    rate_r, catch_r, yards_r, g["player_id"], g["position_group"],
                    g["opponent_team"], bool(g["home"]), trailing_air_yards=float(ay),
                    trailing_share=g.get("trailing_share"), n_draws=n_draws, seed=row_seed)
                model_mu, model_sigma = _moment_match_lognormal(mc_sample)
                prior = season_totals.get(g["player_id"], [])
                base_mu = (float(np.mean(prior)) if prior
                          else rec_r.position_intercept.get(g["position_group"],
                                                            rec_r.intercept_fallback))
                base_sigma = rec_r.sigma.get(g["position_group"], rec_r.sigma_global)
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
