"""Same-game joint-outcome Monte Carlo simulation: draws correlated player-yardage
samples sharing one simulated game state, for computing JOINT (not marginal)
probabilities of same-game prop combinations.

model.py and game_model.py are unmodified and have zero knowledge this module exists --
callers fit both as usual and pass their already-validated outputs in here. No existing
marginal distribution changes; the shared shock only ever exists inside a simulation
draw, never written back to any Ratings object.

See docs/superpowers/specs/2026-09-17-nfl-props-joint-correlation-sim-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import OFFSET

N_DRAWS = 10_000

# Calibrated via calibrate_joint_sensitivity() (Task 2) against real 3-season cached
# nflverse data -- see that function and Tasks 4/5 of docs/superpowers/plans/2026-09-17-
# nfl-props-joint-correlation-sim.md for how these get populated. Starts at 0.0 (a true
# no-op) so this module's own tests are meaningful before calibration has run.
#
# Run 2026-09-17 (Task 4, Step 7): data.load_player_stats(3) / data.load_games(),
# start = (stats["date"].max() - timedelta(days=365)).date() = 2025-09-14
# (stats["date"].max() = 2026-09-14):
#   pass_yds: 0.030194480744343055 -> 0.030
#   rush_yds: 0.023558369202182485 -> 0.024
#
# Task 4, Step 8 -- real Gate 1 check (check_correlation_direction(stats, games, start)
# with the above sensitivities, 1-year window, default n=1000): n_pairs=578 (well above
# the ~20 floor, no window widening needed), real_corr=-0.0274 (deterministic -- real
# walk-forward data has no RNG). simulated_corr is NOT reproducible at the default
# n=1000: 4 consecutive runs of the exact brief command gave -0.1209, +0.0530, +0.0396,
# -0.0214 -- the sign itself flips run to run because check_correlation_direction never
# seeds simulate_player_yards, and the true per-pair signal is tiny relative to n=1000's
# Monte Carlo noise. A diagnostic-only rerun with n=50000 (same data/sensitivities, just
# less MC noise; not a parameter search for a passing result) converges to a stable
# simulated_corr=+0.0279 -- OPPOSITE SIGN from real_corr, and both magnitudes are
# negligibly close to zero (~0.03). GATE 1 FAILED on both prongs of the stated
# criterion: simulated_corr is not a reproducible non-negligible signal, and even its
# denoised value has the wrong sign. This is exactly the risk anticipated in Task 4's
# brief: the shared shock here is built from the game's TOTAL deviation, and pace.py's
# own calibrated sensitivities are both positive relative to total for pass_yds and
# rush_yds, so a shared total-based shock structurally pushes a QB's passing and his
# team's leading rusher's rushing in the SAME direction, while the real-world effect is
# a game-script/margin effect (weakly negative empirically here). Per the brief: Task 5
# does NOT proceed on this result. A follow-up using game_model.predicted_margin instead
# of (or alongside) total deviation is a plausible fix but is a separate design change
# requiring its own sign-off -- not applied here. pass_yds/rush_yds above are still the
# real calibrated-against-total values (this module's own calibration, kept independent
# of pace.py's SENSITIVITY per calibrate_joint_sensitivity()'s docstring), not tuned or
# cherry-picked to pass the gate.
JOINT_SENSITIVITY: dict[str, float] = {"pass_yds": 0.030, "rush_yds": 0.024, "rec_yds": 0.0}


def simulate_player_yards(total_mu: float, total_sigma: float, league_avg_total: float,
                          players: list[tuple[float, float, str]],
                          sensitivity: dict[str, float] | None = None,
                          n: int = N_DRAWS, seed: int | None = None) -> np.ndarray:
    """Draw `n` joint Monte Carlo samples of every player's REAL yards in ONE game,
    correlated via one shared per-draw total deviation from a typical game.

    `players`: one (mu, sigma, stat) tuple per player being jointly priced -- mu/sigma
    are each player's OWN marginal from model.predicted_distribution(), completely
    unmodified; `stat` selects which sensitivity[stat] shock applies to that player for
    each draw. Returns shape (n, len(players)) of real yards (OFFSET already inverted,
    clamped non-negative), one row per draw, columns in `players`' order.
    """
    s = JOINT_SENSITIVITY if sensitivity is None else sensitivity
    rng = np.random.default_rng(seed)
    total_draws = rng.normal(total_mu, total_sigma, size=n)
    # A drawn total at or below 0 is a real (if rare) possibility under a Normal total
    # distribution; floor at 1.0 before the log so total_dev stays finite, mirroring
    # _safe_log_yards's own floor-guard philosophy in model.py.
    total_dev = np.log(np.maximum(total_draws, 1.0) / league_avg_total)
    out = np.zeros((n, len(players)))
    for i, (mu, sigma, stat) in enumerate(players):
        shocked_mu = mu + s[stat] * total_dev
        log_yards = rng.normal(shocked_mu, sigma)
        out[:, i] = np.maximum(0.0, np.exp(log_yards) - OFFSET)
    return out


def joint_prob(samples: np.ndarray, conditions: list[tuple[int, str, float]]) -> float:
    """Empirical P(every condition holds) across `samples` (n_draws x n_players).
    `conditions`: list of (player_column_index, 'over'|'under', line).
    """
    mask = np.ones(samples.shape[0], dtype=bool)
    for idx, selection, line in conditions:
        mask &= (samples[:, idx] > line) if selection == "over" else (samples[:, idx] < line)
    return float(mask.mean())


def calibrate_joint_sensitivity(stats, games, stat: str, start, **kwargs) -> float:
    """Delegates to pace.calibrate_sensitivity() -- the same underlying regression
    (residual vs. this game's own predicted total deviation, walk-forward, leakage-safe)
    answers the question this module needs too. Kept as a separate entry point (not
    literally pace.SENSITIVITY) so this module's calibration is run, recorded, and
    validated independently -- see this module's JOINT_SENSITIVITY comment for why the
    SAME regression can be validated differently for a different use (a shared joint
    shock vs. a permanent marginal shift).
    """
    from . import pace
    return pace.calibrate_sensitivity(stats, games, stat, start, **kwargs)


def qb_leading_rusher_pairs(stats) -> "pd.DataFrame":
    """One row per (game_id, team) where a real, qualifying starting QB (attempts >=
    QUALIFY_MIN['pass_yds']) and a real leading rusher (max carries > 0 among that same
    team's RELEVANT_POSITIONS['rush_yds'] teammates in that SAME actual game) both
    appear and are two DIFFERENT players. Selecting which real players to examine for an
    already-completed game is not leakage -- each player's own future prediction still
    only uses data strictly before that game's date; only WHICH players to look at uses
    real, already-known game participants -- standard backtest evaluation practice, the
    same as every other walk-forward row in this project.

    Columns: game_id, date, team, opponent_team, home (bool, for this team), qb_player_id,
    qb_attempts, rusher_player_id, rusher_carries.
    """
    from .model import QUALIFY_MIN, RELEVANT_POSITIONS

    qbs = stats[(stats["position_group"] == "QB") & (stats["attempts"] >= QUALIFY_MIN["pass_yds"])]
    qbs = qbs[["game_id", "date", "team", "opponent_team", "home", "player_id", "attempts"]]
    qbs = qbs.rename(columns={"player_id": "qb_player_id", "attempts": "qb_attempts"})

    rushers = stats[stats["position_group"].isin(RELEVANT_POSITIONS["rush_yds"]) & (stats["carries"] > 0)]
    rushers = rushers.sort_values("carries", ascending=False).drop_duplicates(["game_id", "team"], keep="first")
    rushers = rushers[["game_id", "team", "player_id", "carries"]]
    rushers = rushers.rename(columns={"player_id": "rusher_player_id", "carries": "rusher_carries"})

    pairs = qbs.merge(rushers, on=["game_id", "team"], how="inner")
    pairs = pairs[pairs["qb_player_id"] != pairs["rusher_player_id"]]
    return pairs.reset_index(drop=True)


def check_correlation_direction(stats, games, start, sensitivity=None, n: int = 1000,
                                halflife_days: float = 180.0, reg: float = 5.0,
                                pass_min_games: int = 50, rush_min_games: int = 200,
                                game_halflife_days: float = 365.0, game_reg=None,
                                game_min_games=None) -> dict:
    """Gate 1: for real historical QB+leading-rusher pairs (walk-forward, no leakage),
    compare the SIMULATED correlation this module's mechanism produces against the REAL
    empirical correlation between their actual residuals. Returns {"n_pairs": int,
    "real_corr": float, "simulated_corr": float} (both correlations NaN if fewer than 2
    pairs are found) so the caller can judge direction/magnitude agreement before Gate 2.
    """
    from . import backtest, game_backtest, game_model as gm

    game_reg = gm.TOTAL_REG if game_reg is None else game_reg
    game_min_games = gm.MIN_TOTAL_GAMES if game_min_games is None else game_min_games

    pass_preds = backtest.walk_forward(stats, "pass_yds", start, halflife_days=halflife_days,
                                       reg=reg, min_games=pass_min_games)
    rush_preds = backtest.walk_forward(stats, "rush_yds", start, halflife_days=halflife_days,
                                       reg=reg, min_games=rush_min_games)
    game_preds = game_backtest.walk_forward_total(games, start, halflife_days=game_halflife_days,
                                                  reg=game_reg, min_games=game_min_games)

    relevant_stats = stats[stats["date"] >= pd.Timestamp(start)]
    pairs = qb_leading_rusher_pairs(relevant_stats)

    pass_cols = pass_preds[["player_id", "date", "y", "model_mu", "model_sigma"]].rename(
        columns={"y": "qb_y", "model_mu": "qb_mu", "model_sigma": "qb_sigma"})
    rush_cols = rush_preds[["player_id", "date", "y", "model_mu", "model_sigma"]].rename(
        columns={"y": "rusher_y", "model_mu": "rusher_mu", "model_sigma": "rusher_sigma"})

    joined = pairs.merge(pass_cols, left_on=["qb_player_id", "date"], right_on=["player_id", "date"])
    joined = joined.merge(rush_cols, left_on=["rusher_player_id", "date"], right_on=["player_id", "date"],
                          suffixes=("", "_r"))

    joined["home_team"] = np.where(joined["home"], joined["team"], joined["opponent_team"])
    joined["away_team"] = np.where(joined["home"], joined["opponent_team"], joined["team"])
    game_cols = game_preds.rename(columns={"model_mu": "total_mu", "model_sigma": "total_sigma"})[
        ["gameday", "home_team", "away_team", "total_mu", "total_sigma", "league_avg_total"]]
    joined = joined.merge(game_cols, left_on=["date", "home_team", "away_team"],
                          right_on=["gameday", "home_team", "away_team"])

    if len(joined) < 2:
        return {"n_pairs": len(joined), "real_corr": float("nan"), "simulated_corr": float("nan")}

    qb_residual = (joined["qb_y"] - joined["qb_mu"]).to_numpy()
    rusher_residual = (joined["rusher_y"] - joined["rusher_mu"]).to_numpy()
    real_corr = float(np.corrcoef(qb_residual, rusher_residual)[0, 1])

    sim_qb_dev, sim_rusher_dev = [], []
    for _, row in joined.iterrows():
        players = [(row["qb_mu"], row["qb_sigma"], "pass_yds"),
                  (row["rusher_mu"], row["rusher_sigma"], "rush_yds")]
        samples = simulate_player_yards(row["total_mu"], row["total_sigma"], row["league_avg_total"],
                                        players, sensitivity=sensitivity, n=n)
        log_qb = np.log(np.maximum(samples[:, 0], 1.0 - OFFSET) + OFFSET)
        log_rusher = np.log(np.maximum(samples[:, 1], 1.0 - OFFSET) + OFFSET)
        sim_qb_dev.append(float(np.mean(log_qb)) - row["qb_mu"])
        sim_rusher_dev.append(float(np.mean(log_rusher)) - row["rusher_mu"])
    simulated_corr = float(np.corrcoef(sim_qb_dev, sim_rusher_dev)[0, 1])

    return {"n_pairs": len(joined), "real_corr": real_corr, "simulated_corr": simulated_corr}
