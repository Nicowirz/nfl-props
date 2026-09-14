"""Post-hoc adjustment of player-yardage predictions for a game's expected pace
(scoring environment), using the already-independently-fitted game-outcome total model.

model.py has zero knowledge this module exists -- the caller (cli.py) is responsible for
fitting game_model, getting each game's predicted_total, and applying pace_adjust() to
the mu it already got from model.predicted_distribution(). This module must import only
numpy at the top level (no other nfl_props module) so that backtest.py can import it
without creating a circular import -- see calibrate_sensitivity()'s own deferred imports
below for why that matters.

See docs/superpowers/specs/2026-09-14-nfl-props-game-pace-adjustment-design.md.
"""
from __future__ import annotations

import numpy as np

# Calibrated via calibrate_sensitivity() against 3 seasons of real nflverse data, run
# 2026-09-14 with start = 1 year before the most recent game in the dataset (the
# brief's original 2-year lookback left too little pre-start history at this point in
# the 2026 season and was shortened to match the project's proven-working 1-year
# default -- see task-4-report.md). Results: pass_yds=0.020, rush_yds=0.008,
# rec_yds=-0.045.
SENSITIVITY: dict[str, float] = {"pass_yds": 0.020, "rush_yds": 0.008, "rec_yds": -0.045}


def pace_adjust(mu: float, predicted_total: float, league_avg_total: float, stat: str,
                sensitivity: dict[str, float] | None = None) -> float:
    """Shift a player's log(yards + OFFSET) mu by how far this game's predicted total is
    from a typical game, in the same additive log space model.py's ability/defense/
    home_field terms already live in. predicted_total == league_avg_total is a no-op by
    construction (log(1) == 0), and sensitivity[stat] == 0.0 is always a no-op regardless
    of predicted_total.
    """
    s = SENSITIVITY if sensitivity is None else sensitivity
    return mu + s[stat] * float(np.log(predicted_total / league_avg_total))


def calibrate_sensitivity(stats, games, stat: str, start, halflife_days: float = 180.0,
                          reg: float = 5.0, min_games: int = 200,
                          game_halflife_days: float = 365.0, game_reg: float | None = None,
                          game_min_games: int | None = None) -> float:
    """Derive how much a game's predicted total (known only as of that game's own date,
    never the actual final score) correlates with the player model's own residual, for
    `stat`. Reuses backtest.walk_forward() (unadjusted) and
    game_backtest.walk_forward_total() exactly as they already exist -- both are already
    leakage-safe, refitting as of each historical date -- and joins each player-game row
    to its game's predicted total via (date, home_team, away_team). Using the actual
    final total instead of the game model's own prediction would leak future information
    into the calibration; the whole point is to learn what the signal was worth knowing
    in advance, matching exactly how pace_adjust() is used at real prediction time.

    Returns a single float (0.0 if there isn't enough overlapping data to fit anything
    meaningful) -- the caller decides whether/how to persist it into SENSITIVITY.
    """
    from . import backtest, game_backtest, game_model as gm

    game_reg = gm.TOTAL_REG if game_reg is None else game_reg
    game_min_games = gm.MIN_TOTAL_GAMES if game_min_games is None else game_min_games

    player_preds = backtest.walk_forward(stats, stat, start, halflife_days=halflife_days,
                                         reg=reg, min_games=min_games)
    game_preds = game_backtest.walk_forward_total(games, start, halflife_days=game_halflife_days,
                                                  reg=game_reg, min_games=game_min_games)
    if player_preds.empty or game_preds.empty:
        return 0.0

    player_preds = player_preds.copy()
    player_preds["home_team"] = np.where(player_preds["home"], player_preds["team"],
                                         player_preds["opponent_team"])
    player_preds["away_team"] = np.where(player_preds["home"], player_preds["opponent_team"],
                                         player_preds["team"])

    game_key = game_preds.rename(columns={"model_mu": "game_total_mu"})[
        ["gameday", "home_team", "away_team", "game_total_mu", "league_avg_total"]]

    joined = player_preds.merge(game_key, left_on=["date", "home_team", "away_team"],
                                right_on=["gameday", "home_team", "away_team"], how="inner")
    if joined.empty:
        return 0.0

    residual = (joined["y"] - joined["model_mu"]).to_numpy(dtype=float)
    total_dev = np.log(joined["game_total_mu"].to_numpy(dtype=float)
                       / joined["league_avg_total"].to_numpy(dtype=float))
    days_ago = (joined["date"].max() - joined["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    denom = float(np.sum(w * total_dev ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(w * total_dev * residual) / denom)
