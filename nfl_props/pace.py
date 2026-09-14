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

# Calibrated via calibrate_sensitivity() against 3 seasons of real nflverse data -- see
# that function's docstring for the exact procedure, and Task 4 of
# docs/superpowers/plans/2026-09-14-nfl-props-game-pace-adjustment.md for how these
# three numbers were produced. Starts at 0.0 (a true no-op) so pace_adjust()'s own tests
# are meaningful before calibration has run.
SENSITIVITY: dict[str, float] = {"pass_yds": 0.0, "rush_yds": 0.0, "rec_yds": 0.0}


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
