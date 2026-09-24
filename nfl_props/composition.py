"""Point-estimate composition of the three Stage 1 sub-models into a single E[rec_yds]
prediction (spec's "Baselines to compare against", item 3: "opportunity x efficiency
point estimate, no Monte Carlo -- a sanity-check step before the full simulation").

E[rec_yds] = E[Targets] x P(catch) x E[Yards | catch]

Assumes independence between catch outcome and per-catch yardage for this point-estimate
purpose -- a standard, defensible simplification for a MEAN (the identity holds exactly
under independence). The eventual full Monte Carlo composition (spec section 1.3, a
separate later plan) does not need this assumption and can model any real correlation
directly through joint simulation instead.

`trailing_air_yards` has NO default -- it is required, not optional. Unlike
`trailing_share` (whose `None` fallback to the position-group average is an established,
already-validated convention in rate_model.predicted_rate), there is no equivalently safe
fallback for aDOT: a probe run against real data before this plan was written showed that
using a wrong/missing air_yards proxy silently changes CatchRate's prediction with no
error, and using the wrong fallback (0.0, or the group average) for a real player
produces a materially wrong estimate with no signal that anything went wrong. Forcing
every caller to supply it explicitly is deliberate.
"""
from __future__ import annotations

from . import catch_model, markets, rate_model, yards_model


def predicted_rec_yds_point_estimate(
    rate_ratings: rate_model.RateRatings,
    catch_ratings: catch_model.CatchRatings,
    yards_ratings: yards_model.YardsRatings,
    player_id: str,
    position_group: str,
    opponent_team: str,
    home: bool,
    trailing_air_yards: float,
    trailing_share: float | None = None,
) -> float:
    """E[rec_yds] for one player-game, composed from the three separately-fit Stage 1
    sub-models. `trailing_share` follows rate_model.predicted_rate's own existing
    convention (None falls back to the position-group average share); `trailing_air_yards`
    does not have this fallback and must always be supplied by the caller (see module
    docstring).
    """
    lam, _dispersion = rate_model.predicted_rate(
        rate_ratings, player_id, position_group, opponent_team, home, trailing_share=trailing_share)
    p_catch = catch_model.predicted_catch_rate(
        catch_ratings, player_id, position_group, opponent_team, home, air_yards=trailing_air_yards)
    mu, sigma = yards_model.predicted_yards_distribution(
        yards_ratings, player_id, position_group, opponent_team, home)
    e_yards_per_catch = markets.mean_yards(mu, sigma)
    return lam * p_catch * e_yards_per_catch
