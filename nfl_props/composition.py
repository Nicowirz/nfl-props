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

simulate_rec_yds() (spec section 1.3, item 6 of "Baselines to compare against") composes
the same three sub-models by Monte Carlo instead of a closed-form product: draw Targets
(Poisson/NB via rate_model.nb_params, mirroring fantasy.py's own local
quasi-Poisson-vs-NB branch rather than importing that module-private helper -- this
project's established convention shares the real reparameterization math
(rate_model.nb_params, public) but lets each caller keep its own thin dispatch branch
around it, exactly as rate_backtest.py's own NLL scoring already does), draw Catches as
Binomial(targets, p_catch) (p_catch is a fixed scalar per player-game here, so this is
exact, not an approximation), then draw per-catch log-normal Yards via a vectorized
(n_draws, max_catches) masked matrix and sum each row -- mirroring joint_sim.py's
np.maximum(0.0, np.exp(log_yards) - OFFSET) inversion idiom as a coding-style template
(not its same-game shared-shock mechanism, a different, already-gate-failed problem: see
docs/superpowers/plans/2026-09-24-nfl-props-rec-yards-point-estimate-composition.md's
"Next step"). A probe run against real fitted models before this plan was written
confirmed the MC sample's mean converges to predicted_rec_yds_point_estimate()'s
closed-form value within Monte Carlo noise (<1.1% relative difference at n=50,000 across
five real WR player-games).
"""
from __future__ import annotations

import numpy as np

from . import catch_model, markets, model, rate_model, yards_model


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


def simulate_rec_yds(
    rate_ratings: rate_model.RateRatings,
    catch_ratings: catch_model.CatchRatings,
    yards_ratings: yards_model.YardsRatings,
    player_id: str,
    position_group: str,
    opponent_team: str,
    home: bool,
    trailing_air_yards: float,
    trailing_share: float | None = None,
    n_draws: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Monte Carlo draw of total rec_yds for one player-game, composing the three
    separately-fit Stage 1 sub-models: Targets (quasi-Poisson/NB) -> Catches
    (Binomial(targets, p_catch), exact given a fixed p_catch) -> per-catch Yards
    (log-normal, vectorized). Returns an array of shape (n_draws,), every value >= 0.0.

    `trailing_air_yards`/`trailing_share` follow predicted_rec_yds_point_estimate()'s own
    conventions exactly (see this module's docstring). `seed=None` uses fresh entropy
    each call (non-reproducible); pass an int for reproducible draws.
    """
    rng = np.random.default_rng(seed)
    lam, dispersion = rate_model.predicted_rate(
        rate_ratings, player_id, position_group, opponent_team, home,
        trailing_share=trailing_share)
    p_catch = catch_model.predicted_catch_rate(
        catch_ratings, player_id, position_group, opponent_team, home,
        air_yards=trailing_air_yards)
    mu, sigma = yards_model.predicted_yards_distribution(
        yards_ratings, player_id, position_group, opponent_team, home)

    if dispersion <= 1.0 + 1e-9:
        targets = rng.poisson(lam, n_draws)
    else:
        r, p = rate_model.nb_params(lam, dispersion)
        targets = rng.negative_binomial(r, p, n_draws)
    catches = rng.binomial(targets, p_catch)

    max_catches = int(catches.max()) if n_draws > 0 else 0
    if max_catches == 0:
        return np.zeros(n_draws)
    log_yards = rng.normal(mu, sigma, size=(n_draws, max_catches))
    per_catch_yards = np.maximum(0.0, np.exp(log_yards) - model.OFFSET)
    mask = np.arange(max_catches)[None, :] < catches[:, None]
    return (per_catch_yards * mask).sum(axis=1)
