from datetime import date

import numpy as np
import pytest

from nfl_props import catch_model, composition, rate_model, yards_model


def _rate_ratings():
    return rate_model.RateRatings(
        stat="targets", players=["P1"], player_name={"P1": "P1"}, position_group={"P1": "WR"},
        ability={"P1": 0.2}, teams=["T0"], defense={"T0": 0.1},
        position_intercept={"WR": 1.5}, intercept_fallback=1.5,
        share_coef=1.2, share_fallback={"WR": 0.2},
        home_field=0.05, dispersion={"WR": 1.0}, dispersion_global=1.0,
        as_of=date(2025, 9, 1), n_games=500, game_counts={"P1": 30},
    )


def _catch_ratings():
    return catch_model.CatchRatings(
        players=["P1"], player_name={"P1": "P1"}, position_group={"P1": "WR"},
        ability={"P1": 0.1}, teams=["T0"], defense={"T0": -0.05},
        position_intercept={"WR": 0.6}, intercept_fallback=0.6,
        air_yards_coef=-0.08, home_field=0.05,
        as_of=date(2025, 9, 1), n_targets=500, target_counts={"P1": 60},
    )


def _yards_ratings():
    return yards_model.YardsRatings(
        players=["P1"], player_name={"P1": "P1"}, position_group={"P1": "WR"},
        ability={"P1": 0.15}, teams=["T0"], defense={"T0": 0.02},
        position_intercept={"WR": 4.6}, intercept_fallback=4.6,
        home_field=0.05, sigma={"WR": 0.35}, sigma_global=0.35,
        as_of=date(2025, 9, 1), n_catches=400, catch_counts={"P1": 40},
    )


def test_predicted_rec_yds_point_estimate_matches_hand_computed_product():
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    trailing_share, trailing_air_yards = 0.25, 8.5

    result = composition.predicted_rec_yds_point_estimate(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=trailing_air_yards, trailing_share=trailing_share)

    lam, _ = rate_model.predicted_rate(rate_r, "P1", "WR", "T0", True, trailing_share=trailing_share)
    p_catch = catch_model.predicted_catch_rate(catch_r, "P1", "WR", "T0", True, air_yards=trailing_air_yards)
    mu, sigma = yards_model.predicted_yards_distribution(yards_r, "P1", "WR", "T0", True)
    from nfl_props import markets
    e_yards = markets.mean_yards(mu, sigma)
    expected = lam * p_catch * e_yards

    assert result == pytest.approx(expected)
    assert result > 0.0


def test_predicted_rec_yds_point_estimate_requires_trailing_air_yards():
    """trailing_air_yards has no default -- a caller who forgets it gets a clear
    TypeError at the call site, not a silently-wrong prediction using some fallback
    depth value. Verified as a real, previously-caught pitfall: passing a wrong/missing
    aDOT proxy silently changes CatchRate's prediction with no error at all, so this
    parameter is deliberately required, not optional with a default.
    """
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    try:
        composition.predicted_rec_yds_point_estimate(rate_r, catch_r, yards_r, "P1", "WR", "T0", True)
        assert False, "expected TypeError for missing required trailing_air_yards"
    except TypeError:
        pass


def test_predicted_rec_yds_point_estimate_higher_share_gives_higher_estimate():
    """Sanity check using real generated inputs (not a hardcoded value): a higher
    trailing_share must predict a higher E[Targets] and therefore a higher point
    estimate, all else equal -- this is the exact real pitfall a probe run caught
    before this plan was written (using the None/group-average fallback instead of a
    player's own real share silently underestimated real high-volume players by
    roughly 2-3x).
    """
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    low = composition.predicted_rec_yds_point_estimate(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True, trailing_air_yards=8.5, trailing_share=0.10)
    high = composition.predicted_rec_yds_point_estimate(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True, trailing_air_yards=8.5, trailing_share=0.35)
    assert high > low


def test_simulate_rec_yds_mean_converges_to_point_estimate():
    """A real probe run (before this plan was written) against these exact fixture
    values, at n_draws=200_000 and seed=123, measured a relative difference of -0.011%
    between the MC sample mean and the closed-form point estimate -- well inside a 3%
    tolerance, chosen generously to stay robust to a different numpy version's RNG
    stream while still being tight enough to catch a real implementation bug (e.g. a
    missing catches-vs-targets step, or an unmasked matrix sum).
    """
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    trailing_share, trailing_air_yards = 0.25, 8.5

    point_est = composition.predicted_rec_yds_point_estimate(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=trailing_air_yards, trailing_share=trailing_share)

    samples = composition.simulate_rec_yds(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=trailing_air_yards, trailing_share=trailing_share,
        n_draws=200_000, seed=123)

    mc_mean = samples.mean()
    rel_diff = abs(mc_mean - point_est) / point_est
    assert rel_diff < 0.03, f"MC mean {mc_mean} vs point estimate {point_est}: {rel_diff:.4f}"


def test_simulate_rec_yds_returns_correct_shape_and_nonnegative():
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    samples = composition.simulate_rec_yds(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=8.5, trailing_share=0.25, n_draws=5_000, seed=1)
    assert isinstance(samples, np.ndarray)
    assert samples.shape == (5_000,)
    assert np.all(samples >= 0.0)


def test_simulate_rec_yds_same_seed_is_reproducible():
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    a = composition.simulate_rec_yds(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=8.5, trailing_share=0.25, n_draws=5_000, seed=99)
    b = composition.simulate_rec_yds(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=8.5, trailing_share=0.25, n_draws=5_000, seed=99)
    assert np.array_equal(a, b)


def test_simulate_rec_yds_near_zero_catch_rate_gives_all_zero_draws():
    """A real probe run confirmed catch_model.predicted_catch_rate() returns
    p_catch ~= 2.06e-9 for this fixture (position_intercept=-30.0 pushes the logistic
    eta far into its floor). With targets averaging in the tens, the probability that
    ANY of 50,000 independent draws records even a single catch is astronomically
    small -- at seed=42 a real run produced exactly zero catches in all 50,000 draws.
    This is a real, reproducible-for-this-seed outcome, not a probabilistic assertion.
    """
    from datetime import date
    rate_r, yards_r = _rate_ratings(), _yards_ratings()
    catch_r_zero = catch_model.CatchRatings(
        players=["P1"], player_name={"P1": "P1"}, position_group={"P1": "WR"},
        ability={"P1": 0.1}, teams=["T0"], defense={"T0": -0.05},
        position_intercept={"WR": -30.0}, intercept_fallback=-30.0,
        air_yards_coef=-0.08, home_field=0.05,
        as_of=date(2025, 9, 1), n_targets=500, target_counts={"P1": 60},
    )

    samples = composition.simulate_rec_yds(
        rate_r, catch_r_zero, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=8.5, trailing_share=0.25, n_draws=50_000, seed=42)

    assert np.all(samples == 0.0)


def test_simulate_rec_yds_zero_draws_returns_empty_array():
    rate_r, catch_r, yards_r = _rate_ratings(), _catch_ratings(), _yards_ratings()
    samples = composition.simulate_rec_yds(
        rate_r, catch_r, yards_r, "P1", "WR", "T0", True,
        trailing_air_yards=8.5, trailing_share=0.25, n_draws=0, seed=1)
    assert samples.shape == (0,)
