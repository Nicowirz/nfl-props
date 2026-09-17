import numpy as np
import pytest

from nfl_props import fantasy
from nfl_props.markets import mean_yards


def test_simulate_player_yardage_only_matches_expected_mean():
    mu, sigma = 4.5, 0.3
    expected_yards = mean_yards(mu, sigma)
    proj = fantasy.simulate_player("P1", {"rec_yds": (mu, sigma)}, {}, n_samples=200_000, seed=1)
    assert proj.mean == pytest.approx(expected_yards * fantasy.PPR_SCORING["rec_yds"], rel=0.03)


def test_simulate_player_combines_yardage_and_receptions():
    mu, sigma = 4.5, 0.3
    lam, disp = 5.0, 1.0
    proj = fantasy.simulate_player("P1", {"rec_yds": (mu, sigma)}, {"receptions": (lam, disp)},
                                   n_samples=200_000, seed=2)
    expected = mean_yards(mu, sigma) * fantasy.PPR_SCORING["rec_yds"] + lam * fantasy.PPR_SCORING["receptions"]
    assert proj.mean == pytest.approx(expected, rel=0.03)


def test_quasi_poisson_matches_plain_poisson_when_dispersion_is_one():
    rng = np.random.default_rng(3)
    samples = fantasy._sample_quasi_poisson(rng, lam=4.0, dispersion=1.0, n_samples=200_000)
    assert samples.mean() == pytest.approx(4.0, abs=0.05)
    assert samples.var() == pytest.approx(4.0, rel=0.1)


def test_quasi_poisson_inflates_variance_when_overdispersed():
    rng = np.random.default_rng(4)
    samples = fantasy._sample_quasi_poisson(rng, lam=4.0, dispersion=2.0, n_samples=200_000)
    assert samples.mean() == pytest.approx(4.0, rel=0.05)
    assert samples.var() == pytest.approx(8.0, rel=0.1)  # dispersion * lam


def test_prob_a_over_b_favors_higher_projection():
    a = fantasy.simulate_player("A", {"rec_yds": (5.0, 0.3)}, {}, n_samples=100_000, seed=5)
    b = fantasy.simulate_player("B", {"rec_yds": (3.0, 0.3)}, {}, n_samples=100_000, seed=6)
    assert fantasy.prob_a_over_b(a, b) > 0.9


def test_prob_a_over_b_is_near_half_for_identical_distributions():
    a = fantasy.simulate_player("A", {"rec_yds": (4.0, 0.3)}, {}, n_samples=100_000, seed=7)
    b = fantasy.simulate_player("B", {"rec_yds": (4.0, 0.3)}, {}, n_samples=100_000, seed=8)
    assert 0.45 < fantasy.prob_a_over_b(a, b) < 0.55


def test_rank_players_sorts_by_mean_descending():
    a = fantasy.simulate_player("A", {"rec_yds": (5.0, 0.3)}, {}, n_samples=50_000, seed=9)
    b = fantasy.simulate_player("B", {"rec_yds": (3.0, 0.3)}, {}, n_samples=50_000, seed=10)
    table = fantasy.rank_players([b, a])
    assert list(table["player_id"]) == ["A", "B"]
