import pytest

from nfl_props.game_markets import moneyline_prob, prob_margin_over, prob_total_over


def test_prob_margin_over_half_at_zero_for_pickem():
    # pick'em game: mu=0 means no favorite, so P(home wins) should be 50%
    assert prob_margin_over(0.0, 13.0, 0.0) == pytest.approx(0.5, abs=1e-6)


def test_prob_margin_over_monotone_decreasing_in_line():
    probs = [prob_margin_over(3.0, 13.0, ln) for ln in (-10, -3, 0, 3, 10)]
    assert probs == sorted(probs, reverse=True)


def test_moneyline_prob_favors_home_when_mu_positive():
    assert moneyline_prob(7.0, 13.0) > 0.5
    assert moneyline_prob(-7.0, 13.0) < 0.5
    assert moneyline_prob(0.0, 13.0) == pytest.approx(0.5, abs=1e-6)


def test_prob_total_over_half_at_mu():
    assert prob_total_over(44.0, 10.0, 44.0) == pytest.approx(0.5, abs=1e-6)


def test_prob_total_over_monotone_decreasing_in_line():
    probs = [prob_total_over(44.0, 10.0, ln) for ln in (30, 40, 44, 50, 60)]
    assert probs == sorted(probs, reverse=True)
