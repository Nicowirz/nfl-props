import pytest

from nfl_props.markets import devig, fair_odds, mean_yards, median_yards, parse_odds, prob_over, to_american


def test_prob_over_half_at_median():
    mu, sigma = 5.0, 0.4
    line = median_yards(mu)
    assert prob_over(mu, sigma, line) == pytest.approx(0.5, abs=1e-6)


def test_prob_over_monotone_decreasing_in_line():
    mu, sigma = 5.0, 0.4
    probs = [prob_over(mu, sigma, ln) for ln in (50, 75, 100, 125, 150)]
    assert probs == sorted(probs, reverse=True)


def test_mean_greater_than_median_for_lognormal():
    mu, sigma = 5.0, 0.5
    assert mean_yards(mu, sigma) > median_yards(mu)


def test_devig():
    p = devig([1.91, 1.91])
    assert sum(p) == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[1])


def test_fair_odds_roundtrip():
    assert fair_odds(0.5) == pytest.approx(2.0)


def test_odds_conversions():
    assert parse_odds("+150") == pytest.approx(2.5)
    assert parse_odds("-110") == pytest.approx(1.909, abs=1e-3)
    assert parse_odds("1.91") == pytest.approx(1.91)
    assert parse_odds(185.0) == pytest.approx(2.85)
    assert parse_odds(-240.0) == pytest.approx(1.4167, abs=1e-3)
    with pytest.raises(ValueError):
        parse_odds("0.5")
    assert to_american(2.5) == "+150"
    assert to_american(1.5) == "-200"
