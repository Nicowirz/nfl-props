import numpy as np
import pytest

from nfl_props import pace


def test_pace_adjust_identity_when_total_matches_league_average():
    mu = 3.8
    assert pace.pace_adjust(mu, predicted_total=44.0, league_avg_total=44.0, stat="rec_yds",
                            sensitivity={"rec_yds": 0.2}) == pytest.approx(mu)


def test_pace_adjust_shifts_up_for_a_higher_than_average_total():
    mu = 3.8
    adjusted = pace.pace_adjust(mu, predicted_total=55.0, league_avg_total=44.0, stat="rec_yds",
                                sensitivity={"rec_yds": 0.2})
    assert adjusted > mu
    assert adjusted == pytest.approx(mu + 0.2 * np.log(55.0 / 44.0))


def test_pace_adjust_shifts_down_for_a_lower_than_average_total():
    mu = 3.8
    adjusted = pace.pace_adjust(mu, predicted_total=33.0, league_avg_total=44.0, stat="rush_yds",
                                sensitivity={"rush_yds": 0.15})
    assert adjusted < mu


def test_pace_adjust_zero_sensitivity_is_always_a_no_op():
    mu = 3.8
    for total in (20.0, 44.0, 70.0):
        assert pace.pace_adjust(mu, predicted_total=total, league_avg_total=44.0, stat="pass_yds",
                                sensitivity={"pass_yds": 0.0}) == pytest.approx(mu)


def test_pace_adjust_uses_module_default_sensitivity_when_not_given():
    # module-default SENSITIVITY starts at 0.0 for every stat until Task 4 calibrates
    # real values, so this is a no-op for now -- this test just proves the default path
    # (no explicit sensitivity argument) wires through to pace.SENSITIVITY correctly.
    mu = 3.8
    assert pace.pace_adjust(mu, predicted_total=99.0, league_avg_total=44.0,
                            stat="rec_yds") == pytest.approx(mu)
