import pytest

from nfl_props.markets import prob_over
from nfl_props.parlay import Leg, build_parlays, evaluate_legs, legs_from_csv


def _mu_sigma():
    # mu/sigma chosen so prob_over(mu, sigma, 250) ~= 0.55, a clean edge vs -110 odds
    return {("Josh Allen", "pass_yds"): (5.55, 0.30)}


def test_evaluate_legs_pure_model():
    legs = [Leg("Josh Allen", "pass_yds", "over", 250.0, 1.91)]
    evals = evaluate_legs(legs, _mu_sigma(), market_weight=0.0)
    e = evals[0]
    expected_p = prob_over(5.55, 0.30, 250.0)
    assert e.p_model == pytest.approx(expected_p)
    assert e.p == pytest.approx(expected_p)
    assert e.p_book is None
    assert e.implied == pytest.approx(1 / 1.91)
    assert e.edge == pytest.approx(expected_p - 1 / 1.91)


def test_evaluate_legs_devigs_when_both_sides_present():
    legs = [
        Leg("Josh Allen", "pass_yds", "over", 250.0, 1.95),
        Leg("Josh Allen", "pass_yds", "under", 250.0, 1.87),
    ]
    evals = evaluate_legs(legs, _mu_sigma(), market_weight=1.0)  # pure book
    over_eval = next(e for e in evals if e.leg.selection == "over")
    under_eval = next(e for e in evals if e.leg.selection == "under")
    assert over_eval.p_book == pytest.approx(over_eval.p)
    assert over_eval.p + under_eval.p == pytest.approx(1.0)
    assert over_eval.devig_exact and under_eval.devig_exact


def test_build_parlays_excludes_contradictory_legs():
    legs = [
        Leg("Josh Allen", "pass_yds", "over", 250.0, 1.91),
        Leg("Josh Allen", "pass_yds", "under", 250.0, 1.91),
        Leg("CMC", "rush_yds", "over", 80.0, 1.91),
    ]
    mu_sigma = {("Josh Allen", "pass_yds"): (5.6, 0.3), ("CMC", "rush_yds"): (4.5, 0.35)}
    evals = evaluate_legs(legs, mu_sigma, market_weight=0.0)
    parlays = build_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0)
    for pl in parlays:
        keys = {le.leg.key() for le in pl.legs}
        assert len(keys) == len(pl.legs)


def test_build_parlays_ranks_by_ev():
    legs = [Leg("Josh Allen", "pass_yds", "over", 200.0, 2.5),
            Leg("CMC", "rush_yds", "over", 40.0, 2.2)]
    mu_sigma = {("Josh Allen", "pass_yds"): (5.6, 0.3), ("CMC", "rush_yds"): (4.5, 0.35)}
    evals = evaluate_legs(legs, mu_sigma, market_weight=0.0)
    parlays = build_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0)
    assert len(parlays) == 1
    assert parlays[0].ev == pytest.approx(parlays[0].prob * parlays[0].odds - 1)
    assert parlays[0].kelly >= 0


def test_legs_from_csv(tmp_path):
    p = tmp_path / "odds.csv"
    p.write_text("player,stat,line,over_odds,under_odds\n"
                 "Josh Allen,pass_yds,250.5,-115,-105\n"
                 "CMC,rushing,80.5,-110,\n")
    legs = legs_from_csv(str(p))
    assert len(legs) == 3  # Allen over+under, CMC over only
    kinds = {(lg.player, lg.stat, lg.selection) for lg in legs}
    assert ("Josh Allen", "pass_yds", "over") in kinds
    assert ("Josh Allen", "pass_yds", "under") in kinds
    assert ("CMC", "rush_yds", "over") in kinds
    assert ("CMC", "rush_yds", "under") not in kinds


def test_legs_from_csv_rejects_unknown_stat(tmp_path):
    p = tmp_path / "odds.csv"
    p.write_text("player,stat,line,over_odds,under_odds\nX,touchdowns,1.5,-110,-110\n")
    with pytest.raises(ValueError):
        legs_from_csv(str(p))
