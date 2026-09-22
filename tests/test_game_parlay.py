import pytest

from nfl_props.game_markets import moneyline_prob, prob_margin_over, prob_total_over
from nfl_props.parlay import (GameLeg, build_game_parlays, evaluate_game_legs,
                              game_legs_from_csv, model_book_gap)


def _distributions():
    # SEA home vs NE away: margin mu=+3 (SEA favored by 3), total mu=44.5
    return {("SEA", "NE"): (3.0, 13.0, 44.5, 10.0)}


def test_evaluate_game_legs_moneyline_pure_model():
    legs = [GameLeg("SEA", "NE", "moneyline", "home", None, 1.91)]
    evals = evaluate_game_legs(legs, _distributions(), market_weight=0.0)
    e = evals[0]
    expected_p = moneyline_prob(3.0, 13.0)
    assert e.p_model == pytest.approx(expected_p)
    assert e.p == pytest.approx(expected_p)
    assert e.p_book is None


def test_evaluate_game_legs_spread_respects_sign_convention():
    # spread_line = +3 means SEA (home) favored by 3; home covers iff margin > 3
    legs = [GameLeg("SEA", "NE", "spread", "home", 3.0, 1.91)]
    evals = evaluate_game_legs(legs, _distributions(), market_weight=0.0)
    expected_p = prob_margin_over(3.0, 13.0, 3.0)
    assert evals[0].p_model == pytest.approx(expected_p)
    assert expected_p == pytest.approx(0.5, abs=1e-6)  # mu equals the line -> even money


def test_evaluate_game_legs_total_devigs_when_both_sides_present():
    legs = [
        GameLeg("SEA", "NE", "total", "over", 44.5, 1.95),
        GameLeg("SEA", "NE", "total", "under", 44.5, 1.87),
    ]
    evals = evaluate_game_legs(legs, _distributions(), market_weight=1.0)  # pure book
    over_eval = next(e for e in evals if e.leg.selection == "over")
    under_eval = next(e for e in evals if e.leg.selection == "under")
    assert over_eval.p_book == pytest.approx(over_eval.p)
    assert over_eval.p + under_eval.p == pytest.approx(1.0)


def test_build_game_parlays_excludes_same_game_same_market():
    legs = [
        GameLeg("SEA", "NE", "moneyline", "home", None, 1.91),
        GameLeg("SEA", "NE", "moneyline", "away", None, 1.91),
        GameLeg("KC", "LAC", "moneyline", "home", None, 1.91),
    ]
    dists = {("SEA", "NE"): (3.0, 13.0, 44.5, 10.0), ("KC", "LAC"): (5.0, 13.0, 47.0, 10.0)}
    evals = evaluate_game_legs(legs, dists, market_weight=0.0)
    parlays = build_game_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0)
    for pl in parlays:
        games = {(le.leg.home_team, le.leg.away_team, le.leg.market) for le in pl.legs}
        assert len(games) == len(pl.legs)


def test_spread_label_shows_favorite_with_minus_sign():
    home_favored = GameLeg("PHI", "WAS", "spread", "home", 5.5, 1.91)
    assert "-5.5" in home_favored.label
    away_leg_same_line = GameLeg("PHI", "WAS", "spread", "away", 5.5, 1.91)
    assert "+5.5" in away_leg_same_line.label
    away_favored = GameLeg("NYG", "DAL", "spread", "home", -3.0, 1.91)
    assert "+3" in away_favored.label  # home is the +3 underdog
    away_favored_pick = GameLeg("NYG", "DAL", "spread", "away", -3.0, 1.91)
    assert "-3" in away_favored_pick.label  # away is the -3 favorite


def test_game_legs_from_csv(tmp_path):
    p = tmp_path / "game_odds.csv"
    p.write_text("home_team,away_team,market,selection,line,odds\n"
                 "SEA,NE,moneyline,home,,+150\n"
                 "SEA,NE,moneyline,away,,-180\n"
                 "SEA,NE,spread,home,3.0,-110\n"
                 "SEA,NE,total,over,44.5,-110\n")
    legs = game_legs_from_csv(str(p))
    assert len(legs) == 4
    kinds = {(lg.market, lg.selection) for lg in legs}
    assert ("moneyline", "home") in kinds and ("moneyline", "away") in kinds
    assert ("spread", "home") in kinds
    assert ("total", "over") in kinds


def test_game_legs_from_csv_rejects_unknown_market(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("home_team,away_team,market,selection,line,odds\nSEA,NE,touchdown,home,,-110\n")
    with pytest.raises(ValueError):
        game_legs_from_csv(str(p))


def test_game_legs_from_csv_requires_line_for_spread(tmp_path):
    p = tmp_path / "no_line.csv"
    p.write_text("home_team,away_team,market,selection,line,odds\nSEA,NE,spread,home,,-110\n")
    with pytest.raises(ValueError):
        game_legs_from_csv(str(p))


def test_build_game_parlays_excludes_implausible_model_book_disagreement():
    legs = [
        GameLeg("SEA", "NE", "moneyline", "home", None, 1.91),
        GameLeg("SEA", "NE", "moneyline", "away", None, 1.91),
        GameLeg("KC", "LAC", "moneyline", "home", None, 5.0),
        GameLeg("KC", "LAC", "moneyline", "away", None, 1.10),
    ]
    dists = {
        ("SEA", "NE"): (3.0, 13.0, 44.5, 10.0),   # model ~= book here (both near even)
        ("KC", "LAC"): (25.0, 13.0, 47.0, 10.0),  # model thinks KC is a lock; book disagrees
    }
    evals = evaluate_game_legs(legs, dists, market_weight=0.0)
    suspect = next(e for e in evals if e.leg.home_team == "KC" and e.leg.selection == "home")
    normal = next(e for e in evals if e.leg.home_team == "SEA" and e.leg.selection == "home")
    assert model_book_gap(suspect) > 0.30
    assert suspect.edge > normal.edge

    parlays = build_game_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0, max_model_gap=0.30)
    for pl in parlays:
        assert all(le.leg.home_team != "KC" for le in pl.legs)
