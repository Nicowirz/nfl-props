import pytest

from nfl_props.markets import prob_over
from nfl_props.parlay import Leg, build_parlays, evaluate_legs, legs_from_csv, model_book_gap


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


def test_build_parlays_excludes_different_line_same_player_stat():
    # Same player/stat at different lines (over 300 + under 200) are correlated/potentially
    # mutually exclusive and must not be recommended together, even though Leg.key() (which
    # requires an exact line match for devig grouping) treats them as distinct legs.
    legs = [
        Leg("Josh Allen", "pass_yds", "over", 300.0, 2.0),
        Leg("Josh Allen", "pass_yds", "under", 200.0, 2.0),
    ]
    mu_sigma = {("Josh Allen", "pass_yds"): (5.6, 0.3)}
    evals = evaluate_legs(legs, mu_sigma, market_weight=0.0)
    parlays = build_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0)
    # The only possible 2-leg combo is these two same-player/stat legs, so no parlay
    # should survive at all -- and even if others existed, none may pair these two.
    assert parlays == []
    for pl in parlays:
        player_stats = {(le.leg.player, le.leg.stat) for le in pl.legs}
        assert len(player_stats) == len(pl.legs)


def test_build_parlays_ranks_by_ev():
    legs = [Leg("Josh Allen", "pass_yds", "over", 200.0, 2.5),
            Leg("CMC", "rush_yds", "over", 40.0, 2.2)]
    mu_sigma = {("Josh Allen", "pass_yds"): (5.6, 0.3), ("CMC", "rush_yds"): (4.5, 0.35)}
    evals = evaluate_legs(legs, mu_sigma, market_weight=0.0)
    # max_model_gap=1.0: this test's synthetic odds produce a large pure-model edge on
    # purpose (to get a clean, unambiguous ranking signal); the plausibility filter is
    # exercised separately in test_build_parlays_excludes_implausible_model_book_disagreement.
    parlays = build_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0, max_model_gap=1.0)
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


def test_model_book_gap_uses_devigged_book_prob_when_available():
    legs = [
        Leg("Josh Allen", "pass_yds", "over", 250.0, 2.0),
        Leg("Josh Allen", "pass_yds", "under", 250.0, 2.0),
    ]
    # mu/sigma chosen so the model thinks "over" is far more likely than the devigged
    # 50/50 book price implies.
    evals = evaluate_legs(legs, {("Josh Allen", "pass_yds"): (6.5, 0.3)}, market_weight=0.0)
    over_eval = next(e for e in evals if e.leg.selection == "over")
    assert over_eval.p_book is not None
    assert model_book_gap(over_eval) == pytest.approx(abs(over_eval.p_model - over_eval.p_book))


def test_model_book_gap_falls_back_to_raw_implied_when_no_devig():
    legs = [Leg("Josh Allen", "pass_yds", "over", 250.0, 1.91)]
    evals = evaluate_legs(legs, {("Josh Allen", "pass_yds"): (5.55, 0.30)}, market_weight=0.0)
    e = evals[0]
    assert e.p_book is None
    assert model_book_gap(e) == pytest.approx(abs(e.p_model - e.implied))


def test_build_parlays_excludes_implausible_model_book_disagreement():
    # Leg A: model and (devigged) book roughly agree -- a normal, trustworthy edge.
    # Leg B: model thinks "over" is far likelier than the book's devigged price implies --
    # a >30pp disagreement, the kind a thin/stale market or a model blind spot produces,
    # not genuine insight -- and must not be allowed to win purely on raw edge.
    legs = [
        Leg("CMC", "rush_yds", "over", 80.0, 1.87),
        Leg("CMC", "rush_yds", "under", 80.0, 1.95),
        Leg("Suspect Player", "pass_yds", "over", 150.0, 1.87),
        Leg("Suspect Player", "pass_yds", "under", 150.0, 1.95),
    ]
    mu_sigma = {
        ("CMC", "rush_yds"): (4.5, 0.35),        # model ~= book here
        ("Suspect Player", "pass_yds"): (6.5, 0.3),  # model way above the devigged book price
    }
    evals = evaluate_legs(legs, mu_sigma, market_weight=0.0)
    suspect = next(e for e in evals if e.leg.player == "Suspect Player" and e.leg.selection == "over")
    normal = next(e for e in evals if e.leg.player == "CMC" and e.leg.selection == "over")
    assert model_book_gap(suspect) > 0.30
    assert suspect.edge > normal.edge  # the implausible leg looks "better" on raw edge alone

    parlays = build_parlays(evals, min_legs=2, max_legs=2, min_edge=-1.0, max_model_gap=0.30)
    for pl in parlays:
        assert all(le.leg.player != "Suspect Player" for le in pl.legs)
