import numpy as np
import pandas as pd

from nfl_props import catch_model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic_catch(n_players=24, n_teams=8, targets_per_player=150, seed=0, ability_override=None,
                     true_air_yards_coef=-0.08):
    """Simulate target rows from known ability/defense/home/aDOT-sensitivity parameters,
    with true Bernoulli-distributed catch outcomes. Wider ability/defense spread (std 0.6/
    0.5) and more targets per player (150) than rate_model.py's analogous count-data
    fixtures -- a single Bernoulli trial carries far less information than a Poisson count,
    so recovering ability/defense correlation at a useful level needs more signal and more
    rows than test_rate_model.py's _synthetic_poisson() needs.
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.6, n_players)))
    if ability_override:
        ability.update(ability_override)
    defense = dict(zip(teams, rng.normal(0, 0.5, n_teams)))
    intercept, home_field = 0.6, 0.05
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for p in players:
        team = player_team[p]
        opp_pool = [t for t in teams if t != team]
        for _ in range(targets_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            air_yards = float(rng.normal(9.0, 6.0))
            eta = (intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
                  + true_air_yards_coef * air_yards)
            complete = float(rng.uniform() < _sigmoid(eta))
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home,
                "complete": complete, "air_yards": air_yards,
            })
            d += pd.Timedelta(hours=6)
    return (pd.DataFrame(rows), players, teams, ability, defense, intercept, home_field,
            true_air_yards_coef)


def test_fit_catch_rate_recovers_ability_defense_and_air_yards_coef():
    df, players, teams, ability, defense, intercept, home_field, true_coef = _synthetic_catch()
    r = catch_model.fit_catch_rate(df, reg=0.1, halflife_days=100_000, min_targets=100)
    est_ability = np.array([r.ability[p] for p in players])
    true_ability = np.array([ability[p] for p in players])
    est_defense = np.array([r.defense[t] for t in teams])
    true_defense = np.array([defense[t] for t in teams])
    assert np.corrcoef(true_ability, est_ability)[0, 1] > 0.85
    assert np.corrcoef(true_defense, est_defense)[0, 1] > 0.9
    assert abs(r.home_field - home_field) < 0.1
    assert abs(r.air_yards_coef - true_coef) < 0.03


def test_fit_catch_rate_respects_as_of():
    df, *_ = _synthetic_catch()
    cutoff = df["date"].iloc[400].date()
    r = catch_model.fit_catch_rate(df, as_of=cutoff, min_targets=100)
    assert r.n_targets == int((df["date"] < pd.Timestamp(cutoff)).sum())
    assert r.as_of == cutoff


def test_fit_catch_rate_as_of_matches_manual_prefilter():
    """Leakage regression guard (spec's 'Leakage safety' section explicitly calls for
    this): a golden fit using `as_of` internal filtering must be numerically IDENTICAL to
    manually pre-filtering the input to `date < as_of` and passing the same explicit
    `as_of` -- confirms `as_of` filtering can never accidentally include a same-day-or-
    later row. Verified for real: exact match (diff < 1e-9) on ability, defense,
    home_field, and air_yards_coef for this fixture before being written here.
    """
    df, *_ = _synthetic_catch()
    cutoff = df["date"].iloc[400].date()
    r1 = catch_model.fit_catch_rate(df, as_of=cutoff, min_targets=100)
    manual = df[df["date"] < pd.Timestamp(cutoff)]
    r2 = catch_model.fit_catch_rate(manual, as_of=cutoff, min_targets=100)
    assert set(r1.ability) == set(r2.ability)
    assert all(abs(r1.ability[p] - r2.ability[p]) < 1e-9 for p in r1.ability)
    assert all(abs(r1.defense[t] - r2.defense[t]) < 1e-9 for t in r1.defense)
    assert abs(r1.home_field - r2.home_field) < 1e-9
    assert abs(r1.air_yards_coef - r2.air_yards_coef) < 1e-9


def test_predicted_catch_rate_flags_unknown_player():
    df, players, teams, *_ = _synthetic_catch()
    r = catch_model.fit_catch_rate(df, min_targets=100)
    p = catch_model.predicted_catch_rate(r, "unknown-id", "WR", teams[0], True, air_yards=8.0)
    assert 0.0 < p < 1.0
    assert "unknown-id" in r.prior_players


def test_predicted_catch_rate_deeper_targets_are_less_likely_to_be_caught():
    """Sanity check on the fitted aDOT sensitivity's SIGN using real generated data (not a
    hardcoded coefficient assertion): deep targets must predict a lower catch probability
    than short ones, matching the true_air_yards_coef < 0 used to generate the fixture
    and the well-established real-football relationship.
    """
    df, players, teams, *_ = _synthetic_catch()
    r = catch_model.fit_catch_rate(df, min_targets=100)
    pid, opp = df["player_id"].iloc[0], df["opponent_team"].iloc[0]
    p_deep = catch_model.predicted_catch_rate(r, pid, "WR", opp, True, air_yards=25.0)
    p_short = catch_model.predicted_catch_rate(r, pid, "WR", opp, True, air_yards=2.0)
    assert p_short > p_deep


def test_fit_catch_rate_raises_below_min_targets():
    df, *_ = _synthetic_catch(targets_per_player=5)
    try:
        catch_model.fit_catch_rate(df, min_targets=100_000)
        assert False, "expected ValueError for too few targets"
    except ValueError as e:
        assert "need at least" in str(e)
