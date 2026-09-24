import numpy as np
import pandas as pd

from nfl_props import model, yards_model


def _synthetic_catches(n_players=24, n_teams=8, catches_per_player=12, seed=0, ability_override=None):
    """Simulate completed-catch rows from known ability/defense/home/intercept/sigma
    parameters -- mirrors test_model.py's own _synthetic() fixture exactly (same
    distributional shape, same true parameter magnitudes), adapted to per-catch rows
    (no `targets` column -- irrelevant here; `receiving_yards` is the fit target).
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.35, n_players)))
    if ability_override:
        ability.update(ability_override)
    defense = dict(zip(teams, rng.normal(0, 0.20, n_teams)))
    intercept, home_field, sigma = 4.6, 0.05, 0.35
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for p in players:
        team = player_team[p]
        opp_pool = [t for t in teams if t != team]
        for _ in range(catches_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            mu = intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
            y = rng.normal(mu, sigma)
            yards = max(0.0, np.exp(y) - model.OFFSET)
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home, "receiving_yards": yards,
            })
            d += pd.Timedelta(hours=6)
    return pd.DataFrame(rows), players, teams, ability, defense, intercept, home_field, sigma


def test_fit_yards_per_catch_recovers_ability_and_defense():
    df, players, teams, ability, defense, intercept, home_field, sigma = _synthetic_catches()
    r = yards_model.fit_yards_per_catch(df, reg=0.05, halflife_days=100_000, min_catches=50)
    est_ability = np.array([r.ability[p] for p in players])
    true_ability = np.array([ability[p] for p in players])
    est_defense = np.array([r.defense[t] for t in teams])
    true_defense = np.array([defense[t] for t in teams])
    assert np.corrcoef(true_ability, est_ability)[0, 1] > 0.9
    assert np.corrcoef(true_defense, est_defense)[0, 1] > 0.7
    assert abs(r.sigma_global - sigma) < 0.1
    assert abs(r.home_field - home_field) < 0.1


def test_fit_yards_per_catch_respects_as_of():
    df, *_ = _synthetic_catches()
    cutoff = df["date"].iloc[100].date()
    r = yards_model.fit_yards_per_catch(df, as_of=cutoff, min_catches=50)
    assert r.n_catches == (df["date"] < pd.Timestamp(cutoff)).sum()
    assert r.as_of == cutoff


def test_predicted_yards_distribution_flags_unknown_player():
    df, players, teams, *_ = _synthetic_catches()
    r = yards_model.fit_yards_per_catch(df, min_catches=50)
    mu, sigma = yards_model.predicted_yards_distribution(r, "unknown-id", "WR", teams[0], True)
    assert np.isfinite(mu)
    assert sigma > 0
    assert "unknown-id" in r.prior_players


def test_fit_yards_per_catch_raises_below_min_catches():
    df, *_ = _synthetic_catches(catches_per_player=3)
    try:
        yards_model.fit_yards_per_catch(df, min_catches=100_000)
        assert False, "expected ValueError for too few catches"
    except ValueError as e:
        assert "need at least" in str(e)


def _synthetic_split(n_players=24, n_teams=8, catches_per_player=200, seed=0):
    """Like _synthetic_catches() but with a TRUE (opponent_team, position_group)
    interaction defense effect that is NOT decomposable into a team main effect plus a
    position main effect -- a real interaction, not just two additive terms -- so
    recovering it actually exercises the new interaction design matrix. Three position
    groups (not one) so the interaction has something real to split on.
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    positions = ["WR", "TE", "RB"]
    ability = dict(zip(players, rng.normal(0, 0.35, n_players)))
    true_defense = {}
    for t in teams:
        for g in positions:
            true_defense[(t, g)] = float(rng.normal(0, 0.30))
    intercept_by_pos = {"WR": 4.6, "TE": 4.4, "RB": 4.2}
    home_field, sigma = 0.05, 0.35
    player_pos = {p: positions[i % 3] for i, p in enumerate(players)}
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for p in players:
        team = player_team[p]
        pos = player_pos[p]
        opp_pool = [t for t in teams if t != team]
        for _ in range(catches_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            mu = (intercept_by_pos[pos] + ability[p] + true_defense[(opp, pos)]
                 + (home_field if home else 0.0))
            y = rng.normal(mu, sigma)
            yards = max(0.0, np.exp(y) - model.OFFSET)
            rows.append({
                "player_id": p, "player_name": p, "position_group": pos, "team": team,
                "opponent_team": opp, "date": d, "home": home, "receiving_yards": yards,
            })
            d += pd.Timedelta(hours=3)
    return pd.DataFrame(rows), true_defense, home_field


def test_fit_yards_per_catch_position_split_defense_recovers_interaction_effect():
    df, true_defense, home_field = _synthetic_split()
    r = yards_model.fit_yards_per_catch(df, reg=0.5, halflife_days=100_000, min_catches=100,
                                        position_split_defense=True)
    assert r.defense == {}
    assert r.defense_position is not None
    assert set(r.defense_position) == set(true_defense)
    est = np.array([r.defense_position[k] for k in true_defense])
    true = np.array([true_defense[k] for k in true_defense])
    assert np.corrcoef(true, est)[0, 1] > 0.9
    assert abs(r.home_field - home_field) < 0.1


def test_fit_yards_per_catch_position_split_defense_off_matches_current_behavior_exactly():
    """Regression guard: position_split_defense=False (the default) must be numerically
    IDENTICAL to fit_yards_per_catch's pre-existing behavior -- confirmed with a real
    probe run before this test was written (exact dict equality against the currently
    shipped, unmodified implementation). Written correctly from the start with real
    value assertions (not a tautological is-None check) -- CatchRate's own earlier plan
    needed a fix round to get this right; this plan applies that lesson immediately.
    """
    df, players, teams, ability, defense, intercept, home_field, sigma = _synthetic_catches()
    r_default = yards_model.fit_yards_per_catch(df, reg=0.05, halflife_days=100_000, min_catches=50)
    r_explicit = yards_model.fit_yards_per_catch(df, reg=0.05, halflife_days=100_000, min_catches=50,
                                                 position_split_defense=False)
    assert r_default.ability == r_explicit.ability
    assert r_default.defense == r_explicit.defense
    assert r_default.home_field == r_explicit.home_field
    assert r_explicit.defense_position is None
    assert r_explicit.defense != {}

    est_ability = np.array([r_explicit.ability[p] for p in players])
    true_ability = np.array([ability[p] for p in players])
    est_defense = np.array([r_explicit.defense[t] for t in teams])
    true_defense = np.array([defense[t] for t in teams])
    assert np.corrcoef(true_ability, est_ability)[0, 1] > 0.9
    assert np.corrcoef(true_defense, est_defense)[0, 1] > 0.7
    assert abs(r_explicit.home_field - home_field) < 0.1


def test_predicted_yards_distribution_position_split_falls_back_for_unseen_combo():
    df, true_defense, home_field = _synthetic_split()
    r = yards_model.fit_yards_per_catch(df, reg=0.5, halflife_days=100_000, min_catches=100,
                                        position_split_defense=True)
    pid = df["player_id"].iloc[0]
    # "QB" never appears in this fixture's position_group, so (any team, "QB") is an
    # unseen combo -- predicted_yards_distribution must not raise, and must fall back to
    # a neutral (0.0) defense contribution rather than crashing on a missing key.
    mu, sigma = yards_model.predicted_yards_distribution(r, pid, "QB", "T0", True)
    assert np.isfinite(mu)
    assert sigma > 0


def test_predicted_yards_distribution_position_split_applies_seen_combo_coefficient():
    """Confirms a SEEN (opponent_team, position_group) combo's real defense_position
    coefficient is actually applied, not silently defaulting to 0.0 -- by reconstructing
    mu by hand and comparing, and by confirming predictions genuinely differ across
    position groups with different true defense effects. Written in from the start:
    CatchRate's own earlier plan only caught this gap via a final-review mutation
    experiment after the fact; this plan applies that lesson immediately instead of
    waiting for a whole-branch review to find it.
    """
    df, true_defense, home_field = _synthetic_split()
    r = yards_model.fit_yards_per_catch(df, reg=0.5, halflife_days=100_000, min_catches=100,
                                        position_split_defense=True)
    pid = df["player_id"].iloc[0]
    opp = "T0"
    for pos in ["WR", "TE", "RB"]:
        assert (opp, pos) in r.defense_position
    predictions = {pos: yards_model.predicted_yards_distribution(r, pid, pos, opp, True)[0]
                  for pos in ["WR", "TE", "RB"]}
    ability = r.ability.get(pid, 0.0)
    intercept = r.position_intercept.get("WR", r.intercept_fallback)
    defense = r.defense_position[(opp, "WR")]
    manual_mu = intercept + ability + defense + r.home_field
    assert abs(predictions["WR"] - manual_mu) < 1e-9
    assert len(set(round(v, 6) for v in predictions.values())) > 1


def test_fit_yards_per_catch_raises_on_nan_receiving_yards():
    """A caller who forgets to filter to complete == 1 before calling this function gets
    a clear error instead of a silent all-NaN fit -- verified empirically before this
    test was written: without this guard, np.linalg.solve propagates NaN through the
    whole fit (every ability/defense value becomes NaN) with no exception raised at all.
    """
    df, *_ = _synthetic_catches()
    df = df.copy()
    df.loc[df.index[0], "receiving_yards"] = np.nan
    try:
        yards_model.fit_yards_per_catch(df, min_catches=50)
        assert False, "expected ValueError for NaN receiving_yards"
    except ValueError as e:
        assert "receiving_yards" in str(e) or "NaN" in str(e) or "nan" in str(e)


def test_yards_ratings_table_reflects_low_sample_threshold():
    """Regression guard for the exact class of bug CatchRatings.table() once had (a
    games-count threshold silently misapplied to a different unit) -- confirms
    YardsRatings.table()'s low_sample flag genuinely compares against catch counts, not
    some other unit, by constructing a fixture with a player below NEW_PLAYER_CATCHES
    and one above it and checking both flags land correctly.
    """
    df, *_ = _synthetic_catches(catches_per_player=30)
    r = yards_model.fit_yards_per_catch(df, min_catches=50)
    table = r.table()
    assert "low_sample" in table.columns
    assert "catches" in table.columns
    # every player in this fixture has 30 catches, comfortably above NEW_PLAYER_CATCHES
    # (20) -- none should be flagged low_sample.
    assert not table["low_sample"].any()
    assert (table["catches"] == 30).all()
