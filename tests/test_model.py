import numpy as np
import pandas as pd
import pytest

from nfl_props import model


def _synthetic(n_players=24, n_teams=8, games_per_player=12, seed=0, ability_override=None):
    """Simulate player-games from known ability/defense/home/intercept parameters.

    `ability_override` sets specific players' TRUE ability before any rows are generated
    (not after) -- e.g. to give two players an identical true ability so a later test can
    isolate the effect of sample size alone.
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
        for _ in range(games_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            mu = intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
            y = rng.normal(mu, sigma)
            yards = max(0.0, np.exp(y) - model.OFFSET)
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home,
                "receiving_yards": yards, "targets": 6.0,
                "passing_yards": 0.0, "attempts": 0.0, "rushing_yards": 0.0, "carries": 0.0,
            })
            d += pd.Timedelta(days=1)
    return pd.DataFrame(rows), players, teams, ability, defense, intercept, home_field, sigma


def test_fit_recovers_ability_and_defense():
    df, players, teams, ability, defense, intercept, home_field, sigma = _synthetic()
    r = model.fit(df, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    est_ability = np.array([r.ability[p] for p in players])
    true_ability = np.array([ability[p] for p in players])
    est_defense = np.array([r.defense[t] for t in teams])
    true_defense = np.array([defense[t] for t in teams])
    assert np.corrcoef(true_ability, est_ability)[0, 1] > 0.9
    assert np.corrcoef(true_defense, est_defense)[0, 1] > 0.7
    assert abs(r.sigma_global - sigma) < 0.1
    assert abs(r.home_field - home_field) < 0.1


def test_fit_respects_as_of():
    df, *_ = _synthetic()
    cutoff = df["date"].iloc[100].date()
    r = model.fit(df, "rec_yds", as_of=cutoff, min_games=50)
    assert r.n_games == (df["date"] < pd.Timestamp(cutoff)).sum()
    assert r.as_of == cutoff


def test_below_threshold_games_are_still_used_to_fit():
    # Fitting on qualifying games only would systematically overstate a low-volume
    # player's true output (a qualifying game is, by construction, one where he got
    # real volume) -- so a below-QUALIFY_MIN game must still enter the fit, not be
    # dropped from it. QUALIFY_MIN now only gates the min_games sanity check.
    df, *_ = _synthetic()
    df.loc[df.index[0], "targets"] = 1.0  # below QUALIFY_MIN["rec_yds"] = 2
    r = model.fit(df, "rec_yds", min_games=50)
    assert r.n_games == len(df)


def test_min_games_gate_counts_only_qualifying_rows():
    df, *_ = _synthetic(games_per_player=3)  # 24 players * 3 games = 72 total rows
    df["targets"] = 1.0  # every row now below QUALIFY_MIN["rec_yds"] = 2 -> 0 qualifying
    try:
        model.fit(df, "rec_yds", min_games=50)
        assert False, "expected a ValueError: not enough qualifying games"
    except ValueError as e:
        assert "0" in str(e)


def test_low_sample_player_is_shrunk_toward_zero():
    df, players, teams, ability, defense, *_ = _synthetic(
        games_per_player=20, ability_override={"P0": 0.6, "P1": 0.6})
    p_many, p_few = "P0", "P1"  # both have the same TRUE ability; only sample size differs
    few_idx = df[df["player_id"] == p_few].index[4:]  # keep only 4 games for p_few
    df = df.drop(few_idx)
    r = model.fit(df, "rec_yds", reg=1.0, min_games=50)
    assert abs(r.ability[p_few]) < abs(r.ability[p_many])


def test_predicted_distribution_flags_unknown_player():
    df, players, teams, *_ = _synthetic()
    r = model.fit(df, "rec_yds", min_games=50)
    mu, sigma = model.predicted_distribution(r, "unknown-id", "WR", teams[0], True)
    assert np.isfinite(mu) and sigma > 0
    assert "unknown-id" in r.prior_players


def test_table_sorted_by_ability():
    df, *_ = _synthetic()
    r = model.fit(df, "rec_yds", min_games=50)
    t = r.table()
    assert list(t["ability"]) == sorted(t["ability"], reverse=True)


def test_fit_handles_extreme_negative_yardage_without_nan():
    df, players, teams, ability, defense, intercept, home_field, sigma = _synthetic()
    bad_row = df.iloc[[0]].copy()
    bad_row["receiving_yards"] = -13.0  # a real NFL outlier (fumbled lateral / trick play)
    df = pd.concat([df, bad_row], ignore_index=True)
    r = model.fit(df, "rec_yds", min_games=50)
    assert np.isfinite(r.intercept_fallback)
    assert all(np.isfinite(v) for v in r.position_intercept.values())
    assert all(np.isfinite(v) for v in r.ability.values())
    assert all(np.isfinite(v) for v in r.defense.values())


def _synthetic_two_groups(seed=0):
    """RB group: abundant qualifying games, baseline near 90 yards. QB group: near-zero
    true baseline (mirroring a pocket passer's occasional scramble), but only one game in
    five clears the qualifying-carries threshold, so most of each QB's own games never
    enter the fit. Regression fixture for the bug where a low-sample player's ability,
    once shrunk toward 0, fell back to the OTHER group's (RB-dominated) shared intercept
    instead of his own position group's baseline.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(6)]
    defense = dict(zip(teams, rng.normal(0, 0.1, 6)))
    home_field, sigma = 0.05, 0.3
    rows = []
    d = pd.Timestamp("2024-09-01")

    rb_intercept = 4.2
    rb_ability = dict(zip([f"RB{i}" for i in range(10)], rng.normal(0, 0.2, 10)))
    for p, ab in rb_ability.items():
        team = rng.choice(teams)
        for _ in range(20):
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            mu = rb_intercept + ab + defense[opp] + (home_field if home else 0.0)
            yards = max(0.0, np.exp(rng.normal(mu, sigma)) - model.OFFSET)
            rows.append({"player_id": p, "player_name": p, "position_group": "RB", "team": team,
                        "opponent_team": opp, "date": d, "home": home,
                        "rushing_yards": yards, "carries": 15.0,
                        "passing_yards": 0.0, "attempts": 0.0, "receiving_yards": 0.0, "targets": 0.0})
            d += pd.Timedelta(days=1)

    qb_intercept = 1.2  # exp(1.2) - OFFSET ~= -6.7: usually a handful of yards or fewer
    qb_players = [f"QB{i}" for i in range(6)]
    for p in qb_players:
        team = rng.choice(teams)
        for g_i in range(20):
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            mu = qb_intercept + defense[opp] + (home_field if home else 0.0)
            yards = max(0.0, np.exp(rng.normal(mu, sigma)) - model.OFFSET)
            carries = 6.0 if g_i % 5 == 0 else 2.0  # qualifies (>=5) one game in five
            rows.append({"player_id": p, "player_name": p, "position_group": "QB", "team": team,
                        "opponent_team": opp, "date": d, "home": home,
                        "rushing_yards": yards, "carries": carries,
                        "passing_yards": 0.0, "attempts": 0.0, "receiving_yards": 0.0, "targets": 0.0})
            d += pd.Timedelta(days=1)

    return pd.DataFrame(rows), qb_players, rb_intercept, qb_intercept


def test_low_qualifying_group_does_not_inherit_other_groups_baseline():
    df, qb_players, rb_intercept, qb_intercept = _synthetic_two_groups()
    r = model.fit(df, "rush_yds", reg=5.0, min_games=50)
    # Every QB has very few qualifying games of his own, so his ability shrinks toward 0.
    # The prediction should still land near the QB group's own true baseline, not the
    # RB-dominated shared baseline the bug used to fall back on.
    mu, _ = model.predicted_distribution(r, qb_players[0], "QB", "T0", home=False)
    assert abs(mu - qb_intercept) < abs(mu - rb_intercept)
    assert mu < (rb_intercept + qb_intercept) / 2


def _synthetic_share_games(n_prior=6, seed=7):
    """One player (P1, WR, team A) with `n_prior` weekly games plus one more (the query
    game) -- a teammate (P2) fills out team A's target total each game so target_share is
    well-defined. A second, unrelated team (B) with two of its own players is included so
    the fixture also exercises the position-group fallback for a low-history player (P3,
    fewer than NEW_PLAYER_GAMES games).
    """
    rng = np.random.default_rng(seed)
    rows = []
    d = pd.Timestamp("2024-09-01")
    p1_targets = []
    for i in range(n_prior + 1):
        t1 = float(rng.integers(3, 10))
        t2 = float(rng.integers(3, 10))
        p1_targets.append((d, t1, t1 + t2))
        rows.append({"player_id": "P1", "position_group": "WR", "team": "A",
                     "opponent_team": "B", "game_id": f"g{i}", "date": d,
                     "targets": t1, "carries": 0.0, "home": True,
                     "receiving_yards": 0.0, "rushing_yards": 0.0, "passing_yards": 0.0,
                     "attempts": 0.0})
        rows.append({"player_id": "P2", "position_group": "WR", "team": "A",
                     "opponent_team": "B", "game_id": f"g{i}", "date": d,
                     "targets": t2, "carries": 0.0, "home": True,
                     "receiving_yards": 0.0, "rushing_yards": 0.0, "passing_yards": 0.0,
                     "attempts": 0.0})
        d += pd.Timedelta(days=7)
    # P3 on team B: only 2 games -- below NEW_PLAYER_GAMES, must use the fallback.
    for i in range(2):
        rows.append({"player_id": "P3", "position_group": "WR", "team": "B",
                     "opponent_team": "A", "game_id": f"g{i}", "date": pd.Timestamp("2024-09-01") + pd.Timedelta(days=7 * i),
                     "targets": 5.0, "carries": 0.0, "home": False,
                     "receiving_yards": 0.0, "rushing_yards": 0.0, "passing_yards": 0.0,
                     "attempts": 0.0})
        rows.append({"player_id": "P4", "position_group": "WR", "team": "B",
                     "opponent_team": "A", "game_id": f"g{i}", "date": pd.Timestamp("2024-09-01") + pd.Timedelta(days=7 * i),
                     "targets": 3.0, "carries": 0.0, "home": False,
                     "receiving_yards": 0.0, "rushing_yards": 0.0, "passing_yards": 0.0,
                     "attempts": 0.0})
    return pd.DataFrame(rows), p1_targets


def test_add_trailing_share_requires_game_id():
    df = pd.DataFrame([{"player_id": "P1", "position_group": "WR", "team": "A",
                        "date": pd.Timestamp("2024-09-01"), "targets": 5.0}])
    try:
        model.add_trailing_share(df, "rec_yds")
        assert False, "expected a ValueError for a missing game_id column"
    except ValueError as e:
        assert "game_id" in str(e)


def test_add_trailing_share_matches_independent_weighted_calculation():
    df, p1_targets = _synthetic_share_games(n_prior=6)
    out = model.add_trailing_share(df, "rec_yds", halflife_days=180.0)
    # P1's LAST game (index n_prior, the query game): compute the expected trailing
    # share independently from the first n_prior games' (date, own_targets, team_total).
    query_date = p1_targets[-1][0]
    prior = p1_targets[:-1]
    shares = np.array([t / total for _, t, total in prior])
    days_ago = np.array([(query_date - d).days for d, _, _ in prior], dtype=float)
    w = 0.5 ** (days_ago / 180.0)
    expected = float(np.sum(w * shares) / np.sum(w))
    row = out[(out["player_id"] == "P1") & (out["game_id"] == f"g{len(p1_targets) - 1}")].iloc[0]
    assert row["trailing_share"] == pytest.approx(expected)


def test_add_trailing_share_never_uses_the_rows_own_game():
    # P1's FIRST game (g0) has zero prior games -- its trailing_share must come from the
    # low-history fallback, never from g0's own target/team-total values.
    df, p1_targets = _synthetic_share_games(n_prior=6)
    out = model.add_trailing_share(df, "rec_yds", halflife_days=180.0)
    own_share = p1_targets[0][1] / p1_targets[0][2]
    row = out[(out["player_id"] == "P1") & (out["game_id"] == "g0")].iloc[0]
    assert row["trailing_share"] != pytest.approx(own_share)


def test_add_trailing_share_low_history_uses_group_fallback():
    df, _ = _synthetic_share_games(n_prior=6)
    out = model.add_trailing_share(df, "rec_yds", halflife_days=180.0)
    # P3 has only 2 games (both team B, position_group WR) -- below NEW_PLAYER_GAMES(4),
    # so both of P3's rows must equal the WR group's fallback average (computed only
    # from players who DO have >= 4 prior games -- i.e. P1's later rows). Both P3 rows
    # should get the identical fallback value (not each other's own thin data).
    p3_rows = out[out["player_id"] == "P3"]
    assert p3_rows["trailing_share"].nunique() == 1


def test_add_trailing_share_never_returns_nan():
    df, _ = _synthetic_share_games(n_prior=6)
    out = model.add_trailing_share(df, "rec_yds", halflife_days=180.0)
    assert out["trailing_share"].notna().all()


def test_current_trailing_share_matches_independent_calculation():
    df, p1_targets = _synthetic_share_games(n_prior=6)
    as_of = p1_targets[-1][0] + pd.Timedelta(days=7)  # one week after P1's last game
    result = model.current_trailing_share(df, "rec_yds", "P1", as_of=as_of.date(), halflife_days=180.0)
    shares = np.array([t / total for _, t, total in p1_targets])
    days_ago = np.array([(as_of - d).days for d, _, _ in p1_targets], dtype=float)
    w = 0.5 ** (days_ago / 180.0)
    expected = float(np.sum(w * shares) / np.sum(w))
    assert result == pytest.approx(expected)


def test_current_trailing_share_none_for_unknown_player():
    df, _ = _synthetic_share_games(n_prior=6)
    assert model.current_trailing_share(df, "rec_yds", "nobody") is None


def test_current_trailing_share_none_below_threshold():
    df, _ = _synthetic_share_games(n_prior=6)
    # P3 only has 2 games total -- below NEW_PLAYER_GAMES(4).
    assert model.current_trailing_share(df, "rec_yds", "P3") is None
