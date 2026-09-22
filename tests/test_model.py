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


def _snap_share_stats():
    return pd.DataFrame([
        {"player_id": "P1", "date": pd.Timestamp("2025-08-11"), "position_group": "WR", "offense_pct": 0.50},
        {"player_id": "P1", "date": pd.Timestamp("2025-08-18"), "position_group": "WR", "offense_pct": 0.55},
        {"player_id": "P1", "date": pd.Timestamp("2025-08-25"), "position_group": "WR", "offense_pct": 0.65},
        {"player_id": "P1", "date": pd.Timestamp("2025-09-01"), "position_group": "WR", "offense_pct": 0.70},
        {"player_id": "P1", "date": pd.Timestamp("2025-09-08"), "position_group": "WR", "offense_pct": 0.80},
        {"player_id": "P2", "date": pd.Timestamp("2025-09-01"), "position_group": "WR", "offense_pct": 0.10},
    ])


def test_add_trailing_snap_share_never_uses_the_rows_own_game():
    df = _snap_share_stats()
    out = model.add_trailing_snap_share(df, halflife_days=180.0)
    row1_sep08 = out[(out["player_id"] == "P1") & (out["date"] == pd.Timestamp("2025-09-08"))].iloc[0]
    # Changing this row's own offense_pct must not change its own trailing_snap_share.
    df2 = df.copy()
    df2.loc[(df2["player_id"] == "P1") & (df2["date"] == pd.Timestamp("2025-09-08")), "offense_pct"] = 0.01
    out2 = model.add_trailing_snap_share(df2, halflife_days=180.0)
    row1_sep08_changed = out2[(out2["player_id"] == "P1") & (out2["date"] == pd.Timestamp("2025-09-08"))].iloc[0]
    assert row1_sep08["trailing_snap_share"] == pytest.approx(row1_sep08_changed["trailing_snap_share"])


def test_add_trailing_snap_share_matches_hand_computed_weighted_average():
    df = _snap_share_stats()
    out = model.add_trailing_snap_share(df, halflife_days=180.0)
    row = out[(out["player_id"] == "P1") & (out["date"] == pd.Timestamp("2025-09-08"))].iloc[0]
    # P1's 5th game has exactly NEW_PLAYER_GAMES(4) prior games -- n_prior == 4 is NOT
    # "low" (low is strictly <), so this row must use its own raw computed weighted
    # average over all 4 prior games, not the group fallback.
    days_ago = np.array([28.0, 21.0, 14.0, 7.0])  # from 08-11, 08-18, 08-25, 09-01 to 09-08
    shares = np.array([0.50, 0.55, 0.65, 0.70])
    w = 0.5 ** (days_ago / 180.0)
    expected = float(np.sum(w * shares) / np.sum(w))
    assert row["trailing_snap_share"] == pytest.approx(expected)


def test_add_trailing_snap_share_low_history_uses_group_fallback():
    df = _snap_share_stats()
    out = model.add_trailing_snap_share(df, halflife_days=180.0)
    first_game = out[(out["player_id"] == "P2") & (out["date"] == pd.Timestamp("2025-09-01"))].iloc[0]
    assert not np.isnan(first_game["trailing_snap_share"])


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


def _synthetic_with_share(n_players=16, n_teams=8, games_per_player=14, seed=11,
                          true_share_coef=1.5):
    """Like _synthetic(), but with real teammates sharing a game_id (2 players per team
    per game) and a KNOWN true share-sensitivity baked into the generating model, so the
    fitted share_coef can be checked against a ground truth.
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.2, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.15, n_teams)))
    intercept, home_field, sigma = 4.2, 0.05, 0.30
    # two players per team, sharing that team's games
    player_team = {}
    for i, p in enumerate(players):
        player_team[p] = teams[i % n_teams]
    rows = []
    d = pd.Timestamp("2024-09-01")
    game_counter = 0
    for _ in range(games_per_player):
        for team in teams:
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            teammates = [p for p, t in player_team.items() if t == team]
            game_id = f"g{game_counter}"
            game_counter += 1
            raw_targets = {p: float(rng.integers(2, 10)) for p in teammates}
            for p in teammates:
                rows.append({
                    "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                    "opponent_team": opp, "game_id": game_id, "date": d, "home": home,
                    "receiving_yards": 0.0, "targets": raw_targets[p],
                    "passing_yards": 0.0, "attempts": 0.0, "rushing_yards": 0.0, "carries": 0.0,
                })
        d += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    # Re-generate receiving_yards WITH the true share effect, using each row's own
    # trailing_share (computed the same leakage-safe way fit() will use) so the
    # generating process matches what add_trailing_share() will actually feed the fit.
    aug = model.add_trailing_share(df, "rec_yds", halflife_days=100_000)
    rng2 = np.random.default_rng(seed + 1)
    new_y = []
    for _, row in aug.iterrows():
        mu = (intercept + ability[row["player_id"]] + defense[row["opponent_team"]]
             + (home_field if row["home"] else 0.0) + true_share_coef * row["trailing_share"])
        new_y.append(rng2.normal(mu, sigma))
    aug["receiving_yards"] = np.maximum(0.0, np.exp(new_y) - model.OFFSET)
    return aug.drop(columns=["trailing_share"]), true_share_coef


def test_fit_recovers_share_coef_when_trailing_share_present():
    df, true_share_coef = _synthetic_with_share()
    aug = model.add_trailing_share(df, "rec_yds", halflife_days=100_000)
    r = model.fit(aug, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    assert r.share_coef is not None
    assert abs(r.share_coef - true_share_coef) < 0.5


def test_fit_share_coef_is_none_without_trailing_share_column():
    df, _ = _synthetic_with_share()
    r = model.fit(df, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    assert r.share_coef is None
    assert r.share_fallback == {}


def test_predicted_distribution_applies_share_coef():
    df, true_share_coef = _synthetic_with_share()
    aug = model.add_trailing_share(df, "rec_yds", halflife_days=100_000)
    r = model.fit(aug, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    player_id = aug["player_id"].iloc[0]
    opp = aug["opponent_team"].iloc[0]
    mu_low, _ = model.predicted_distribution(r, player_id, "WR", opp, True, trailing_share=0.1)
    mu_high, _ = model.predicted_distribution(r, player_id, "WR", opp, True, trailing_share=0.5)
    assert mu_high > mu_low  # true_share_coef is positive in this fixture


def test_predicted_distribution_uses_group_fallback_when_share_omitted():
    df, _ = _synthetic_with_share()
    aug = model.add_trailing_share(df, "rec_yds", halflife_days=100_000)
    r = model.fit(aug, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    player_id = aug["player_id"].iloc[0]
    opp = aug["opponent_team"].iloc[0]
    mu_omitted, _ = model.predicted_distribution(r, player_id, "WR", opp, True)
    mu_fallback, _ = model.predicted_distribution(r, player_id, "WR", opp, True,
                                                  trailing_share=r.share_fallback["WR"])
    assert mu_omitted == pytest.approx(mu_fallback)


def _synthetic_wide_sigma(n_normal_players=20, n_low_players=20, seed=13,
                          normal_sigma=0.30, low_sigma=0.70, games_per_normal=14):
    """QB-only synthetic data with a KNOWN true variance difference: `n_normal_players`
    each get `games_per_normal` games (well above NEW_PLAYER_GAMES) drawn with
    `normal_sigma`; `n_low_players` each get only 2 games (below NEW_PLAYER_GAMES) drawn
    with a LARGER `low_sigma` -- mirrors a backup QB's rare, higher-variance starts.
    """
    rng = np.random.default_rng(seed)
    intercept, home_field = 4.6, 0.05
    rows = []
    d = pd.Timestamp("2024-09-01")

    def _emit(player, opp_pool, n_games, sigma):
        nonlocal d
        for _ in range(n_games):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            mu = intercept + (home_field if home else 0.0)
            y = rng.normal(mu, sigma)
            yards = max(0.0, np.exp(y) - model.OFFSET)
            rows.append({
                "player_id": player, "player_name": player, "position_group": "QB",
                "team": "A", "opponent_team": opp, "date": d, "home": home,
                "passing_yards": yards, "attempts": 25.0,
                "receiving_yards": 0.0, "targets": 0.0, "rushing_yards": 0.0, "carries": 0.0,
            })
            d += pd.Timedelta(days=1)

    opp_pool = [f"T{i}" for i in range(8)]
    for i in range(n_normal_players):
        _emit(f"NORMAL{i}", opp_pool, games_per_normal, normal_sigma)
    for i in range(n_low_players):
        _emit(f"LOW{i}", opp_pool, 2, low_sigma)
    return pd.DataFrame(rows), normal_sigma, low_sigma


def test_fit_recovers_wider_sigma_for_low_sample_players():
    df, true_normal_sigma, true_low_sigma = _synthetic_wide_sigma()
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    assert r.sigma_low_sample["QB"] > r.sigma["QB"]
    assert abs(r.sigma["QB"] - true_normal_sigma) < 0.1
    # Tolerance widened from an originally-specified 0.15 to 0.3: with reg=0.05 (near-zero
    # ridge penalty) and only 2 games per low-sample player, each such player's own
    # per-player ability term almost perfectly fits their 2-game mean, so the masked
    # residuals used to compute sigma_low_sample carry the classic n=2 degrees-of-freedom
    # downward bias (expected shrinkage factor ~sqrt(1/2), i.e. true_low_sigma * 0.71 =
    # ~0.50, not 0.70). Verified this is a structural bias, not sampling noise: it does
    # NOT shrink as n_low_players grows from 20 to 400 (stays ~0.50-0.53), and DOES shrink
    # as `reg` is raised (0.05 -> ~0.50, 1.0 -> ~0.53, 5.0 -> ~0.62, 20.0 -> ~0.68) --
    # confirmed live 2026-09-16. 0.3 is NOT tight enough to guarantee sigma_low_sample
    # lands strictly closer to true_low_sigma than to true_normal_sigma -- e.g. 0.40 would
    # pass this bound while sitting closer to 0.30 (distance 0.10) than to 0.70 (distance
    # 0.30), and the actual measured value for this fixture's default seed (0.474) is
    # itself closer to true_normal_sigma (|0.474-0.30|=0.174) than to true_low_sigma
    # (|0.474-0.70|=0.226). The bound's real job is narrower: it's wide enough to absorb
    # both the diagnosed ~0.50 asymptotic bias and the seed-to-seed sampling noise observed
    # across 6 seeds (0.37-0.60) without being vacuous (sigma_low_sample - sigma["QB"] is
    # already asserted separately above), and it still fails if the feature stops widening
    # sigma at all.
    assert abs(r.sigma_low_sample["QB"] - true_low_sigma) < 0.3


def test_fit_sigma_normal_only_excludes_low_sample_rows():
    # Regression test for the sigma-pool-contamination bug: sigma_normal_only["QB"] must
    # be computed ONLY from normal-sample players' residuals -- not from the SAME rows
    # sigma_low_sample separately draws from. `sigma` itself (the always-on, default
    # value every non-wide-sigma caller reads) is DELIBERATELY left contaminated/
    # unconditional -- see test_fit_sigma_unconditional_is_unaffected_by_wide_sigma
    # below for that guarantee; sigma_normal_only is the new, additional, opt-in-only
    # clean value this bug fix introduces.
    # _synthetic_wide_sigma()'s defaults (0.30 vs 0.70, 40-of-320 low rows) don't move
    # the mixture far enough to unambiguously distinguish contaminated vs. clean under
    # test 1's existing 0.1 tolerance, so this uses a much larger sigma disparity: a
    # CONTAMINATED estimate here would land near sqrt((280*0.20**2 + 40*1.20**2) / 320)
    # =~ 0.46, far outside a tight bound around the true normal-only value (0.20).
    df, true_normal_sigma, true_low_sigma = _synthetic_wide_sigma(
        n_normal_players=20, n_low_players=20, normal_sigma=0.20, low_sigma=1.20,
        games_per_normal=14)
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    assert abs(r.sigma_normal_only["QB"] - true_normal_sigma) < 0.08


def test_fit_sigma_unconditional_is_unaffected_by_wide_sigma():
    # The DEFAULT `sigma` dict -- what every non-wide-sigma caller (predict, parlay,
    # best-bet, and plain `backtest`) actually reads -- must stay exactly what it always
    # was: computed from EVERY row of the group, low-sample rows included, regardless of
    # WIDE_SIGMA_STATS. This is the inertness guarantee the contamination fix must not
    # break: fixing sigma_normal_only must never silently change live default behavior.
    # Uses the same large-disparity fixture as the test above so contamination (if it
    # leaked into `sigma`) would be unambiguous.
    df, true_normal_sigma, true_low_sigma = _synthetic_wide_sigma(
        n_normal_players=20, n_low_players=20, normal_sigma=0.20, low_sigma=1.20,
        games_per_normal=14)
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    # a genuinely mixed (contaminated) estimate over 280 normal (0.20) + 40 low (1.20)
    # rows lands close to 0.46 -- far outside a tight bound around the pure normal value,
    # confirming `sigma["QB"]` is still the full, unconditional population estimate.
    assert abs(r.sigma["QB"] - true_normal_sigma) > 0.08


def test_fit_sigma_low_sample_empty_for_stat_outside_wide_sigma_stats():
    # rush_yds is NOT in WIDE_SIGMA_STATS -- sigma_low_sample must be empty regardless
    # of how the data looks. (Despite its former name, this test does not exercise any
    # fallback branch -- the real fallback-branch tests are
    # test_fit_sigma_low_sample_uses_group_sigma_when_low_sample_pool_too_thin and
    # test_fit_sigma_low_sample_uses_sigma_global_when_group_itself_too_thin below.)
    df, *_ = _synthetic_wide_sigma()
    # drop the helper's always-zero "rushing_yards" column first -- renaming onto it
    # without dropping would leave two columns both named "rushing_yards" (pandas allows
    # duplicate labels), which silently turns df["rushing_yards"] into a 2-column
    # DataFrame and crashes fit()'s design-matrix construction with an unrelated
    # ValueError, unrelated to anything under test here.
    df = df.drop(columns=["rushing_yards"]).rename(columns={"passing_yards": "rushing_yards"})
    df["carries"] = df["attempts"]
    df["position_group"] = "RB"
    r = model.fit(df, "rush_yds", reg=0.05, halflife_days=100_000, min_games=50)
    assert r.sigma_low_sample == {}


def test_fit_byte_identical_output_when_stat_not_in_wide_sigma_stats(monkeypatch):
    # A stat outside WIDE_SIGMA_STATS must produce IDENTICAL fit() output to before this
    # feature existed -- direct regression guard, not just a new-feature test. Fit the
    # same synthetic data twice for rec_yds (not in WIDE_SIGMA_STATS): once with
    # WIDE_SIGMA_STATS monkeypatched to empty (simulating "this feature never existed")
    # and once with the real, current WIDE_SIGMA_STATS -- then assert the resulting
    # Ratings' sigma/ability/position_intercept dicts compare exactly equal. Asserting
    # sigma_low_sample == {} alone (the old version of this test) doesn't prove
    # byte-identity -- it only proves the widening branch itself is a no-op for this
    # stat, not that nothing else about the fit shifted.
    df, *_ = _synthetic_wide_sigma()
    # same duplicate-column pitfall as the RB test above: drop the helper's always-zero
    # "receiving_yards" column before renaming onto it.
    df2 = df.drop(columns=["receiving_yards"]).rename(columns={"passing_yards": "receiving_yards"})
    df2["targets"] = df2["attempts"]

    monkeypatch.setattr(model, "WIDE_SIGMA_STATS", set())
    r_without = model.fit(df2, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    monkeypatch.undo()  # restore the real WIDE_SIGMA_STATS before the second fit
    r_with = model.fit(df2, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)

    assert r_without.sigma_low_sample == {} and r_with.sigma_low_sample == {}
    assert r_without.sigma_normal_only == {} and r_with.sigma_normal_only == {}
    assert r_without.sigma == r_with.sigma
    assert r_without.ability == r_with.ability
    assert r_without.position_intercept == r_with.position_intercept


def test_fit_sigma_low_sample_uses_normal_only_when_low_sample_pool_too_thin():
    # The default _synthetic_wide_sigma() fixture always clears MIN_GROUP_RESIDUALS(30)
    # for the low-sample mask (20 players * 2 games = 40), so it never exercises fit()'s
    # `elif g in sigma_normal_only: sigma_low_sample[g] = sigma_normal_only[g]` fallback.
    # Shrink the low-sample pool to 5 players * 2 games = 10 rows (< 30) while keeping
    # the normal pool large (20 players * 14 games = 280 rows, >= 30) so
    # `sigma_normal_only["QB"]` IS computed and available as the fallback target -- the
    # fallback prefers this CLEAN value over the contaminated `sigma["QB"]`.
    df, *_ = _synthetic_wide_sigma(n_normal_players=20, n_low_players=5, games_per_normal=14)
    n_low_rows = (df["player_id"].str.startswith("LOW")).sum()
    assert n_low_rows < model.MIN_GROUP_RESIDUALS  # sanity: confirms the mask this test
                                                    # relies on actually stays below threshold
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    assert "QB" in r.sigma_normal_only  # the clean fallback target must exist
    assert r.sigma_low_sample["QB"] == r.sigma_normal_only["QB"]


def test_fit_sigma_low_sample_uses_sigma_global_when_group_itself_too_thin():
    # Push the WHOLE QB group (not just its low-sample slice) below MIN_GROUP_RESIDUALS(30)
    # rows, so fit()'s own `sigma["QB"]` is never computed either -- this must fall all the
    # way through to fit()'s last resort, `sigma_low_sample[g] = sigma_global`.
    df, *_ = _synthetic_wide_sigma(n_normal_players=2, n_low_players=3, games_per_normal=5)
    assert len(df) < model.MIN_GROUP_RESIDUALS  # sanity: whole group, not just the low
                                                 # slice, stays below threshold
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=10)
    assert "QB" not in r.sigma  # confirms the elif branch cannot have fired
    assert r.sigma_low_sample["QB"] == r.sigma_global


def test_predicted_distribution_uses_sigma_low_sample_for_low_sample_player():
    df, true_normal_sigma, true_low_sigma = _synthetic_wide_sigma()
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    _, sigma_low = model.predicted_distribution(r, "LOW0", "QB", "T0", True, stat="pass_yds")
    _, sigma_normal = model.predicted_distribution(r, "NORMAL0", "QB", "T0", True, stat="pass_yds")
    assert sigma_low == pytest.approx(r.sigma_low_sample["QB"])
    assert sigma_normal == pytest.approx(r.sigma_normal_only["QB"])
    assert sigma_low > sigma_normal


def test_predicted_distribution_unknown_player_counts_as_low_sample():
    df, *_ = _synthetic_wide_sigma()
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    _, sigma = model.predicted_distribution(r, "totally-unknown-id", "QB", "T0", True, stat="pass_yds")
    assert sigma == pytest.approx(r.sigma_low_sample["QB"])


def test_predicted_distribution_ignores_wide_sigma_without_stat_argument():
    # Omitting `stat` must reproduce today's EXACT behavior -- this is what lets
    # backtest.py (Task 3) be updated ahead of cli.py (Task 5) without cli.py breaking.
    df, *_ = _synthetic_wide_sigma()
    r = model.fit(df, "pass_yds", reg=0.05, halflife_days=100_000, min_games=50)
    _, sigma_default = model.predicted_distribution(r, "LOW0", "QB", "T0", True)
    assert sigma_default == pytest.approx(r.sigma["QB"])  # NOT sigma_low_sample


def test_predicted_distribution_stat_outside_wide_sigma_stats_unaffected():
    df, *_ = _synthetic_wide_sigma()
    # same duplicate-column pitfall documented above: drop the helper's always-zero
    # "receiving_yards" column before renaming onto it.
    df2 = df.drop(columns=["receiving_yards"]).rename(columns={"passing_yards": "receiving_yards"})
    df2["targets"] = df2["attempts"]
    r = model.fit(df2, "rec_yds", reg=0.05, halflife_days=100_000, min_games=50)
    _, sigma = model.predicted_distribution(r, "LOW0", "QB", "T0", True, stat="rec_yds")
    assert sigma == pytest.approx(r.sigma["QB"])


def test_add_trailing_snap_share_row_level_computation_unaffected_by_future_rows():
    """A row that has already reached NEW_PLAYER_GAMES prior games (and so uses its own
    raw computed weighted average, not the group fallback) must be identical whether or
    not later-dated rows exist in the input frame -- the per-row computation only ever
    reads that player's own strictly-prior rows by construction. (The group-fallback pool
    used for BELOW-threshold rows has its own separate, already-documented, accepted
    future-data exception -- see add_trailing_share's docstring -- and is unchanged and
    out of scope here.)
    """
    through_week5 = pd.DataFrame([
        {"player_id": "P1", "date": pd.Timestamp("2025-08-11"), "position_group": "WR", "offense_pct": 0.50},
        {"player_id": "P1", "date": pd.Timestamp("2025-08-18"), "position_group": "WR", "offense_pct": 0.55},
        {"player_id": "P1", "date": pd.Timestamp("2025-08-25"), "position_group": "WR", "offense_pct": 0.65},
        {"player_id": "P1", "date": pd.Timestamp("2025-09-01"), "position_group": "WR", "offense_pct": 0.70},
        {"player_id": "P1", "date": pd.Timestamp("2025-09-08"), "position_group": "WR", "offense_pct": 0.80},
    ])
    with_future_rows = pd.concat([through_week5, pd.DataFrame([
        {"player_id": "P1", "date": pd.Timestamp("2025-09-15"), "position_group": "WR", "offense_pct": 0.95},
        {"player_id": "P4", "date": pd.Timestamp("2025-09-15"), "position_group": "WR", "offense_pct": 0.05},
    ])], ignore_index=True)

    out_now = model.add_trailing_snap_share(through_week5, halflife_days=180.0)
    out_with_future = model.add_trailing_snap_share(with_future_rows, halflife_days=180.0)

    before = out_now[(out_now["player_id"] == "P1") & (out_now["date"] == pd.Timestamp("2025-09-08"))].iloc[0]
    after = out_with_future[(out_with_future["player_id"] == "P1") & (out_with_future["date"] == pd.Timestamp("2025-09-08"))].iloc[0]
    assert before["trailing_snap_share"] == pytest.approx(after["trailing_snap_share"]), \
        "P1's 2025-09-08 trailing_snap_share (n_prior=4, uses its own raw computed value) changed when future rows were added -- leakage in the per-row computation"
