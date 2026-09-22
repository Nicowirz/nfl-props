import numpy as np
import pandas as pd

from nfl_props import model, rate_model


def _synthetic_poisson(n_players=24, n_teams=8, games_per_player=16, seed=0, ability_override=None):
    """Simulate player-games from known ability/defense/home/intercept parameters,
    with true Poisson-distributed counts (dispersion == 1)."""
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.3, n_players)))
    if ability_override:
        ability.update(ability_override)
    defense = dict(zip(teams, rng.normal(0, 0.2, n_teams)))
    intercept, home_field = 1.6, 0.05  # exp(1.6) ~= 5 receptions/game baseline
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for p in players:
        team = player_team[p]
        opp_pool = [t for t in teams if t != team]
        for _ in range(games_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            eta = intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
            receptions = float(rng.poisson(np.exp(eta)))
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home,
                "receptions": receptions, "targets": 6.0,
            })
            d += pd.Timedelta(days=1)
    return pd.DataFrame(rows), players, teams, ability, defense, intercept, home_field


def test_fit_poisson_recovers_ability_and_defense():
    df, players, teams, ability, defense, intercept, home_field = _synthetic_poisson()
    r = rate_model.fit_poisson(df, "receptions", reg=0.05, halflife_days=100_000, min_games=50)
    est_ability = np.array([r.ability[p] for p in players])
    true_ability = np.array([ability[p] for p in players])
    est_defense = np.array([r.defense[t] for t in teams])
    true_defense = np.array([defense[t] for t in teams])
    assert np.corrcoef(true_ability, est_ability)[0, 1] > 0.8
    assert np.corrcoef(true_defense, est_defense)[0, 1] > 0.6
    assert abs(r.home_field - home_field) < 0.15


def test_fit_poisson_respects_as_of():
    df, *_ = _synthetic_poisson()
    cutoff = df["date"].iloc[100].date()
    r = rate_model.fit_poisson(df, "receptions", as_of=cutoff, min_games=50)
    assert r.n_games == (df["date"] < pd.Timestamp(cutoff)).sum()
    assert r.as_of == cutoff


def test_low_sample_player_is_shrunk_toward_zero():
    df, players, teams, ability, defense, *_ = _synthetic_poisson(
        games_per_player=20, ability_override={"P0": 0.6, "P1": 0.6})
    p_many, p_few = "P0", "P1"  # both have the same TRUE ability; only sample size differs
    few_idx = df[df["player_id"] == p_few].index[4:]
    df = df.drop(few_idx)
    r = rate_model.fit_poisson(df, "receptions", reg=1.0, min_games=50)
    assert abs(r.ability[p_few]) < abs(r.ability[p_many])


def test_predicted_rate_flags_unknown_player():
    df, players, teams, *_ = _synthetic_poisson()
    r = rate_model.fit_poisson(df, "receptions", min_games=50)
    lam, disp = rate_model.predicted_rate(r, "unknown-id", "WR", teams[0], True)
    assert np.isfinite(lam) and lam > 0
    assert disp >= 1.0
    assert "unknown-id" in r.prior_players


def test_dispersion_floor_is_never_below_one():
    df, *_ = _synthetic_poisson()
    r = rate_model.fit_poisson(df, "receptions", min_games=50)
    assert r.dispersion_global >= 1.0
    assert all(v >= 1.0 for v in r.dispersion.values())


def _synthetic_with_share(n_players=16, n_teams=8, games_per_player=14, seed=11, true_share_coef=1.2):
    """Like _synthetic_poisson(), but with real teammates sharing a game_id (so target
    share is well-defined) and a KNOWN true share-sensitivity baked into the generating
    model, mirroring test_model.py's _synthetic_with_share()."""
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.2, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.15, n_teams)))
    intercept, home_field = 1.4, 0.05
    player_team = {p: teams[i % n_teams] for i, p in enumerate(players)}
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
                    "receptions": 0.0, "targets": raw_targets[p],
                })
        d += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    aug = model.add_trailing_share(df, "receptions", halflife_days=100_000)
    rng2 = np.random.default_rng(seed + 1)
    new_y = []
    for _, row in aug.iterrows():
        eta = (intercept + ability[row["player_id"]] + defense[row["opponent_team"]]
              + (home_field if row["home"] else 0.0) + true_share_coef * row["trailing_share"])
        new_y.append(rng2.poisson(np.exp(eta)))
    aug["receptions"] = np.array(new_y, dtype=float)
    return aug.drop(columns=["trailing_share"]), true_share_coef


def test_fit_poisson_recovers_share_coef():
    df, true_share_coef = _synthetic_with_share()
    aug = model.add_trailing_share(df, "receptions", halflife_days=100_000)
    r = rate_model.fit_poisson(aug, "receptions", reg=0.05, halflife_days=100_000, min_games=50)
    assert r.share_coef is not None
    assert abs(r.share_coef - true_share_coef) < 0.5


def _synthetic_targets(n_players=24, n_teams=8, games_per_player=16, seed=0, ability_override=None):
    """Like _synthetic_poisson() but simulates the TARGETS column with true Poisson counts
    (dispersion == 1), leaving `receptions` as an unused placeholder column -- a separate
    fixture so the existing, already-approved receptions tests' fixture stays untouched.
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.3, n_players)))
    if ability_override:
        ability.update(ability_override)
    defense = dict(zip(teams, rng.normal(0, 0.2, n_teams)))
    intercept, home_field = 1.8, 0.05  # exp(1.8) ~= 6 targets/game baseline
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for p in players:
        team = player_team[p]
        opp_pool = [t for t in teams if t != team]
        for _ in range(games_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            eta = intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
            targets = float(rng.poisson(np.exp(eta)))
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home,
                "targets": targets, "receptions": 0.0,
            })
            d += pd.Timedelta(days=1)
    return pd.DataFrame(rows), players, teams, ability, defense, intercept, home_field


def test_fit_poisson_recovers_ability_and_defense_for_targets():
    df, players, teams, ability, defense, intercept, home_field = _synthetic_targets()
    r = rate_model.fit_poisson(df, "targets", reg=0.05, halflife_days=100_000, min_games=50)
    est_ability = np.array([r.ability[p] for p in players])
    true_ability = np.array([ability[p] for p in players])
    est_defense = np.array([r.defense[t] for t in teams])
    true_defense = np.array([defense[t] for t in teams])
    assert np.corrcoef(true_ability, est_ability)[0, 1] > 0.8
    assert np.corrcoef(true_defense, est_defense)[0, 1] > 0.6
    assert abs(r.home_field - home_field) < 0.15


def _synthetic_targets_with_share(n_players=16, n_teams=8, games_per_player=14, seed=11, true_share_coef=1.2):
    """Like _synthetic_with_share() but for the targets stat. Unlike receptions (whose
    share-driving column "targets" differs from its own modeled column "receptions"),
    targets' share-driving column IS its own modeled column (QUALIFY_COLUMN["targets"] ==
    "targets") -- so trailing_share must be computed from the RAW seed targets BEFORE they
    are overwritten by the simulated Poisson counts, and the returned frame keeps that
    already-computed trailing_share rather than dropping it, since recomputing it from the
    post-overwrite "targets" column would be circular (it would no longer reflect the share
    values that actually generated the simulated counts).
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.2, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.15, n_teams)))
    intercept, home_field = 1.6, 0.05
    player_team = {p: teams[i % n_teams] for i, p in enumerate(players)}
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
                    "receptions": 0.0, "targets": raw_targets[p],
                })
        d += pd.Timedelta(days=7)
    df = pd.DataFrame(rows)
    aug = model.add_trailing_share(df, "targets", halflife_days=100_000)
    rng2 = np.random.default_rng(seed + 1)
    new_y = []
    for _, row in aug.iterrows():
        eta = (intercept + ability[row["player_id"]] + defense[row["opponent_team"]]
              + (home_field if row["home"] else 0.0) + true_share_coef * row["trailing_share"])
        new_y.append(rng2.poisson(np.exp(eta)))
    aug["targets"] = np.array(new_y, dtype=float)
    return aug, true_share_coef


def test_fit_poisson_recovers_share_coef_for_targets():
    aug, true_share_coef = _synthetic_targets_with_share()
    r = rate_model.fit_poisson(aug, "targets", reg=0.05, halflife_days=100_000, min_games=50)
    assert r.share_coef is not None
    assert abs(r.share_coef - true_share_coef) < 0.5
