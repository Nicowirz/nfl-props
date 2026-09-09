import numpy as np
import pandas as pd

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


def test_qualifying_filter_excludes_low_usage_games():
    df, *_ = _synthetic()
    df.loc[df.index[0], "targets"] = 1.0  # below QUALIFY_MIN["rec_yds"] = 2
    r = model.fit(df, "rec_yds", min_games=50)
    assert r.n_games == (df["targets"] >= model.QUALIFY_MIN["rec_yds"]).sum()


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
