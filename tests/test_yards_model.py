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
