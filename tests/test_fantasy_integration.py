"""Closes the one structural gap the final review found: existing fantasy.py tests only
feed simulate_player hand-constructed (mu, sigma)/(lambda, dispersion) tuples, never a
real model.fit()/rate_model.fit_poisson() output. This fits both on the same synthetic
player-games and threads their real predictions through the simulator end to end.
"""
import numpy as np
import pandas as pd

from nfl_props import fantasy, model, rate_model


def _synthetic_player_games(n_players=20, n_teams=8, games_per_player=16, seed=0):
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.3, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.2, n_teams)))
    yards_intercept, rec_intercept, home_field = 4.0, 1.6, 0.05
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for p in players:
        team = player_team[p]
        opp_pool = [t for t in teams if t != team]
        for _ in range(games_per_player):
            opp = rng.choice(opp_pool)
            home = bool(rng.integers(0, 2))
            common = ability[p] + defense[opp] + (home_field if home else 0.0)
            yards = float(np.exp(rng.normal(yards_intercept + common, 0.4))) - 10.0
            receptions = float(rng.poisson(np.exp(rec_intercept + common)))
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home,
                "receiving_yards": yards, "targets": 6.0, "receptions": receptions,
            })
            d += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def test_simulate_player_with_real_model_and_rate_model_outputs():
    df = _synthetic_player_games()
    yardage_ratings = model.fit(df, "rec_yds", reg=1.0, halflife_days=100_000, min_games=50)
    rate_ratings = rate_model.fit_poisson(df, "receptions", reg=1.0, halflife_days=100_000, min_games=50)

    opp = next(t for t in df["team"].unique() if t != df.loc[df["player_id"] == "P0", "team"].iloc[0])
    mu, sigma = model.predicted_distribution(yardage_ratings, "P0", "WR", opp, True)
    lam, disp = rate_model.predicted_rate(rate_ratings, "P0", "WR", opp, True)

    proj = fantasy.simulate_player("P0", {"rec_yds": (mu, sigma)}, {"receptions": (lam, disp)}, seed=42)
    assert np.isfinite(proj.mean)
    assert proj.mean > 0
