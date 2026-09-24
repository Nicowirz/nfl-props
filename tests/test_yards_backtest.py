import numpy as np
import pandas as pd

from nfl_props import model, yards_backtest


def _synthetic_multi_week(n_players=10, n_teams=4, weeks=20, catches_per_player_per_week=2, seed=0):
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.3, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.2, n_teams)))
    # intercept=4.6 (not a lower value like 2.3): exp(2.3) ~= 9.97 sits almost exactly at
    # OFFSET=10.0's floor, so ~50% of simulated rows would collapse to yards=0.0 under
    # this fixture's own max(0.0, exp(y) - OFFSET) clip -- a floor-censoring effect that
    # destroys the linear signal (confirmed empirically and analytically during Task 1's
    # review, which hit and fixed the identical bug in this plan's Task 1 fixture).
    intercept, home_field, sigma = 4.6, 0.05, 0.35
    player_team = {p: teams[i % n_teams] for i, p in enumerate(players)}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for week in range(1, weeks + 1):
        for p in players:
            team = player_team[p]
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            for _ in range(catches_per_player_per_week):
                mu = intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
                y = rng.normal(mu, sigma)
                yards = max(0.0, np.exp(y) - model.OFFSET)
                rows.append({
                    "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                    "opponent_team": opp, "season": 2024, "week": week, "date": d, "home": home,
                    "receiving_yards": yards,
                })
        d += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_walk_forward_returns_one_row_per_relevant_catch_after_start():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = yards_backtest.walk_forward(df, start, halflife_days=100_000, min_catches=50)
    expected_n = int((df["date"] >= pd.Timestamp(start)).sum())
    assert len(preds) == expected_n


def test_summarize_model_is_competitive_with_naive_baseline():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = yards_backtest.walk_forward(df, start, halflife_days=100_000, min_catches=50)
    s = yards_backtest.summarize(preds)
    assert s["n"] == len(preds)
    assert s["nll_model"] < s["nll_baseline"] * 1.2


def test_calibration_bins_cover_all_predictions():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = yards_backtest.walk_forward(df, start, halflife_days=100_000, min_catches=50)
    table = yards_backtest.calibration(preds)
    assert table["n"].sum() == len(preds)
