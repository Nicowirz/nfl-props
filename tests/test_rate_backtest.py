import numpy as np
import pandas as pd

from nfl_props import rate_backtest


def _synthetic_multi_week(n_players=10, n_teams=4, weeks=16, seed=0):
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.25, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.15, n_teams)))
    intercept = 1.5
    player_team = {p: teams[i % n_teams] for i, p in enumerate(players)}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for week in range(1, weeks + 1):
        for p in players:
            team = player_team[p]
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            lam = np.exp(intercept + ability[p] + defense[opp])
            receptions = float(rng.poisson(lam))
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "season": 2024, "week": week, "date": d, "home": home,
                "receptions": receptions, "targets": 6.0,
            })
        d += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_walk_forward_returns_one_row_per_relevant_game_after_start():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = rate_backtest.walk_forward(df, "receptions", start, halflife_days=100_000, min_games=20)
    expected_n = int((df["date"] >= pd.Timestamp(start)).sum())
    assert len(preds) == expected_n


def test_summarize_model_is_competitive_with_naive_baseline():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = rate_backtest.walk_forward(df, "receptions", start, halflife_days=100_000, min_games=20)
    s = rate_backtest.summarize(preds)
    assert s["n"] == len(preds)
    assert s["nll_model"] < s["nll_baseline"] * 1.2


def test_calibration_bins_cover_all_predictions():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = rate_backtest.walk_forward(df, "receptions", start, halflife_days=100_000, min_games=20)
    table = rate_backtest.calibration(preds)
    assert table["n"].sum() == len(preds)
