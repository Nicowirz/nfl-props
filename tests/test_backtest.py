import numpy as np
import pandas as pd

from nfl_props import backtest


def _synthetic_weeks(n_players=16, n_teams=8, n_weeks=30, seed=1):
    """Multi-week synthetic season(s) with a real opponent-defense effect, so a baseline
    that ignores the opponent should score worse than the fitted model.
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.30, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.35, n_teams)))  # sizeable, so it's worth capturing
    intercept, home_field, sigma = 4.5, 0.05, 0.30
    player_team = {p: rng.choice(teams) for p in players}
    rows = []
    season, date0 = 2024, pd.Timestamp("2024-09-05")
    for week in range(1, n_weeks + 1):
        wk_season = season if week <= 18 else season + 1
        wk = week if week <= 18 else week - 18
        d = date0 + pd.Timedelta(weeks=week - 1)
        for p in players:
            team = player_team[p]
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            mu = intercept + ability[p] + defense[opp] + (home_field if home else 0.0)
            y = rng.normal(mu, sigma)
            yards = max(0.0, np.exp(y) - 10.0)
            rows.append({
                "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                "opponent_team": opp, "date": d, "home": home, "season": wk_season, "week": wk,
                "receiving_yards": yards, "targets": 6.0,
                "passing_yards": 0.0, "attempts": 0.0, "rushing_yards": 0.0, "carries": 0.0,
            })
    return pd.DataFrame(rows)


def test_walk_forward_produces_one_row_per_test_game():
    df = _synthetic_weeks()
    start = df["date"].iloc[len(df) // 2].date()
    preds = backtest.walk_forward(df, "rec_yds", start, min_games=50)
    expected = (df["date"] >= pd.Timestamp(start)).sum()
    assert len(preds) == expected
    assert set(preds.columns) >= {"date", "player_id", "actual_yards", "y",
                                  "model_mu", "model_sigma", "base_mu", "base_sigma"}


def test_model_beats_opponent_blind_baseline():
    df = _synthetic_weeks()
    start = df["date"].iloc[len(df) // 2].date()
    preds = backtest.walk_forward(df, "rec_yds", start, reg=0.05, min_games=50)
    s = backtest.summarize(preds)
    assert s["nll_model"] < s["nll_baseline"]


def test_calibration_pit_is_roughly_uniform():
    df = _synthetic_weeks(n_weeks=40)
    start = df["date"].iloc[len(df) // 3].date()
    preds = backtest.walk_forward(df, "rec_yds", start, reg=0.05, min_games=50)
    z = (preds["y"] - preds["model_mu"]) / preds["model_sigma"]
    from scipy.stats import norm
    pit = norm.cdf(z)
    assert 0.35 < pit.mean() < 0.65
    cal = backtest.calibration(preds)
    assert cal["n"].sum() == len(preds)
