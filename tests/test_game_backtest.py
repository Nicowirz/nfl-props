import numpy as np
import pandas as pd
from scipy.stats import norm

from nfl_props import game_backtest


def _synthetic_season_margins(n_teams=16, n_weeks=14, seed=1):
    """Multi-week synthetic games with a real per-team power spread, so a baseline that
    ignores team identity (home-field-only) should score worse than the fitted model.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    power = dict(zip(teams, rng.normal(0, 8.0, n_teams)))  # sizeable spread
    home_field, sigma = 2.0, 13.0
    rows = []
    for week in range(1, n_weeks + 1):
        d = pd.Timestamp("2024-09-05") + pd.Timedelta(weeks=week - 1)
        order = list(rng.permutation(teams))
        for i in range(0, n_teams - 1, 2):
            home, away = order[i], order[i + 1]
            mu = home_field + power[home] - power[away]
            margin = rng.normal(mu, sigma)
            rows.append({
                "game_id": f"g{len(rows)}", "season": 2024, "game_type": "REG", "week": week,
                "gameday": d, "home_team": home, "away_team": away,
                "home_score": 20.0 + margin / 2, "away_score": 20.0 - margin / 2,
                "neutral": False,
            })
    return pd.DataFrame(rows)


def test_walk_forward_margin_produces_one_row_per_test_game():
    df = _synthetic_season_margins()
    start = df["gameday"].iloc[len(df) // 2].date()
    preds = game_backtest.walk_forward_margin(df, start, min_games=20)
    expected = (df["gameday"] >= pd.Timestamp(start)).sum()
    assert len(preds) == expected
    assert set(preds.columns) >= {"gameday", "home_team", "away_team", "actual_margin",
                                  "model_mu", "model_sigma", "base_mu", "base_sigma"}


def test_margin_model_beats_home_field_only_baseline():
    df = _synthetic_season_margins()
    start = df["gameday"].iloc[len(df) // 2].date()
    preds = game_backtest.walk_forward_margin(df, start, min_games=20)
    s = game_backtest.summarize(preds, "actual_margin")
    assert s["nll_model"] < s["nll_baseline"]


def test_margin_calibration_pit_is_roughly_uniform():
    df = _synthetic_season_margins(n_weeks=20)
    start = df["gameday"].iloc[len(df) // 3].date()
    preds = game_backtest.walk_forward_margin(df, start, reg=0.1, min_games=20)
    z = (preds["actual_margin"] - preds["model_mu"]) / preds["model_sigma"]
    pit = norm.cdf(z)
    assert 0.35 < pit.mean() < 0.65
    cal = game_backtest.calibration(preds, "actual_margin")
    assert cal["n"].sum() == len(preds)


def test_walk_forward_total_includes_league_avg_total():
    df = _synthetic_season_margins()
    start = df["gameday"].iloc[len(df) // 2].date()
    preds = game_backtest.walk_forward_total(df, start, min_games=20)
    assert "league_avg_total" in preds.columns
    assert (preds["league_avg_total"] > 0).all()
