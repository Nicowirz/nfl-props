import numpy as np
import pandas as pd
import pytest

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


def _synthetic_games(n_teams=8, n_weeks=30, seed=4):
    """A companion games DataFrame with home_score/away_score/home_team/away_team using
    the SAME team names _synthetic_weeks() uses (T0..T{n_teams-1}), for the pace_adjust
    path's tests -- game_model.fit_score/predicted_total only need this shape, it doesn't
    need to be play-by-play consistent with _synthetic_weeks()'s own player-to-team
    assignments to prove the mechanism runs and applies an adjustment.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    scoring = dict(zip(teams, rng.normal(0, 3.0, n_teams)))
    allowed = dict(zip(teams, rng.normal(0, 3.0, n_teams)))
    rows = []
    date0 = pd.Timestamp("2024-09-05")
    for week in range(1, n_weeks + 1):
        d = date0 + pd.Timedelta(weeks=week - 1)
        order = list(rng.permutation(teams))
        for i in range(0, n_teams - 1, 2):
            home, away = order[i], order[i + 1]
            home_score = max(0.0, 22.0 + scoring[home] + allowed[away] + rng.normal(0, 4))
            away_score = max(0.0, 22.0 + scoring[away] + allowed[home] + rng.normal(0, 4))
            rows.append({
                "season": 2024, "week": week, "gameday": d, "home_team": home, "away_team": away,
                "home_score": home_score, "away_score": away_score, "neutral": False,
            })
    return pd.DataFrame(rows)


def test_walk_forward_row_dict_includes_team_fields_for_the_join():
    df = _synthetic_weeks()
    start = df["date"].iloc[len(df) // 2].date()
    preds = backtest.walk_forward(df, "rec_yds", start, min_games=50)
    assert set(preds.columns) >= {"team", "opponent_team", "home"}


def test_walk_forward_pace_adjust_requires_games():
    df = _synthetic_weeks()
    start = df["date"].iloc[len(df) // 2].date()
    with pytest.raises(ValueError, match="games"):
        backtest.walk_forward(df, "rec_yds", start, min_games=50, pace_adjust=True)


def test_walk_forward_pace_adjust_shifts_model_mu():
    df = _synthetic_weeks()
    games = _synthetic_games()
    start = df["date"].iloc[len(df) // 2].date()
    unadjusted = backtest.walk_forward(df, "rec_yds", start, reg=0.05, min_games=50)
    adjusted = backtest.walk_forward(df, "rec_yds", start, reg=0.05, min_games=50,
                                     pace_adjust=True, games=games,
                                     sensitivity={"rec_yds": 0.3}, game_min_games=20)
    assert len(adjusted) == len(unadjusted)
    # a non-zero sensitivity must change at least some predictions -- with real (not
    # perfectly matched) game data, not every row's total will sit exactly at that
    # week's league average, so at least some model_mu values must differ.
    assert not np.allclose(adjusted["model_mu"].to_numpy(), unadjusted["model_mu"].to_numpy())
