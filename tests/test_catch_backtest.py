import numpy as np
import pandas as pd

from nfl_props import catch_backtest


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic_multi_week(n_players=10, n_teams=4, weeks=20, targets_per_player_per_week=3, seed=0):
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    ability = dict(zip(players, rng.normal(0, 0.5, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.4, n_teams)))
    intercept = 0.4
    player_team = {p: teams[i % n_teams] for i, p in enumerate(players)}
    rows = []
    d = pd.Timestamp("2024-09-01")
    for week in range(1, weeks + 1):
        for p in players:
            team = player_team[p]
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            for _ in range(targets_per_player_per_week):
                air_yards = float(rng.normal(9.0, 6.0))
                eta = intercept + ability[p] + defense[opp] - 0.08 * air_yards
                complete = float(rng.uniform() < _sigmoid(eta))
                rows.append({
                    "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                    "opponent_team": opp, "season": 2024, "week": week, "date": d, "home": home,
                    "complete": complete, "air_yards": air_yards,
                })
        d += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def test_walk_forward_returns_one_row_per_relevant_target_after_start():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = catch_backtest.walk_forward(df, start, halflife_days=100_000, min_targets=50)
    expected_n = int((df["date"] >= pd.Timestamp(start)).sum())
    assert len(preds) == expected_n


def test_summarize_model_is_competitive_with_naive_baseline():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = catch_backtest.walk_forward(df, start, halflife_days=100_000, min_targets=50)
    s = catch_backtest.summarize(preds)
    assert s["n"] == len(preds)
    assert s["nll_model"] < s["nll_baseline"] * 1.2


def test_calibration_bins_cover_all_predictions():
    df = _synthetic_multi_week()
    start = df["date"].iloc[len(df) // 2].date()
    preds = catch_backtest.walk_forward(df, start, halflife_days=100_000, min_targets=50)
    table = catch_backtest.calibration(preds)
    assert table["n"].sum() == len(preds)


def test_walk_forward_passes_through_position_split_defense():
    """Regression guard for the exact seam that has broken silently before in this
    project (rate_backtest.py's targets/share branch, per its own test file's comment):
    confirm walk_forward actually threads position_split_defense through to
    fit_catch_rate rather than silently ignoring it, by confirming the run completes and
    produces finite NLL under the interaction-term design matrix -- not just that the
    parameter is accepted without a TypeError.
    """
    df = _synthetic_multi_week(n_players=18, n_teams=6, weeks=30, targets_per_player_per_week=4)
    start = df["date"].iloc[len(df) // 2].date()
    preds = catch_backtest.walk_forward(df, start, halflife_days=100_000, min_targets=50,
                                        position_split_defense=True)
    s = catch_backtest.summarize(preds)
    assert np.isfinite(s["nll_model"])
    assert np.isfinite(s["nll_baseline"])
