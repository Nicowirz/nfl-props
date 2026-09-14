import pandas as pd
import pytest

from nfl_props import tracking


def _stats():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "player_name": "Breece Hall", "rushing_yards": 102.0,
         "receiving_yards": 10.0, "passing_yards": 0.0},
    ])


def _games():
    return pd.DataFrame([
        {"season": 2026, "week": 1, "home_team": "MIN", "away_team": "GB",
         "home_score": 39.0, "away_score": 22.0},
        {"season": 2026, "week": 1, "home_team": "KC", "away_team": "DEN",
         "home_score": float("nan"), "away_score": float("nan")},
    ])


def _row(**kwargs):
    base = {"season": "2026", "week": "1", "line": "0"}
    base.update({k: str(v) for k, v in kwargs.items()})
    return pd.Series(base)


def test_grade_row_prop_hit():
    row = _row(market="rush_yds", subject="Breece Hall", selection="under", line=110.5)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert actual == "102.0"
    assert result == "HIT"


def test_grade_row_prop_miss():
    row = _row(market="rush_yds", subject="Breece Hall", selection="under", line=89.5)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert result == "MISS"


def test_grade_row_prop_push():
    row = _row(market="rush_yds", subject="Breece Hall", selection="over", line=102.0)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert result == "PUSH"


def test_grade_row_prop_pending_when_player_not_found():
    row = _row(market="rush_yds", subject="Nobody Real", selection="over", line=10.0)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert result == "PENDING"


def test_grade_row_moneyline_home_win():
    row = _row(market="moneyline", subject="GB@MIN", selection="home", line=0.0)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert actual == "17.0"  # margin = 39 - 22
    assert result == "HIT"


def test_grade_row_moneyline_away_selection_loses():
    row = _row(market="moneyline", subject="GB@MIN", selection="away", line=0.0)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert result == "MISS"


def test_grade_row_spread_home_covers():
    # nflverse convention: positive line = home favored; home covers iff margin > line.
    row = _row(market="spread", subject="GB@MIN", selection="home", line=10.5)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert result == "HIT"  # margin 17 > 10.5


def test_grade_row_total_under():
    row = _row(market="total", subject="GB@MIN", selection="under", line=65.5)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert actual == "61.0"  # total = 39 + 22
    assert result == "HIT"  # 61 < 65.5


def test_grade_row_game_pending_when_not_finished():
    row = _row(market="moneyline", subject="DEN@KC", selection="home", line=0.0)
    actual, result = tracking._grade_row(row, _stats(), _games())
    assert result == "PENDING"


def test_summarize_record_and_roi():
    df = pd.DataFrame([
        {"result": "HIT", "odds": "2.0"},
        {"result": "MISS", "odds": "3.0"},
        {"result": "PUSH", "odds": "2.0"},
        {"result": "PENDING", "odds": ""},
    ])
    s = tracking.summarize(df)
    assert s["wins"] == 1
    assert s["losses"] == 1
    assert s["pushes"] == 1
    assert s["pending"] == 1
    assert s["win_rate"] == pytest.approx(0.5)
    assert s["roi"] == pytest.approx(0.0)  # +1 unit won, -1 unit lost, over 2 decided
