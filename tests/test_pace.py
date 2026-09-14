import numpy as np
import pandas as pd
import pytest

from nfl_props import pace


def test_pace_adjust_identity_when_total_matches_league_average():
    mu = 3.8
    assert pace.pace_adjust(mu, predicted_total=44.0, league_avg_total=44.0, stat="rec_yds",
                            sensitivity={"rec_yds": 0.2}) == pytest.approx(mu)


def test_pace_adjust_shifts_up_for_a_higher_than_average_total():
    mu = 3.8
    adjusted = pace.pace_adjust(mu, predicted_total=55.0, league_avg_total=44.0, stat="rec_yds",
                                sensitivity={"rec_yds": 0.2})
    assert adjusted > mu
    assert adjusted == pytest.approx(mu + 0.2 * np.log(55.0 / 44.0))


def test_pace_adjust_shifts_down_for_a_lower_than_average_total():
    mu = 3.8
    adjusted = pace.pace_adjust(mu, predicted_total=33.0, league_avg_total=44.0, stat="rush_yds",
                                sensitivity={"rush_yds": 0.15})
    assert adjusted < mu


def test_pace_adjust_zero_sensitivity_is_always_a_no_op():
    mu = 3.8
    for total in (20.0, 44.0, 70.0):
        assert pace.pace_adjust(mu, predicted_total=total, league_avg_total=44.0, stat="pass_yds",
                                sensitivity={"pass_yds": 0.0}) == pytest.approx(mu)


def test_pace_adjust_uses_module_default_sensitivity_when_not_given():
    # module-default SENSITIVITY starts at 0.0 for every stat until Task 4 calibrates
    # real values, so this is a no-op for now -- this test just proves the default path
    # (no explicit sensitivity argument) wires through to pace.SENSITIVITY correctly.
    mu = 3.8
    assert pace.pace_adjust(mu, predicted_total=99.0, league_avg_total=44.0,
                            stat="rec_yds") == pytest.approx(mu)


def _direct_calibration_inputs(true_sensitivity, date=pd.Timestamp("2024-10-01")):
    """Hand-constructed backtest.walk_forward()/game_backtest.walk_forward_total()-shaped
    DataFrames with a KNOWN true sensitivity baked in via zero-noise residuals, used to
    test calibrate_sensitivity()'s own join + weighted-regression logic directly --
    bypassing the full model-fit pipeline, which entangles player-ability/opponent-defense
    fitting with team identity at small team counts, confounding an end-to-end recovery
    test regardless of calibrate_sensitivity()'s own correctness (verified independently).
    All rows share one date so every recency weight is 1.0, keeping the expected answer
    exactly computable.
    """
    league_avg_total = 44.0
    game_totals = {("T0", "T1"): 50.0, ("T2", "T3"): 38.0}

    player_rows = []
    for (home, away), total in game_totals.items():
        total_dev = np.log(total / league_avg_total)
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            model_mu = 3.8
            residual = true_sensitivity * total_dev
            player_rows.append({
                "date": date, "player_id": f"{team}_WR", "position_group": "WR",
                "team": team, "opponent_team": opp, "home": is_home,
                "actual_yards": 0.0, "y": model_mu + residual,
                "model_mu": model_mu, "model_sigma": 0.3, "base_mu": 0.0, "base_sigma": 0.3,
            })
    player_preds = pd.DataFrame(player_rows)

    game_rows = [{
        "gameday": date, "home_team": home, "away_team": away,
        "actual_total": total, "model_mu": total, "model_sigma": 5.0,
        "base_mu": total, "base_sigma": 5.0, "league_avg_total": league_avg_total,
    } for (home, away), total in game_totals.items()]
    game_preds = pd.DataFrame(game_rows)
    return player_preds, game_preds, date


def test_calibrate_sensitivity_recovers_known_true_value(monkeypatch):
    from nfl_props import backtest, game_backtest
    player_preds, game_preds, date = _direct_calibration_inputs(true_sensitivity=0.2)
    monkeypatch.setattr(backtest, "walk_forward", lambda *a, **k: player_preds)
    monkeypatch.setattr(game_backtest, "walk_forward_total", lambda *a, **k: game_preds)
    recovered = pace.calibrate_sensitivity(pd.DataFrame(), pd.DataFrame(), "rec_yds", date.date())
    assert recovered == pytest.approx(0.2)


def test_calibrate_sensitivity_recovers_near_zero_for_no_effect(monkeypatch):
    from nfl_props import backtest, game_backtest
    player_preds, game_preds, date = _direct_calibration_inputs(true_sensitivity=0.0)
    monkeypatch.setattr(backtest, "walk_forward", lambda *a, **k: player_preds)
    monkeypatch.setattr(game_backtest, "walk_forward_total", lambda *a, **k: game_preds)
    recovered = pace.calibrate_sensitivity(pd.DataFrame(), pd.DataFrame(), "rec_yds", date.date())
    assert recovered == pytest.approx(0.0, abs=1e-9)
