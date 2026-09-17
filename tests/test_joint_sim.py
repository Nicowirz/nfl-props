import numpy as np
import pandas as pd
import pytest

from nfl_props import joint_sim
from nfl_props.model import OFFSET


def test_simulate_player_yards_shape():
    players = [(3.8, 0.3, "pass_yds"), (2.5, 0.4, "rush_yds")]
    samples = joint_sim.simulate_player_yards(44.0, 5.0, 44.0, players, n=500, seed=0)
    assert samples.shape == (500, 2)
    assert np.all(samples >= 0.0)  # real yards, clamped non-negative


def test_simulate_player_yards_zero_sensitivity_matches_pure_marginal():
    # With sensitivity=0.0 for every stat, the shared shock is always a no-op regardless
    # of the drawn total -- each column should be an ordinary Normal(mu, sigma) draw in
    # log space, i.e. its own median should match the unshifted marginal closely at a
    # large sample size.
    mu, sigma = 3.8, 0.3
    players = [(mu, sigma, "pass_yds")]
    samples = joint_sim.simulate_player_yards(44.0, 5.0, 44.0, players,
                                              sensitivity={"pass_yds": 0.0}, n=200_000, seed=1)
    expected_median = np.exp(mu) - OFFSET
    assert abs(np.median(samples[:, 0]) - expected_median) < 1.0


def test_simulate_player_yards_nonzero_sensitivity_shifts_distribution():
    mu, sigma = 3.8, 0.3
    players = [(mu, sigma, "pass_yds")]
    baseline = joint_sim.simulate_player_yards(60.0, 5.0, 44.0, players,
                                               sensitivity={"pass_yds": 0.0}, n=50_000, seed=2)
    shifted = joint_sim.simulate_player_yards(60.0, 5.0, 44.0, players,
                                              sensitivity={"pass_yds": 0.3}, n=50_000, seed=2)
    # total (60) > league_avg_total (44) and sensitivity is positive -> shifted draws
    # should be systematically higher than the zero-sensitivity baseline.
    assert np.mean(shifted[:, 0]) > np.mean(baseline[:, 0])


def test_simulate_player_yards_uses_module_default_sensitivity_when_omitted():
    players = [(3.8, 0.3, "pass_yds")]
    a = joint_sim.simulate_player_yards(44.0, 5.0, 44.0, players, n=100, seed=5)
    b = joint_sim.simulate_player_yards(44.0, 5.0, 44.0, players,
                                        sensitivity=joint_sim.JOINT_SENSITIVITY, n=100, seed=5)
    assert np.array_equal(a, b)


def test_simulate_player_yards_deterministic_with_seed():
    players = [(3.8, 0.3, "pass_yds"), (2.5, 0.4, "rush_yds")]
    a = joint_sim.simulate_player_yards(44.0, 5.0, 44.0, players, n=100, seed=42)
    b = joint_sim.simulate_player_yards(44.0, 5.0, 44.0, players, n=100, seed=42)
    assert np.array_equal(a, b)


def test_joint_prob_computes_empirical_frequency():
    # 4 draws, 2 players: column 0 over 10 in rows 0,1,2; column 1 under 5 in rows 0,2.
    samples = np.array([
        [15.0, 3.0],
        [12.0, 8.0],
        [20.0, 1.0],
        [5.0, 2.0],
    ])
    p = joint_sim.joint_prob(samples, [(0, "over", 10.0), (1, "under", 5.0)])
    assert p == pytest.approx(2 / 4)  # rows 0 and 2 satisfy both conditions


def test_joint_prob_single_condition_matches_marginal_frequency():
    samples = np.array([[15.0], [5.0], [20.0], [1.0]])
    p = joint_sim.joint_prob(samples, [(0, "over", 10.0)])
    assert p == pytest.approx(2 / 4)


def test_calibrate_joint_sensitivity_delegates_to_pace(monkeypatch):
    from nfl_props import pace

    captured = {}

    def fake_calibrate_sensitivity(stats, games, stat, start, **kwargs):
        captured["args"] = (stats, games, stat, start, kwargs)
        return 0.123

    monkeypatch.setattr(pace, "calibrate_sensitivity", fake_calibrate_sensitivity)
    result = joint_sim.calibrate_joint_sensitivity("STATS", "GAMES", "rush_yds", "2024-10-01",
                                                    halflife_days=90.0)
    assert result == 0.123
    assert captured["args"] == ("STATS", "GAMES", "rush_yds", "2024-10-01", {"halflife_days": 90.0})


def _stat_row(game_id, team, opp, date, player_id, position_group, home,
             attempts=0.0, carries=0.0):
    return {
        "game_id": game_id, "team": team, "opponent_team": opp, "date": date,
        "player_id": player_id, "position_group": position_group, "home": home,
        "attempts": attempts, "carries": carries,
    }


def test_qb_leading_rusher_pairs_finds_the_real_pair():
    rows = [
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "QB_A", "QB", True, attempts=30),
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "RB1_A", "RB", True, carries=15),
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "RB2_A", "RB", True, carries=4),
        _stat_row("g1", "B", "A", pd.Timestamp("2024-09-01"), "QB_B", "QB", False, attempts=25),
        _stat_row("g1", "B", "A", pd.Timestamp("2024-09-01"), "RB1_B", "RB", False, carries=10),
    ]
    df = pd.DataFrame(rows)
    pairs = joint_sim.qb_leading_rusher_pairs(df)
    assert len(pairs) == 2
    team_a = pairs[pairs["team"] == "A"].iloc[0]
    assert team_a["qb_player_id"] == "QB_A"
    assert team_a["rusher_player_id"] == "RB1_A"  # 15 carries beats RB2_A's 4
    assert team_a["rusher_carries"] == 15


def test_qb_leading_rusher_pairs_excludes_sub_qualifying_qb():
    rows = [
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "QB_A", "QB", True, attempts=3),  # < QUALIFY_MIN
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "RB1_A", "RB", True, carries=15),
    ]
    df = pd.DataFrame(rows)
    pairs = joint_sim.qb_leading_rusher_pairs(df)
    assert len(pairs) == 0


def test_qb_leading_rusher_pairs_excludes_when_qb_is_also_leading_rusher():
    rows = [
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "QB_A", "QB", True, attempts=30, carries=12),
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "RB1_A", "RB", True, carries=4),
    ]
    df = pd.DataFrame(rows)
    pairs = joint_sim.qb_leading_rusher_pairs(df)
    assert len(pairs) == 0  # the QB himself has more carries than any teammate -- no valid pair


def test_qb_leading_rusher_pairs_excludes_team_with_no_qualifying_rusher():
    rows = [
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "QB_A", "QB", True, attempts=30),
        _stat_row("g1", "A", "B", pd.Timestamp("2024-09-01"), "RB1_A", "RB", True, carries=0),
    ]
    df = pd.DataFrame(rows)
    pairs = joint_sim.qb_leading_rusher_pairs(df)
    assert len(pairs) == 0


def test_check_correlation_direction_returns_real_and_simulated_corr(monkeypatch):
    from nfl_props import backtest, game_backtest

    date = pd.Timestamp("2024-10-01")
    pass_preds = pd.DataFrame([
        {"player_id": "QB_A", "date": date, "y": 4.2, "model_mu": 4.0, "model_sigma": 0.3},
    ])
    rush_preds = pd.DataFrame([
        {"player_id": "RB1_A", "date": date, "y": 2.0, "model_mu": 2.5, "model_sigma": 0.4},
    ])
    game_preds = pd.DataFrame([
        {"gameday": date, "home_team": "A", "away_team": "B",
         "model_mu": 44.0, "model_sigma": 5.0, "league_avg_total": 44.0},
    ])
    monkeypatch.setattr(backtest, "walk_forward",
                        lambda stats, stat, start, **k: pass_preds if stat == "pass_yds" else rush_preds)
    monkeypatch.setattr(game_backtest, "walk_forward_total", lambda *a, **k: game_preds)

    stats = pd.DataFrame([
        {"game_id": "g1", "team": "A", "opponent_team": "B", "date": date,
         "player_id": "QB_A", "position_group": "QB", "home": True, "attempts": 30.0, "carries": 0.0},
        {"game_id": "g1", "team": "A", "opponent_team": "B", "date": date,
         "player_id": "RB1_A", "position_group": "RB", "home": True, "attempts": 0.0, "carries": 15.0},
    ])
    result = joint_sim.check_correlation_direction(stats, "GAMES", date.date(), n=200)
    assert result["n_pairs"] == 1
    assert -1.0 <= result["simulated_corr"] <= 1.0 or pd.isna(result["simulated_corr"])
