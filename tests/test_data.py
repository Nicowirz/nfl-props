import pandas as pd
import pytest

from nfl_props import data


def test_join_snap_counts_maps_pfr_id_to_player_id():
    snaps = pd.DataFrame([
        {"game_id": "2025_01_ARI_NO", "pfr_player_id": "McBrTr00", "position": "TE",
         "team": "ARI", "offense_pct": 0.85},
        {"game_id": "2025_01_ARI_NO", "pfr_player_id": "UnknownX00", "position": "WR",
         "team": "ARI", "offense_pct": 0.40},
    ])
    players = pd.DataFrame([
        {"player_id": "00-0037744", "pfr_id": "McBrTr00"},
    ])
    out = data.join_snap_counts(snaps, players)
    assert list(out.columns) == ["game_id", "player_id", "team", "offense_pct"]
    assert len(out) == 1
    assert out.iloc[0]["player_id"] == "00-0037744"
    assert out.iloc[0]["offense_pct"] == 0.85


def test_join_snap_counts_drops_unmatched_rows_without_raising():
    snaps = pd.DataFrame([
        {"game_id": "2025_01_ARI_NO", "pfr_player_id": "NoMatch00", "position": "WR",
         "team": "ARI", "offense_pct": 0.5},
    ])
    players = pd.DataFrame([{"player_id": "00-0000001", "pfr_id": "SomeoneElse00"}])
    out = data.join_snap_counts(snaps, players)
    assert len(out) == 0
    assert list(out.columns) == ["game_id", "player_id", "team", "offense_pct"]


def _synthetic_pbp():
    return pd.DataFrame([
        # ARI@NO, game 1: two targets to P1, one incomplete to P2, one run play (not a pass)
        {"game_id": "2025_01_ARI_NO", "posteam": "ARI", "defteam": "NO", "pass": 1,
         "receiver_player_id": "00-1111", "receiving_yards": 12.0, "air_yards": 8.0,
         "epa": 0.50, "xyac_epa": 0.30, "pass_oe": 5.0, "xpass": 0.55},
        {"game_id": "2025_01_ARI_NO", "posteam": "ARI", "defteam": "NO", "pass": 1,
         "receiver_player_id": "00-1111", "receiving_yards": 4.0, "air_yards": 2.0,
         "epa": -0.10, "xyac_epa": 0.10, "pass_oe": -2.0, "xpass": 0.50},
        {"game_id": "2025_01_ARI_NO", "posteam": "ARI", "defteam": "NO", "pass": 1,
         "receiver_player_id": "00-2222", "receiving_yards": 0.0, "air_yards": 15.0,
         "epa": -0.40, "xyac_epa": 0.05, "pass_oe": 3.0, "xpass": 0.60},
        {"game_id": "2025_01_ARI_NO", "posteam": "ARI", "defteam": "NO", "pass": 0,
         "receiver_player_id": None, "receiving_yards": 0.0, "air_yards": 0.0,
         "epa": 0.20, "xyac_epa": 0.0, "pass_oe": 0.0, "xpass": 0.0},
    ])


def test_aggregate_pbp_receiving_counts_targets_and_averages_efficiency():
    out = data.aggregate_pbp_receiving(_synthetic_pbp())
    p1 = out[out["player_id"] == "00-1111"].iloc[0]
    assert p1["game_id"] == "2025_01_ARI_NO"
    assert p1["targets_pbp"] == 2
    assert p1["air_yards_pbp"] == pytest.approx(10.0)  # 8.0 + 2.0
    assert p1["epa_per_target"] == pytest.approx(0.20)  # mean(0.50, -0.10)
    assert p1["xyac_epa_per_target"] == pytest.approx(0.20)  # mean(0.30, 0.10)
    assert len(out) == 2  # only the two real receivers, not the run play


def test_aggregate_pbp_team_pass_rate_averages_over_pass_plays_only():
    out = data.aggregate_pbp_team_pass_rate(_synthetic_pbp())
    ari = out[(out["game_id"] == "2025_01_ARI_NO") & (out["team"] == "ARI")].iloc[0]
    assert ari["pass_oe_game"] == pytest.approx(2.0)  # mean(5.0, -2.0, 3.0), run play excluded
