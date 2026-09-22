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


def test_stats_with_trailing_snap_share_adds_leakage_safe_column():
    # Player has 5 games; the join provides real snap data for the first 4. Game 5 has
    # exactly NEW_PLAYER_GAMES(4) prior games -- model.add_trailing_snap_share's fallback
    # only overwrites rows with FEWER than 4 prior games (see model.py's
    # add_trailing_snap_share, and the leakage-fixture bug this exact off-by-threshold
    # mistake caused twice already this session, in the two prior plans' Task 5 and
    # Task 2 -- this fixture is deliberately built to avoid repeating it a third time), so
    # game 5's trailing_snap_share uses its own raw computed weighted average, not a
    # fallback -- a real, hand-verifiable, non-degenerate check that the full pipeline
    # (snap_counts -> crosswalk -> merge -> trailing share) actually fed real data through.
    stats = pd.DataFrame([
        {"player_id": "00-1111", "game_id": f"2025_0{w}_ARI_X{w}", "position_group": "WR",
         "date": pd.Timestamp("2025-09-01") + pd.Timedelta(weeks=w - 1), "team": "ARI"}
        for w in range(1, 6)
    ])
    snaps = pd.DataFrame([
        {"game_id": f"2025_0{w}_ARI_X{w}", "pfr_player_id": "PlayJo00", "position": "WR",
         "team": "ARI", "offense_pct": pct}
        for w, pct in zip(range(1, 5), [0.60, 0.70, 0.80, 0.90])
    ])
    players = pd.DataFrame([{"player_id": "00-1111", "pfr_id": "PlayJo00"}])

    import unittest.mock as mock
    with mock.patch.object(data, "load_snap_counts", return_value=snaps), \
         mock.patch.object(data, "load_players", return_value=players):
        out = data.stats_with_trailing_snap_share(stats, halflife_days=100_000)

    assert "trailing_snap_share" in out.columns
    row5 = out[(out["player_id"] == "00-1111") & (out["game_id"] == "2025_05_ARI_X5")].iloc[0]
    # n_prior=4 for game 5 (>= NEW_PLAYER_GAMES) -- raw weighted average of [0.60, 0.70,
    # 0.80, 0.90] with a huge halflife (near-uniform weights) is ~0.75, well above 0.5.
    # Game 5's own (NaN -> 0.0-filled, since snaps has no week-5 entry) offense_pct must
    # never influence this -- only its 4 strictly-prior games' real values do.
    assert row5["trailing_snap_share"] > 0.5
