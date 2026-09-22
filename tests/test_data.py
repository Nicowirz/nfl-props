import pandas as pd

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
