"""Download and cache NFL player game logs, schedules, and rosters from nflverse.

URLs verified live against github.com/nflverse/nflverse-data on 2026-09-09.
"""
from __future__ import annotations

import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.csv"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATS_KEEP = [
    "player_id", "player_display_name", "position", "position_group", "team",
    "opponent_team", "season", "week", "season_type", "game_id",
    "completions", "attempts", "passing_yards",
    "carries", "rushing_yards", "receptions", "targets", "receiving_yards",
]


def current_season_start(today: date | None = None) -> int:
    """NFL seasons are named for the year they start; Jan/Feb games belong to the prior year's season."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def _fetch(url: str, cache: Path, max_age_hours: float, refresh: bool) -> pd.DataFrame:
    cache.parent.mkdir(parents=True, exist_ok=True)
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < max_age_hours * 3600
    if refresh or not fresh:
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-props/0.1"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            cache.write_bytes(resp.read())
    return pd.read_csv(cache, encoding="utf-8-sig", low_memory=False)


def load_games(refresh: bool = False) -> pd.DataFrame:
    """All scheduled/completed NFL games, past and future, one row per game."""
    raw = _fetch(GAMES_URL, DATA_DIR / "games.csv", max_age_hours=6, refresh=refresh)
    out = raw[["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
               "home_score", "away_score"]].copy()
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    return out


def load_player_stats(seasons: int = 3, refresh: bool = False, today: date | None = None) -> pd.DataFrame:
    """Weekly player game logs for the last `seasons` seasons (including the current one)."""
    cur = current_season_start(today)
    games = load_games(refresh=refresh)[["game_id", "gameday", "home_team"]]
    frames = []
    for season in range(cur - seasons + 1, cur + 1):
        max_age = 6 if season == cur else 24 * 365 * 10  # finished seasons never change
        raw = _fetch(STATS_URL.format(season=season), DATA_DIR / f"stats_player_week_{season}.csv",
                     max_age, refresh)
        frames.append(raw[[c for c in STATS_KEEP if c in raw.columns]].copy())
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"player_display_name": "player_name"})
    df = df.merge(games, on="game_id", how="left")
    df = df.dropna(subset=["gameday"]).copy()
    df["date"] = df["gameday"]
    df["home"] = df["team"] == df["home_team"]
    for col in ("attempts", "carries", "targets", "passing_yards", "rushing_yards", "receiving_yards"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df.sort_values("date").reset_index(drop=True)


def load_rosters(season: int, week: int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Weekly roster: which team each player is on, position, and roster status."""
    raw = _fetch(ROSTER_URL.format(season=season), DATA_DIR / f"roster_weekly_{season}.csv", 6, refresh)
    out = raw[["season", "week", "team", "position", "status", "full_name", "gsis_id"]].copy()
    out = out.rename(columns={"gsis_id": "player_id"})
    if week is not None:
        out = out[out["week"] == week]
    return out.dropna(subset=["player_id"]).reset_index(drop=True)
