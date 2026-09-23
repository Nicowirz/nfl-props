"""Download and cache NFL player game logs, schedules, and rosters from nflverse.

URLs verified live against github.com/nflverse/nflverse-data on 2026-09-09. Also
downloads player-ID crosswalks, snap counts, play-by-play, and Next Gen Stats
receiving; those additional sources were verified live on 2026-09-22.
"""
from __future__ import annotations

import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from . import model

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.csv"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"
SNAP_COUNTS_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz"
PBP_KEEP = ["game_id", "season", "week", "posteam", "defteam", "pass", "receiver_player_id",
            "receiving_yards", "air_yards", "epa", "xyac_epa", "pass_oe", "xpass", "complete_pass"]
NGS_RECEIVING_URL = "https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{season}_receiving.csv.gz"
NGS_RECEIVING_KEEP = ["player_gsis_id", "season", "week", "avg_cushion", "avg_separation",
                      "avg_intended_air_yards", "percent_share_of_intended_air_yards",
                      "catch_percentage", "avg_yac", "avg_yac_above_expectation"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATS_KEEP = [
    "player_id", "player_display_name", "position", "position_group", "team",
    "opponent_team", "season", "week", "season_type", "game_id",
    "completions", "attempts", "passing_yards", "passing_tds",
    "carries", "rushing_yards", "rushing_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
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
    """All scheduled/completed NFL games, past and future, one row per game.

    spread_line: positive = home team favored, negative = away team favored -- nflverse's
    own convention, the OPPOSITE of the usual "-7 means favored" bettor intuition. Handle
    this explicitly everywhere it's read; never assume the typical sign.

    neutral: True for a game played at a neutral site (Super Bowl, international games),
    per nflverse's own data dictionary ("location" is "Home" or "Neutral") -- home-field
    advantage doesn't apply even though a "home" team is still designated for such games.
    """
    raw = _fetch(GAMES_URL, DATA_DIR / "games.csv", max_age_hours=6, refresh=refresh)
    out = raw[["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
               "home_score", "away_score", "spread_line", "total_line",
               "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds",
               "under_odds", "over_odds"]].copy()
    out["gameday"] = pd.to_datetime(out["gameday"], errors="coerce")
    out["neutral"] = raw["location"] != "Home"
    return out


def load_player_stats(seasons: int = 3, refresh: bool = False, today: date | None = None) -> pd.DataFrame:
    """Weekly player game logs for the last `seasons` seasons (including the current one)."""
    cur = current_season_start(today)
    games = load_games(refresh=refresh)[["game_id", "gameday", "home_team"]]
    frames = []
    for season in range(cur - seasons + 1, cur + 1):
        max_age = 6 if season == cur else 24 * 365 * 10  # finished seasons never change
        try:
            raw = _fetch(STATS_URL.format(season=season), DATA_DIR / f"stats_player_week_{season}.csv",
                         max_age, refresh)
            frames.append(raw[[c for c in STATS_KEEP if c in raw.columns]].copy())
        except urllib.error.HTTPError as e:
            # Current season may not have stats yet (preseason or week 1 before games are played/published).
            # Silently skip it and continue with historical seasons. A 404 on a historical season
            # would re-raise on the next call since it's cached, so it still signals a real problem.
            if season == cur and e.code == 404:
                continue
            raise

    if not frames:
        # No seasons had data available; return empty DataFrame with correct schema.
        return pd.DataFrame(columns=["player_id", "player_name", "position", "position_group", "team",
                                     "opponent_team", "season", "week", "season_type", "game_id",
                                     "attempts", "passing_yards", "passing_tds",
                                     "carries", "rushing_yards", "rushing_tds",
                                     "targets", "receptions", "receiving_yards", "receiving_tds",
                                     "date", "home"])

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"player_display_name": "player_name"})
    df = df.merge(games, on="game_id", how="left")
    df = df.dropna(subset=["gameday"]).copy()
    df["date"] = df["gameday"]
    df["home"] = df["team"] == df["home_team"]
    for col in ("attempts", "carries", "targets", "passing_yards", "rushing_yards", "receiving_yards",
                "passing_tds", "rushing_tds", "receiving_tds"):
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


def load_players(refresh: bool = False) -> pd.DataFrame:
    """gsis_id <-> pfr_id crosswalk (nflverse's `players` release). Needed to join
    snap_counts, which identifies players by pfr_id, onto the rest of this pipeline's
    gsis_id-keyed `player_id`.
    """
    raw = _fetch(PLAYERS_URL, DATA_DIR / "players.csv", max_age_hours=24 * 30, refresh=refresh)
    out = raw[["gsis_id", "pfr_id"]].dropna().drop_duplicates()
    return out.rename(columns={"gsis_id": "player_id"}).reset_index(drop=True)


def load_snap_counts(seasons: int = 3, refresh: bool = False, today: date | None = None) -> pd.DataFrame:
    """Offense snap share per player-game (PFR-keyed -- join_snap_counts() below maps it
    onto this pipeline's gsis_id player_id). Same per-season download/cache/skip-missing-
    current-season pattern as load_player_stats().
    """
    cur = current_season_start(today)
    frames = []
    for season in range(cur - seasons + 1, cur + 1):
        max_age = 6 if season == cur else 24 * 365 * 10
        try:
            raw = _fetch(SNAP_COUNTS_URL.format(season=season),
                         DATA_DIR / f"snap_counts_{season}.csv", max_age, refresh)
            frames.append(raw[["game_id", "pfr_player_id", "position", "team", "offense_pct"]].copy())
        except urllib.error.HTTPError as e:
            if season == cur and e.code == 404:
                continue
            raise
    if not frames:
        return pd.DataFrame(columns=["game_id", "pfr_player_id", "position", "team", "offense_pct"])
    return pd.concat(frames, ignore_index=True)


def join_snap_counts(snaps: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Map snap_counts' pfr_player_id onto this pipeline's gsis_id-based player_id via the
    players crosswalk. A snap-count row with no crosswalk match is dropped, not a hard
    failure -- the same "skipped, not fatal" treatment the Kalshi feed already gives an
    unmatched player name.

    Output includes a `team` column; the stats frame this will eventually be merged onto
    (in a future plan) also has its own `team` column, so merging on
    ["game_id", "player_id"] will produce `team_x`/`team_y` unless the caller renames or
    drops one first.
    """
    merged = snaps.merge(players, left_on="pfr_player_id", right_on="pfr_id", how="inner",
                         validate="many_to_one")
    return merged[["game_id", "player_id", "team", "offense_pct"]].reset_index(drop=True)


def load_pbp(seasons: int = 3, refresh: bool = False, today: date | None = None) -> pd.DataFrame:
    """Play-by-play rows, column-filtered to PBP_KEEP immediately on read (the raw file
    has 372 columns; only pass-play receiving/pass-rate fields are needed here). Same
    per-season download/cache/skip-missing-current-season pattern as load_player_stats().
    """
    cur = current_season_start(today)
    frames = []
    for season in range(cur - seasons + 1, cur + 1):
        max_age = 6 if season == cur else 24 * 365 * 10
        cache = DATA_DIR / f"play_by_play_{season}.csv.gz"
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < max_age * 3600
            if refresh or not fresh:
                req = urllib.request.Request(PBP_URL.format(season=season),
                                             headers={"User-Agent": "nfl-props/0.1"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    cache.write_bytes(resp.read())
            raw = pd.read_csv(cache, compression="gzip", encoding="utf-8-sig",
                              low_memory=False, usecols=lambda c: c in PBP_KEEP)
            frames.append(raw)
        except urllib.error.HTTPError as e:
            if season == cur and e.code == 404:
                continue
            raise
    if not frames:
        return pd.DataFrame(columns=PBP_KEEP)
    return pd.concat(frames, ignore_index=True)


def aggregate_pbp_receiving(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-player-game receiving aggregates from real targets only (pass plays with a
    real receiver_player_id). A descriptive fact about a completed game, like the
    existing targets/receiving_yards columns -- only safe to use through a trailing
    transform over a player's own PRIOR games, never as a same-game predictor.
    """
    targets = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()]
    g = targets.groupby(["game_id", "receiver_player_id"], as_index=False)
    out = g.agg(targets_pbp=("receiver_player_id", "size"),
                air_yards_pbp=("air_yards", "sum"),
                epa_per_target=("epa", "mean"),
                xyac_epa_per_target=("xyac_epa", "mean"))
    return out.rename(columns={"receiver_player_id": "player_id"})


def aggregate_pbp_team_pass_rate(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-team-game mean pass-rate-over-expected, over all pass plays (not just
    completed/targeted ones) -- the team-level opportunity signal for how pass-heavy a
    team's real play-calling was that game, independent of any single player's targets.
    """
    passes = pbp[pbp["pass"] == 1]
    out = passes.groupby(["game_id", "posteam"], as_index=False)["pass_oe"].mean()
    return out.rename(columns={"posteam": "team", "pass_oe": "pass_oe_game"})


def load_ngs_receiving(seasons: int = 3, refresh: bool = False, today: date | None = None) -> pd.DataFrame:
    """Next Gen Stats receiving: separation, cushion, air-yards share, catch%, YAC over
    expectation. A 200 response does NOT mean usable weekly data -- verified live on
    2026-09-22, the cached 2024 file (`data/ngs_receiving_2024.csv.gz`) contains only 8
    rows, 4 unique players, weeks [0, 1], with week 0 and week 1 carrying byte-identical
    values for the same 4 players. That is a stub/aggregate release, not real weekly NGS
    data. Callers MUST check actual row count and week granularity before depending on
    this for a real weekly covariate; do not assume a season "being live" means it's
    usable. If a weekly air-yards-share covariate is needed, `aggregate_pbp_receiving`'s
    `air_yards_pbp` column already supports deriving one directly from play-by-play,
    without needing NGS at all.

    Keys on (player_id, season, week), NOT game_id like this file's other three new
    loaders (`join_snap_counts`, `aggregate_pbp_receiving`, `aggregate_pbp_team_pass_rate`
    all key on game_id) -- a caller merging this onto the same frame needs a different
    join key than those three.
    """
    cur = current_season_start(today)
    frames = []
    for season in range(cur - seasons + 1, cur + 1):
        max_age = 6 if season == cur else 24 * 365 * 10
        try:
            raw = _fetch(NGS_RECEIVING_URL.format(season=season),
                        DATA_DIR / f"ngs_receiving_{season}.csv.gz", max_age, refresh)
            frames.append(raw[[c for c in NGS_RECEIVING_KEEP if c in raw.columns]].copy())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    if not frames:
        return pd.DataFrame(columns=["player_id"] + NGS_RECEIVING_KEEP[1:])
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"player_gsis_id": "player_id"})


def load_targets(pbp: pd.DataFrame, stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per real pass attempt at a real, identified receiver -- the target-level
    training data CatchRate|Targets needs (spec section 1.2), distinct from
    aggregate_pbp_receiving()'s per-player-GAME aggregates: a logistic catch-rate model
    needs each target as its own Bernoulli trial, not pre-summed.

    `position_group` comes from `stats` (the player's role label, not an outcome -- no
    leakage risk), keyed by `player_id`; a target thrown to a player with no
    position_group match is dropped, not guessed. `home`/`date` come from `games`'
    schedule, matching the join pattern `game_model.fit_pass_volume` already uses.
    """
    targets = pbp[(pbp["pass"] == 1) & pbp["receiver_player_id"].notna()].copy()
    targets = targets.rename(columns={"receiver_player_id": "player_id"})
    lookup = stats[["player_id", "position_group"]].drop_duplicates("player_id")
    targets = targets.merge(lookup, on="player_id", how="inner")
    sched = games[["game_id", "gameday", "home_team"]].dropna(subset=["gameday"])
    targets = targets.merge(sched, on="game_id", how="inner")
    targets["home"] = targets["posteam"] == targets["home_team"]
    targets["complete"] = targets["complete_pass"].astype(float)
    out = targets.rename(columns={"posteam": "team", "defteam": "opponent_team", "gameday": "date"})
    return out[["player_id", "game_id", "date", "team", "opponent_team", "position_group",
               "home", "complete", "air_yards", "receiving_yards"]].reset_index(drop=True)


def stats_with_trailing_snap_share(stats: pd.DataFrame, seasons: int = 3, refresh: bool = False,
                                   today: date | None = None, halflife_days: float = 180.0) -> pd.DataFrame:
    """Augment `stats` with a leakage-safe `trailing_snap_share` column: fetch snap_counts
    and the player-ID crosswalk, join them (pfr_id -> gsis_id player_id), merge the result
    onto `stats` by (game_id, player_id), then compute the trailing (recency-weighted,
    strictly-prior-games) average via model.add_trailing_snap_share().

    Only `offense_pct` is kept from the join (not `team`) -- join_snap_counts()'s own
    `team` column would collide (`team_x`/`team_y`) with `stats`' own `team` column on
    this merge otherwise. Rows with no snap-count match (crosswalk gap, or a season/week
    snap_counts doesn't cover) get `offense_pct` filled at 0.0 before computing trailing
    share -- treated as "no recorded snaps that game" rather than dropped, so this never
    shrinks the row count of `stats`.
    """
    snaps = load_snap_counts(seasons=seasons, refresh=refresh, today=today)
    players = load_players(refresh=refresh)
    joined = join_snap_counts(snaps, players)
    merged = stats.merge(joined[["game_id", "player_id", "offense_pct"]],
                         on=["game_id", "player_id"], how="left", validate="m:1")
    merged["offense_pct"] = merged["offense_pct"].fillna(0.0)
    result = model.add_trailing_snap_share(merged, halflife_days=halflife_days)
    return result.drop(columns=["offense_pct"])
