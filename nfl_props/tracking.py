"""Log recommended picks (from `best-bet`) and grade them against real results once
games complete. Local-only, same as every other file under data/ -- git-ignored, never
committed, never shared.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "picks_log.csv"
COLUMNS = ["logged_at", "season", "week", "market", "subject", "selection", "line",
          "odds", "p_model", "edge", "actual", "result"]

STAT_MARKETS = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}


def log_picks(rows: list[dict], path: Path = LOG_PATH) -> None:
    """Append picks to the log. Each row needs: season, week, market, subject,
    selection, line, odds, p_model, edge. `actual`/`result` start blank (PENDING).

    market: one of "pass_yds"/"rush_yds"/"rec_yds" (subject = player name) or
    "moneyline"/"spread"/"total" (subject = "AWAY@HOME"); selection: "over"/"under" for
    a stat or total, "home"/"away" for moneyline/spread. line: 0.0 for moneyline (no
    real line), matching how game_leg_model_prob already treats it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["logged_at"] = datetime.now().isoformat(timespec="seconds")
    df["actual"] = ""
    df["result"] = "PENDING"
    for col in ("odds", "p_model", "edge"):
        df[col] = df[col].astype(float).round(4)
    df["line"] = df["line"].astype(float).round(1)
    df = df[COLUMNS]
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _actual_prop(stats: pd.DataFrame, season: int, week: int, player: str, market: str) -> float | None:
    col = STAT_MARKETS[market]
    row = stats[(stats["season"] == season) & (stats["week"] == week)
               & (stats["player_name"].str.lower() == player.lower())]
    return float(row.iloc[0][col]) if not row.empty else None


def _actual_game(games: pd.DataFrame, season: int, week: int, subject: str) -> tuple[float, float] | None:
    """Returns (margin, total) = (home_score - away_score, home_score + away_score), or
    None if the game hasn't finished (or isn't found)."""
    away, home = subject.split("@")
    row = games[(games["season"] == season) & (games["week"] == week)
               & (games["home_team"] == home) & (games["away_team"] == away)]
    if row.empty or pd.isna(row.iloc[0]["home_score"]):
        return None
    g = row.iloc[0]
    return float(g["home_score"] - g["away_score"]), float(g["home_score"] + g["away_score"])


def _grade_row(row: pd.Series, stats: pd.DataFrame, games: pd.DataFrame) -> tuple[str, str]:
    """Returns (actual_str, result) -- result is 'PENDING' if the game hasn't happened yet."""
    season, week, line = int(row["season"]), int(row["week"]), float(row["line"])
    if row["market"] in STAT_MARKETS:
        actual = _actual_prop(stats, season, week, row["subject"], row["market"])
        if actual is None:
            return "", "PENDING"
        if actual == line:
            return str(actual), "PUSH"
        hit = actual > line if row["selection"] == "over" else actual < line
        return str(actual), ("HIT" if hit else "MISS")

    g = _actual_game(games, season, week, row["subject"])
    if g is None:
        return "", "PENDING"
    margin, total = g
    if row["market"] == "total":
        if total == line:
            return str(total), "PUSH"
        hit = total > line if row["selection"] == "over" else total < line
        return str(total), ("HIT" if hit else "MISS")
    # moneyline (line == 0.0) or spread: home covers/wins iff margin > line, matching
    # game_leg_model_prob's exact comparison.
    if margin == line:
        return str(margin), "PUSH"
    hit = margin > line if row["selection"] == "home" else margin < line
    return str(margin), ("HIT" if hit else "MISS")


def grade_log(stats: pd.DataFrame, games: pd.DataFrame, path: Path = LOG_PATH) -> pd.DataFrame:
    """Fill in actual/result for every PENDING row whose game has now completed, save
    the log back, and return the full (now-updated) log."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for i, row in df.iterrows():
        if row["result"] != "PENDING":
            continue
        actual, result = _grade_row(row, stats, games)
        df.at[i, "actual"] = actual
        df.at[i, "result"] = result
    df.to_csv(path, index=False)
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Record (wins/losses/pushes/pending) and flat-1-unit-stake ROI over decided picks."""
    decided = df[df["result"].isin(["HIT", "MISS"])]
    wins = int((decided["result"] == "HIT").sum())
    losses = int((decided["result"] == "MISS").sum())
    pushes = int((df["result"] == "PUSH").sum())
    pending = int((df["result"] == "PENDING").sum())
    odds = decided["odds"].astype(float)
    is_win = (decided["result"] == "HIT").to_numpy()
    pnl = float((odds[is_win] - 1).sum() - (~is_win).sum())
    n = wins + losses
    return {
        "wins": wins, "losses": losses, "pushes": pushes, "pending": pending,
        "win_rate": wins / n if n else float("nan"),
        "roi": pnl / n if n else float("nan"),
    }
