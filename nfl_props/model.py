"""Weighted ridge regression on log(yards + OFFSET): player ability + opponent defense
+ home field, with exponential recency decay. Residuals are normal (per position group),
giving each player-game a log-normal yardage distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

STAT_COLUMN = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}
QUALIFY_COLUMN = {"pass_yds": "attempts", "rush_yds": "carries", "rec_yds": "targets"}
QUALIFY_MIN = {"pass_yds": 10, "rush_yds": 5, "rec_yds": 2}

OFFSET = 10.0             # log(yards + OFFSET) stays finite even for a slightly negative rushing game
NEW_PLAYER_GAMES = 4      # fewer qualifying games than this -> flagged as low-sample in output
MIN_GROUP_RESIDUALS = 30  # fewer residuals than this in a position group -> fall back to sigma_global


def _safe_log_yards(yards) -> np.ndarray:
    """log(yards + OFFSET), with yards floored so the log argument never drops to zero
    or below. A handful of real games have extreme negative yardage (a fumbled lateral,
    a punter's fake-punt carry) that would otherwise NaN out and poison the whole ridge
    fit; those are treated as equivalent to this floor rather than crashing.
    """
    raw = np.asarray(yards, dtype=float)
    return np.log(np.maximum(raw, 1.0 - OFFSET) + OFFSET)


@dataclass
class Ratings:
    stat: str
    players: list[str]
    player_name: dict[str, str]
    position_group: dict[str, str]
    ability: dict[str, float]
    teams: list[str]
    defense: dict[str, float]
    intercept: float
    home_field: float
    sigma: dict[str, float]
    sigma_global: float
    as_of: date
    n_games: int
    game_counts: dict[str, int]
    prior_players: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{
            "player": self.player_name.get(p, p),
            "position": self.position_group.get(p, ""),
            "ability": self.ability[p],
            "games": self.game_counts.get(p, 0),
            "low_sample": self.game_counts.get(p, 0) < NEW_PLAYER_GAMES,
        } for p in self.players]
        return pd.DataFrame(rows).sort_values("ability", ascending=False).reset_index(drop=True)


def fit(stats: pd.DataFrame, stat: str, as_of: date | None = None, halflife_days: float = 180.0,
        reg: float = 5.0, min_games: int = 200) -> Ratings:
    """Fit ratings using qualifying games strictly before `as_of` (default: all rows)."""
    df = stats
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    qcol, qmin = QUALIFY_COLUMN[stat], QUALIFY_MIN[stat]
    df = df[df[qcol] >= qmin].copy()
    if len(df) < min_games:
        raise ValueError(f"need at least {min_games} qualifying games to fit {stat}, have {len(df)}")
    as_of = as_of or df["date"].max().date()

    players = sorted(df["player_id"].unique())
    teams = sorted(set(df["team"]) | set(df["opponent_team"]))
    pidx = {p: i for i, p in enumerate(players)}
    tidx = {t: i for i, t in enumerate(teams)}
    n_p, n_t, n = len(players), len(teams), len(df)

    y = _safe_log_yards(df[STAT_COLUMN[stat]].to_numpy(dtype=float))
    days_ago = (pd.Timestamp(as_of) - df["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    # design matrix columns: [intercept, home, player(n_p), opponent-defense(n_t)]
    X = np.zeros((n, 2 + n_p + n_t))
    X[:, 0] = 1.0
    X[:, 1] = df["home"].to_numpy(dtype=float)
    pi = df["player_id"].map(pidx).to_numpy()
    ti = df["opponent_team"].map(tidx).to_numpy()
    X[np.arange(n), 2 + pi] = 1.0
    X[np.arange(n), 2 + n_p + ti] = 1.0

    penalty = np.zeros(X.shape[1])
    penalty[2:] = reg  # intercept and home field are not regularized
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (y * w)
    b = np.linalg.solve(XtWX + np.diag(penalty), XtWy)

    intercept, home_field = float(b[0]), float(b[1])
    ability = {p: float(b[2 + i]) for p, i in pidx.items()}
    defense = {t: float(b[2 + n_p + i]) for t, i in tidx.items()}

    resid = y - X @ b
    sigma_global = float(np.sqrt(np.average(resid ** 2, weights=w)))
    group = df["position_group"].to_numpy()
    sigma: dict[str, float] = {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() >= MIN_GROUP_RESIDUALS:
            sigma[g] = float(np.sqrt(np.average(resid[mask] ** 2, weights=w[mask])))

    player_name = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    position_group = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()
    game_counts = df["player_id"].value_counts().to_dict()

    return Ratings(
        stat=stat, players=players, player_name=player_name, position_group=position_group,
        ability=ability, teams=teams, defense=defense, intercept=intercept, home_field=home_field,
        sigma=sigma, sigma_global=sigma_global, as_of=as_of, n_games=n, game_counts=game_counts,
    )


def predicted_distribution(r: Ratings, player_id: str, position_group: str, opponent_team: str,
                           home: bool) -> tuple[float, float]:
    """(mu, sigma) of log(yards + OFFSET) for a player-game. Unknown players get the
    league-average ability (0.0) and are flagged in `r.prior_players`, same treatment
    epl-parlay gives a club with no fitted history.
    """
    ability = r.ability.get(player_id)
    if ability is None:
        ability = 0.0
        if player_id not in r.prior_players:
            r.prior_players.append(player_id)
    defense = r.defense.get(opponent_team, 0.0)
    mu = r.intercept + ability + defense + (r.home_field if home else 0.0)
    sigma = r.sigma.get(position_group, r.sigma_global)
    return mu, sigma
