"""Team-level game-outcome ratings, reusing model.py's exact fitting mechanics (closed-form
weighted ridge regression, recency decay) at team level instead of player level.

Margin model: a single net power rating per team. Margin (home_score - away_score) is
zero-sum, so offense and defense are not separately identifiable from margin data alone --
only their difference is -- hence one rating per team, not an ability/defense split.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

MARGIN_REG = 3.0
HOME_FIELD_REG = 0.05  # small, dedicated regularization: keeps the solve well-posed even
                        # if a fit window has zero neutral-site games, which would
                        # otherwise make the home_field column identical to the intercept
                        # column (both all-1s) and make that 2x2 block singular
MIN_MARGIN_GAMES = 100
NEW_TEAM_GAMES = 4  # fewer games than this in the fit window -> flagged as low-sample


@dataclass
class MarginRatings:
    teams: list[str]
    power: dict[str, float]
    intercept: float
    home_field: float
    sigma: float
    as_of: date
    n_games: int
    game_counts: dict[str, int]
    prior_teams: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{
            "team": t, "power": self.power[t], "games": self.game_counts.get(t, 0),
            "low_sample": self.game_counts.get(t, 0) < NEW_TEAM_GAMES,
        } for t in self.teams]
        return pd.DataFrame(rows).sort_values("power", ascending=False).reset_index(drop=True)


def fit_margin(games: pd.DataFrame, as_of: date | None = None, halflife_days: float = 365.0,
              reg: float = MARGIN_REG, min_games: int = MIN_MARGIN_GAMES) -> MarginRatings:
    """Fit team power ratings using completed games strictly before `as_of` (default: all rows)."""
    df = games.dropna(subset=["home_score", "away_score"]).copy()
    if as_of is not None:
        df = df[df["gameday"] < pd.Timestamp(as_of)]
    if len(df) < min_games:
        raise ValueError(f"need at least {min_games} completed games to fit margin, have {len(df)}")
    as_of = as_of or df["gameday"].max().date()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    tidx = {t: i for i, t in enumerate(teams)}
    n_t, n = len(teams), len(df)

    y = (df["home_score"] - df["away_score"]).to_numpy(dtype=float)
    days_ago = (pd.Timestamp(as_of) - df["gameday"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    # design matrix columns: [intercept, home_field, power(n_t): +1 home team, -1 away team]
    X = np.zeros((n, 2 + n_t))
    X[:, 0] = 1.0
    X[:, 1] = (~df["neutral"]).to_numpy(dtype=float)
    hi = df["home_team"].map(tidx).to_numpy()
    ai = df["away_team"].map(tidx).to_numpy()
    X[np.arange(n), 2 + hi] += 1.0
    X[np.arange(n), 2 + ai] -= 1.0

    penalty = np.zeros(X.shape[1])
    penalty[1] = HOME_FIELD_REG
    penalty[2:] = reg
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (y * w)
    b = np.linalg.solve(XtWX + np.diag(penalty), XtWy)

    intercept, home_field = float(b[0]), float(b[1])
    power = {t: float(b[2 + i]) for t, i in tidx.items()}

    resid = y - X @ b
    sigma = float(np.sqrt(np.average(resid ** 2, weights=w)))

    home_counts = df["home_team"].value_counts()
    away_counts = df["away_team"].value_counts()
    game_counts = home_counts.add(away_counts, fill_value=0).astype(int).to_dict()

    return MarginRatings(teams=teams, power=power, intercept=intercept, home_field=home_field,
                         sigma=sigma, as_of=as_of, n_games=n, game_counts=game_counts)


def predicted_margin(r: MarginRatings, home_team: str, away_team: str) -> tuple[float, float]:
    """(mu, sigma) of (home_score - away_score). Unknown teams get power 0.0 (league
    average) and are flagged in `r.prior_teams`, same treatment model.py gives an
    unfitted player.
    """
    home_power = r.power.get(home_team)
    if home_power is None:
        home_power = 0.0
        if home_team not in r.prior_teams:
            r.prior_teams.append(home_team)
    away_power = r.power.get(away_team)
    if away_power is None:
        away_power = 0.0
        if away_team not in r.prior_teams:
            r.prior_teams.append(away_team)
    mu = r.intercept + r.home_field + home_power - away_power
    return mu, r.sigma
