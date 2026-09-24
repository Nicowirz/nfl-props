"""Weighted, ridge-penalized log-normal regression for Yards | Reception (spec section
1.2, Stage 1 efficiency): refits model.py's existing log(yards + OFFSET) ridge structure
conditional on RECEPTIONS -- one row per completed catch (data.load_targets(),
complete == 1), not per player-game -- since a player's yards on a given catch is a
different quantity than his total receiving yards across a game (model.py's rec_yds
fit), and Stage 1's eventual Monte Carlo composition needs this per-catch distribution as
a separate factor (Targets x CatchRate x Yards|Reception), not a replacement for
rec_yds.

This is the BASE slice of Yards|Reception -- opponent defense stays a single scalar per
team (mirroring model.py's existing convention), not yet the spec's position-split
yards-allowed-per-catch variant -- deliberately staged the same way CatchRate's own
position-split term was: validate the base per-catch refit first, ablation-test
position-split separately once this is validated, per this project's "validate before
extending" discipline. CatchRate's own position-split ablation already came back a real
loss (see BASELINE.md), so there is no presumption position-split will help here either.

No sigma_low_sample/WIDE_SIGMA_STATS/share_coef machinery is ported from model.py's
generic-over-stat Ratings -- none of it applies to a brand-new per-catch model with no
existing calibration history to diagnose against; add it later only if a real backtest
surfaces the specific problem it would fix, matching model.py's own WIDE_SIGMA_STATS
history (added only after a measured real calibration gap, not preemptively).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import model

# Any position data.load_targets() attaches a stats match for -- mirrors
# catch_model.RELEVANT_POSITIONS's identical convention.
RELEVANT_POSITIONS = {"QB", "RB", "TE", "WR"}

# model.NEW_PLAYER_GAMES (4) is calibrated for a GAME count; a qualifying receiver
# typically catches several passes per game, so 20 catches is the catch-count analog --
# same reasoning catch_model.NEW_PLAYER_TARGETS already used for targets.
NEW_PLAYER_CATCHES = 20


@dataclass
class YardsRatings:
    players: list[str]
    player_name: dict[str, str]
    position_group: dict[str, str]
    ability: dict[str, float]
    teams: list[str]
    defense: dict[str, float]
    position_intercept: dict[str, float]
    intercept_fallback: float
    home_field: float
    sigma: dict[str, float]
    sigma_global: float
    as_of: date
    n_catches: int
    catch_counts: dict[str, int]
    prior_players: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{
            "player": self.player_name.get(p, p),
            "position": self.position_group.get(p, ""),
            "ability": self.ability[p],
            "catches": self.catch_counts.get(p, 0),
            "low_sample": self.catch_counts.get(p, 0) < NEW_PLAYER_CATCHES,
        } for p in self.players]
        return pd.DataFrame(rows).sort_values("ability", ascending=False).reset_index(drop=True)


def fit_yards_per_catch(targets: pd.DataFrame, as_of: date | None = None, halflife_days: float = 180.0,
                        reg: float = 5.0, min_catches: int = 200) -> YardsRatings:
    """Fit on every COMPLETED-target row from a relevant position, strictly before
    `as_of` (default: all rows). `targets` must already be filtered to complete == 1 by
    the caller -- a catch's yards is undefined (NaN) on an incomplete target, and this
    function does not filter that itself, matching rate_model.fit_poisson's
    "caller decides the population" convention rather than silently dropping rows here.
    """
    df = targets
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    df = df[df["position_group"].isin(RELEVANT_POSITIONS)].copy()
    n = len(df)
    if n < min_catches:
        raise ValueError(f"need at least {min_catches} catches to fit Yards|Reception, have {n}")
    as_of = as_of or df["date"].max().date()

    players = sorted(df["player_id"].unique())
    teams = sorted(set(df["team"]) | set(df["opponent_team"]))
    groups = sorted(df["position_group"].unique())
    pidx = {p: i for i, p in enumerate(players)}
    tidx = {t: i for i, t in enumerate(teams)}
    gidx = {g: i for i, g in enumerate(groups)}
    n_p, n_t, n_g = len(players), len(teams), len(groups)

    y = model._safe_log_yards(df["receiving_yards"].to_numpy(dtype=float))
    days_ago = (pd.Timestamp(as_of) - df["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    X = np.zeros((n, n_g + 1 + n_p + n_t))
    gi = df["position_group"].map(gidx).to_numpy()
    X[np.arange(n), gi] = 1.0
    X[:, n_g] = df["home"].to_numpy(dtype=float)
    pi = df["player_id"].map(pidx).to_numpy()
    ti = df["opponent_team"].map(tidx).to_numpy()
    X[np.arange(n), n_g + 1 + pi] = 1.0
    X[np.arange(n), n_g + 1 + n_p + ti] = 1.0

    penalty = np.zeros(X.shape[1])
    penalty[:n_g] = model.GROUP_INTERCEPT_REG
    penalty[n_g + 1:n_g + 1 + n_p + n_t] = reg

    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (y * w)
    b = np.linalg.solve(XtWX + np.diag(penalty), XtWy)

    position_intercept = {g: float(b[i]) for g, i in gidx.items()}
    home_field = float(b[n_g])
    ability = {p: float(b[n_g + 1 + i]) for p, i in pidx.items()}
    defense = {t: float(b[n_g + 1 + n_p + i]) for t, i in tidx.items()}
    group_counts = df["position_group"].value_counts().to_dict()
    intercept_fallback = float(np.average([position_intercept[g] for g in groups],
                                          weights=[group_counts[g] for g in groups]))

    resid = y - X @ b
    sigma_global = float(np.sqrt(np.average(resid ** 2, weights=w)))
    group = df["position_group"].to_numpy()
    sigma: dict[str, float] = {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() >= model.MIN_GROUP_RESIDUALS:
            sigma[g] = float(np.sqrt(np.average(resid[mask] ** 2, weights=w[mask])))

    player_name = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    position_group = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()
    catch_counts = df["player_id"].value_counts().to_dict()

    return YardsRatings(
        players=players, player_name=player_name, position_group=position_group,
        ability=ability, teams=teams, defense=defense,
        position_intercept=position_intercept, intercept_fallback=intercept_fallback,
        home_field=home_field, sigma=sigma, sigma_global=sigma_global,
        as_of=as_of, n_catches=n, catch_counts=catch_counts,
    )


def predicted_yards_distribution(r: YardsRatings, player_id: str, position_group: str,
                                 opponent_team: str, home: bool) -> tuple[float, float]:
    """(mu, sigma) of log(receiving_yards + OFFSET) for one completed catch. Same
    unknown-player/unknown-group fallback treatment as model.predicted_distribution.
    """
    ability = r.ability.get(player_id)
    if ability is None:
        ability = 0.0
        if player_id not in r.prior_players:
            r.prior_players.append(player_id)
    defense = r.defense.get(opponent_team, 0.0)
    intercept = r.position_intercept.get(position_group, r.intercept_fallback)
    mu = intercept + ability + defense + (r.home_field if home else 0.0)
    sigma = r.sigma.get(position_group, r.sigma_global)
    return mu, sigma
