"""Weighted, ridge-penalized log-normal regression for Yards | Reception (spec section
1.2, Stage 1 efficiency): refits model.py's existing log(yards + OFFSET) ridge structure
conditional on RECEPTIONS -- one row per completed catch (data.load_targets(),
complete == 1), not per player-game -- since a player's yards on a given catch is a
different quantity than his total receiving yards across a game (model.py's rec_yds
fit), and Stage 1's eventual Monte Carlo composition needs this per-catch distribution as
a separate factor (Targets x CatchRate x Yards|Reception), not a replacement for
rec_yds.

The opponent-defense term defaults to a single scalar per team, not position-split --
this model's first validated slice. An optional position-split variant
(yards-allowed-per-catch by opponent x position, via `position_split_defense=True`) was
added and real-data ablation-tested in
docs/superpowers/plans/2026-09-24-nfl-props-yards-position-split-defense.md -- the
result was a real loss (-0.435% relative NLL, see BASELINE.md), so it is NOT the
default and has not been made one.

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

# model.NEW_PLAYER_GAMES (4) is calibrated for a GAME count. A precise catch-count
# conversion would need a qualifying receiver's typical catches/game (targets/game x
# catch rate, both stat-dependent) -- 20 is a round threshold in the same spirit and
# rough magnitude as catch_model.NEW_PLAYER_TARGETS's own 20 (also not a precise
# derivation), not an exact 4-games-of-catches conversion.
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
    defense_position: dict[tuple[str, str], float] | None = None

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
                        reg: float = 5.0, min_catches: int = 200,
                        position_split_defense: bool = False) -> YardsRatings:
    """Fit on every COMPLETED-target row from a relevant position, strictly before
    `as_of` (default: all rows). `targets` must already be filtered to complete == 1 by
    the caller -- a catch's yards is undefined (NaN) on an incomplete target, and this
    function does not filter that itself, matching rate_model.fit_poisson's
    "caller decides the population" convention rather than silently dropping rows here.
    A NaN receiving_yards row (the caller forgot to pre-filter) raises ValueError rather
    than silently producing an all-NaN fit -- verified empirically: without this guard,
    np.linalg.solve propagates NaN through the entire closed-form solve with no
    exception, corrupting every player's ability/defense value, not just the offending
    row's own contribution.

    position_split_defense=False (default): opponent defense is a single scalar per
    team, matching this model's first validated slice (see BASELINE.md's "Yards |
    Reception (first validation)" section) -- this path is byte-identical to that
    already-shipped behavior. position_split_defense=True: opponent defense becomes one
    coefficient per observed (opponent_team, position_group) combination -- a direct
    team-x-position INTERACTION, not team_effect + position_effect added together --
    mirroring catch_model.fit_catch_rate's own position_split_defense design exactly
    (same interaction-column construction, same fallback treatment), adapted for a
    closed-form linear solve instead of logistic IRLS. This is the spec's own
    position-split yards-allowed-per-catch term, validated in
    docs/superpowers/plans/2026-09-24-nfl-props-yards-position-split-defense.md.
    """
    df = targets
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    df = df[df["position_group"].isin(RELEVANT_POSITIONS)].copy()
    n = len(df)
    if n < min_catches:
        raise ValueError(f"need at least {min_catches} catches to fit Yards|Reception, have {n}")
    if df["receiving_yards"].isna().any():
        raise ValueError("targets contains NaN receiving_yards -- filter to complete == 1 "
                         "before calling fit_yards_per_catch")
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

    if position_split_defense:
        combos = sorted(set(zip(df["opponent_team"], df["position_group"])))
        cidx = {c: i for i, c in enumerate(combos)}
        n_c = len(combos)
        X = np.zeros((n, n_g + 1 + n_p + n_c))
        gi = df["position_group"].map(gidx).to_numpy()
        X[np.arange(n), gi] = 1.0
        X[:, n_g] = df["home"].to_numpy(dtype=float)
        pi = df["player_id"].map(pidx).to_numpy()
        ci = np.array([cidx[(t, g)] for t, g in zip(df["opponent_team"], df["position_group"])])
        X[np.arange(n), n_g + 1 + pi] = 1.0
        X[np.arange(n), n_g + 1 + n_p + ci] = 1.0
        penalty = np.zeros(X.shape[1])
        penalty[:n_g] = model.GROUP_INTERCEPT_REG
        penalty[n_g + 1:n_g + 1 + n_p + n_c] = reg
    else:
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
    if position_split_defense:
        defense: dict[str, float] = {}
        defense_position = {c: float(b[n_g + 1 + n_p + i]) for c, i in cidx.items()}
    else:
        defense = {t: float(b[n_g + 1 + n_p + i]) for t, i in tidx.items()}
        defense_position = None
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
        ability=ability, teams=teams, defense=defense, defense_position=defense_position,
        position_intercept=position_intercept, intercept_fallback=intercept_fallback,
        home_field=home_field, sigma=sigma, sigma_global=sigma_global,
        as_of=as_of, n_catches=n, catch_counts=catch_counts,
    )


def predicted_yards_distribution(r: YardsRatings, player_id: str, position_group: str,
                                 opponent_team: str, home: bool) -> tuple[float, float]:
    """(mu, sigma) of log(receiving_yards + OFFSET) for one completed catch. Same
    unknown-player/unknown-group fallback treatment as model.predicted_distribution.
    When `r` was fit with position_split_defense=True, an unseen (opponent_team,
    position_group) combo falls back to a neutral 0.0 defense contribution, same
    fallback value as the non-split path's unseen-team case.
    """
    ability = r.ability.get(player_id)
    if ability is None:
        ability = 0.0
        if player_id not in r.prior_players:
            r.prior_players.append(player_id)
    if r.defense_position is not None:
        defense = r.defense_position.get((opponent_team, position_group), 0.0)
    else:
        defense = r.defense.get(opponent_team, 0.0)
    intercept = r.position_intercept.get(position_group, r.intercept_fallback)
    mu = intercept + ability + defense + (r.home_field if home else 0.0)
    sigma = r.sigma.get(position_group, r.sigma_global)
    return mu, sigma
