"""Weighted, ridge-penalized logistic (Bernoulli) regression (IRLS) for CatchRate |
Targets (spec section 1.2, Stage 1 efficiency). Mirrors rate_model.py's IRLS structure
(position-group intercepts, player ability, opponent-team defense, home field) but fits
a logit-link Bernoulli mean instead of a log-link Poisson mean, since each row of
data.load_targets() is a single 0/1 catch/no-catch trial, not a count -- there is no
closed-form warm start the way rate_model's log(y + 1) trick provides (log-odds of a
single 0/1 observation is undefined at both endpoints), so this starts IRLS at beta=0
(eta=0 -> mu=0.5 everywhere), the standard logistic-regression starting point.

Unlike rate_model.py (generic over several count stats: receptions/targets/TDs), this
module fits exactly one outcome -- Stage 1 has exactly one CatchRate market -- so it
follows game_model.py's dedicated-function convention (fit_margin/fit_score/
fit_pass_volume) rather than rate_model.py's generic-over-`stat` one.

The opponent-defense term here is a single scalar per team, not position-split -- this
is Stage 1's first validated slice of CatchRate. A position-split opponent term
(completion-rate-allowed-to-WR vs. -TE vs. -RB) is the spec's own next ablation stage,
deliberately deferred to a follow-on plan once this base model is validated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import model

# Any position data.load_targets() attaches a stats match for -- mirrors
# rate_model.RELEVANT_POSITIONS's convention for "targets"/"receptions".
RELEVANT_POSITIONS = {"QB", "RB", "TE", "WR"}

MAX_IRLS_ITER = 25
IRLS_TOL = 1e-6
# Clip predicted probability away from the 0/1 boundary -- division-by-zero guard in the
# IRLS working-response/weight update (mirrors rate_model's np.maximum(mu, 1e-6) guard,
# adapted for a bounded [0, 1] mean instead of an unbounded-above Poisson mean).
MU_EPS = 1e-6
# model.NEW_PLAYER_GAMES (4) is calibrated for a GAME count; a typical WR sees ~5
# targets/game, so 20 targets is the target-count analog.
NEW_PLAYER_TARGETS = 20


@dataclass
class CatchRatings:
    players: list[str]
    player_name: dict[str, str]
    position_group: dict[str, str]
    ability: dict[str, float]
    teams: list[str]
    defense: dict[str, float]
    defense_position: dict[tuple[str, str], float] | None
    position_intercept: dict[str, float]
    intercept_fallback: float
    air_yards_coef: float
    home_field: float
    as_of: date
    n_targets: int
    target_counts: dict[str, int]
    prior_players: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        rows = [{
            "player": self.player_name.get(p, p),
            "position": self.position_group.get(p, ""),
            "ability": self.ability[p],
            "targets": self.target_counts.get(p, 0),
            "low_sample": self.target_counts.get(p, 0) < NEW_PLAYER_TARGETS,
        } for p in self.players]
        return pd.DataFrame(rows).sort_values("ability", ascending=False).reset_index(drop=True)


def fit_catch_rate(targets: pd.DataFrame, as_of: date | None = None, halflife_days: float = 180.0,
                   reg: float = 5.0, min_targets: int = 200,
                   position_split_defense: bool = False) -> CatchRatings:
    """Fit on every target row from a relevant position, strictly before `as_of` (default:
    all rows). Unlike rate_model.fit_poisson, there is no qualifying-games threshold here
    to further restrict the fitted population -- every row IS already an individual
    Bernoulli trial, not a player-game aggregate that needs a volume floor.

    position_split_defense=False (default): opponent defense is a single scalar per team,
    matching Stage 1's first validated slice (see BASELINE.md's "CatchRate | Targets
    (first validation)" section) -- this path is byte-identical to that already-shipped
    behavior. position_split_defense=True: opponent defense becomes one coefficient per
    observed (opponent_team, position_group) combination -- a direct team-x-position
    INTERACTION, not team_effect + position_effect added together -- capturing "how this
    specific defense performs against this specific position" (spec's own audit: a
    bend-don't-break defense and a boom-or-bust defense look identical under the
    single-scalar design). This is the spec's own next Stage 1 ablation step
    (`## Ablation plan`), validated in
    docs/superpowers/plans/2026-09-23-nfl-props-catch-rate-position-split-defense.md.
    """
    df = targets
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    df = df[df["position_group"].isin(RELEVANT_POSITIONS)].copy()
    n = len(df)
    if n < min_targets:
        raise ValueError(f"need at least {min_targets} targets to fit CatchRate, have {n}")
    as_of = as_of or df["date"].max().date()

    players = sorted(df["player_id"].unique())
    teams = sorted(set(df["team"]) | set(df["opponent_team"]))
    groups = sorted(df["position_group"].unique())
    pidx = {p: i for i, p in enumerate(players)}
    tidx = {t: i for i, t in enumerate(teams)}
    gidx = {g: i for i, g in enumerate(groups)}
    n_p, n_t, n_g = len(players), len(teams), len(groups)

    y = df["complete"].to_numpy(dtype=float)
    days_ago = (pd.Timestamp(as_of) - df["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    if position_split_defense:
        combos = sorted(set(zip(df["opponent_team"], df["position_group"])))
        cidx = {c: i for i, c in enumerate(combos)}
        n_c = len(combos)
        X = np.zeros((n, n_g + 1 + n_p + n_c + 1))
        gi = df["position_group"].map(gidx).to_numpy()
        X[np.arange(n), gi] = 1.0
        X[:, n_g] = df["home"].to_numpy(dtype=float)
        pi = df["player_id"].map(pidx).to_numpy()
        ci = np.array([cidx[(t, g)] for t, g in zip(df["opponent_team"], df["position_group"])])
        X[np.arange(n), n_g + 1 + pi] = 1.0
        X[np.arange(n), n_g + 1 + n_p + ci] = 1.0
        X[:, n_g + 1 + n_p + n_c] = df["air_yards"].to_numpy(dtype=float)
        penalty = np.zeros(X.shape[1])
        penalty[:n_g] = model.GROUP_INTERCEPT_REG
        penalty[n_g + 1:n_g + 1 + n_p + n_c] = reg
    else:
        X = np.zeros((n, n_g + 1 + n_p + n_t + 1))
        gi = df["position_group"].map(gidx).to_numpy()
        X[np.arange(n), gi] = 1.0
        X[:, n_g] = df["home"].to_numpy(dtype=float)
        pi = df["player_id"].map(pidx).to_numpy()
        ti = df["opponent_team"].map(tidx).to_numpy()
        X[np.arange(n), n_g + 1 + pi] = 1.0
        X[np.arange(n), n_g + 1 + n_p + ti] = 1.0
        X[:, n_g + 1 + n_p + n_t] = df["air_yards"].to_numpy(dtype=float)
        penalty = np.zeros(X.shape[1])
        penalty[:n_g] = model.GROUP_INTERCEPT_REG
        penalty[n_g + 1:n_g + 1 + n_p + n_t] = reg

    beta = np.zeros(X.shape[1])

    for _ in range(MAX_IRLS_ITER):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.clip(1.0 / (1.0 + np.exp(-eta)), MU_EPS, 1.0 - MU_EPS)
        variance = mu * (1.0 - mu)
        z = eta + (y - mu) / variance
        irls_w = w * variance
        XtWX = X.T @ (X * irls_w[:, None])
        XtWz = X.T @ (z * irls_w)
        beta_new = np.linalg.solve(XtWX + np.diag(penalty), XtWz)
        converged = np.max(np.abs(beta_new - beta)) < IRLS_TOL
        beta = beta_new
        if converged:
            break

    position_intercept = {g: float(beta[i]) for g, i in gidx.items()}
    home_field = float(beta[n_g])
    ability = {p: float(beta[n_g + 1 + i]) for p, i in pidx.items()}
    if position_split_defense:
        defense: dict[str, float] = {}
        defense_position = {c: float(beta[n_g + 1 + n_p + i]) for c, i in cidx.items()}
        air_yards_coef = float(beta[n_g + 1 + n_p + n_c])
    else:
        defense = {t: float(beta[n_g + 1 + n_p + i]) for t, i in tidx.items()}
        defense_position = None
        air_yards_coef = float(beta[n_g + 1 + n_p + n_t])
    group_counts = df["position_group"].value_counts().to_dict()
    intercept_fallback = float(np.average([position_intercept[g] for g in groups],
                                          weights=[group_counts[g] for g in groups]))

    player_name = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    position_group = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()
    target_counts = df["player_id"].value_counts().to_dict()

    return CatchRatings(
        players=players, player_name=player_name, position_group=position_group,
        ability=ability, teams=teams, defense=defense, defense_position=defense_position,
        position_intercept=position_intercept, intercept_fallback=intercept_fallback,
        air_yards_coef=air_yards_coef, home_field=home_field,
        as_of=as_of, n_targets=n, target_counts=target_counts,
    )


def predicted_catch_rate(r: CatchRatings, player_id: str, position_group: str, opponent_team: str,
                         home: bool, air_yards: float) -> float:
    """Probability the target is caught. Same unknown-player/unknown-group fallback
    treatment as rate_model.predicted_rate/model.predicted_distribution. When `r` was fit
    with position_split_defense=True, an unseen (opponent_team, position_group) combo
    falls back to a neutral 0.0 defense contribution, same fallback value as the
    non-split path's unseen-team case.
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
    eta = (intercept + ability + defense + (r.home_field if home else 0.0)
          + r.air_yards_coef * air_yards)
    return float(1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20))))
