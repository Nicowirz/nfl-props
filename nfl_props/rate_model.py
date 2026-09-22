"""Weighted, ridge-penalized Poisson regression (IRLS) for count stats: receptions and
touchdowns. Mirrors model.py's ability/opponent-defense/home-field design matrix and
recency weighting, but fits a log-link Poisson mean instead of a log-normal yardage
distribution, since receptions and touchdowns are non-negative integer counts with no
closed-form ridge solution the way log(yards + OFFSET) has.

Overdispersion (variance > mean, which real reception/TD counts often show) is captured
via a quasi-Poisson scale factor -- the weighted mean squared Pearson residual, floored
at 1.0 -- rather than fitting a full negative-binomial log-likelihood. predicted_rate()
returns this as `dispersion`; callers needing to SAMPLE from the resulting distribution
reparameterize it as a negative binomial with the same mean/variance (see fantasy.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import model

STAT_COLUMN = {"receptions": "receptions", "pass_td": "passing_tds",
              "rush_td": "rushing_tds", "rec_td": "receiving_tds", "targets": "targets"}
QUALIFY_COLUMN = {"receptions": "targets", "pass_td": "attempts",
                  "rush_td": "carries", "rec_td": "targets", "targets": "targets"}
QUALIFY_MIN = {"receptions": 2, "pass_td": 10, "rush_td": 5, "rec_td": 2, "targets": 2}
RELEVANT_POSITIONS = {
    "receptions": {"QB", "RB", "TE", "WR"}, "pass_td": {"QB"},
    "rush_td": {"QB", "RB", "TE", "WR"}, "rec_td": {"QB", "RB", "TE", "WR"},
    "targets": {"QB", "RB", "TE", "WR"},
}
# Only receptions and targets have a usage-share covariate in v1 (target share drives
# both). A red-zone-share covariate would be the natural analog for touchdown rates but
# is out of scope -- see the design spec's model-extension section.
SHARE_STATS = {"receptions", "targets"}

MAX_IRLS_ITER = 25
IRLS_TOL = 1e-6


@dataclass
class RateRatings:
    stat: str
    players: list[str]
    player_name: dict[str, str]
    position_group: dict[str, str]
    ability: dict[str, float]
    teams: list[str]
    defense: dict[str, float]
    position_intercept: dict[str, float]
    intercept_fallback: float
    share_coef: float | None
    share_fallback: dict[str, float]
    home_field: float
    dispersion: dict[str, float]
    dispersion_global: float
    as_of: date
    n_games: int
    game_counts: dict[str, int]
    prior_players: list[str] = field(default_factory=list)
    extra_coefs: dict[str, float] = field(default_factory=dict)
    extra_fallback: dict[str, dict[str, float]] = field(default_factory=dict)

    def table(self) -> pd.DataFrame:
        rows = [{
            "player": self.player_name.get(p, p),
            "position": self.position_group.get(p, ""),
            "ability": self.ability[p],
            "games": self.game_counts.get(p, 0),
            "low_sample": self.game_counts.get(p, 0) < model.NEW_PLAYER_GAMES,
        } for p in self.players]
        return pd.DataFrame(rows).sort_values("ability", ascending=False).reset_index(drop=True)


def fit_poisson(stats: pd.DataFrame, stat: str, as_of: date | None = None, halflife_days: float = 180.0,
               reg: float = 5.0, min_games: int = 200,
               extra_covariates: list[str] | None = None) -> RateRatings:
    """Fit ratings on every game from a stat-relevant position, strictly before `as_of`
    (default: all rows) -- same "fit on all relevant games, not just qualifying ones"
    reasoning as model.fit() (see its docstring): restricting to qualifying games would
    overstate every player's true rate.
    """
    df = stats
    if as_of is not None:
        df = df[df["date"] < pd.Timestamp(as_of)]
    df = df[df["position_group"].isin(RELEVANT_POSITIONS[stat])].copy()
    qcol, qmin = QUALIFY_COLUMN[stat], QUALIFY_MIN[stat]
    n_qualifying = int((df[qcol] >= qmin).sum())
    if n_qualifying < min_games:
        raise ValueError(f"need at least {min_games} qualifying games to fit {stat}, have {n_qualifying}")
    as_of = as_of or df["date"].max().date()

    players = sorted(df["player_id"].unique())
    teams = sorted(set(df["team"]) | set(df["opponent_team"]))
    groups = sorted(df["position_group"].unique())
    pidx = {p: i for i, p in enumerate(players)}
    tidx = {t: i for i, t in enumerate(teams)}
    gidx = {g: i for i, g in enumerate(groups)}
    n_p, n_t, n_g, n = len(players), len(teams), len(groups), len(df)

    y = np.maximum(df[STAT_COLUMN[stat]].to_numpy(dtype=float), 0.0)  # counts are never negative
    days_ago = (pd.Timestamp(as_of) - df["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    has_share = stat in SHARE_STATS and "trailing_share" in df.columns
    extra_covariates = extra_covariates or []
    missing = [c for c in extra_covariates if c not in df.columns]
    if missing:
        raise ValueError(f"extra_covariates column(s) not in stats: {missing}")
    non_finite = [c for c in extra_covariates if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
    if non_finite:
        raise ValueError(f"extra_covariates column(s) contain non-finite values: {non_finite}")
    if len(extra_covariates) != len(set(extra_covariates)):
        raise ValueError(f"extra_covariates contains duplicate column names: {extra_covariates}")
    if has_share and "trailing_share" in extra_covariates:
        raise ValueError('"trailing_share" is already a dedicated covariate for share stats; do not list it in extra_covariates')
    n_share_extra = 1 if has_share else 0
    n_extra = n_share_extra + len(extra_covariates)
    X = np.zeros((n, n_g + 1 + n_p + n_t + n_extra))
    gi = df["position_group"].map(gidx).to_numpy()
    X[np.arange(n), gi] = 1.0
    X[:, n_g] = df["home"].to_numpy(dtype=float)
    pi = df["player_id"].map(pidx).to_numpy()
    ti = df["opponent_team"].map(tidx).to_numpy()
    X[np.arange(n), n_g + 1 + pi] = 1.0
    X[np.arange(n), n_g + 1 + n_p + ti] = 1.0
    if has_share:
        X[:, n_g + 1 + n_p + n_t] = df["trailing_share"].to_numpy(dtype=float)
    extra_start = n_g + 1 + n_p + n_t + n_share_extra
    for i, col in enumerate(extra_covariates):
        X[:, extra_start + i] = df[col].to_numpy(dtype=float)

    penalty = np.zeros(X.shape[1])
    penalty[:n_g] = model.GROUP_INTERCEPT_REG
    penalty[n_g + 1:n_g + 1 + n_p + n_t] = reg

    # Initialize beta from a closed-form weighted ridge fit of log(y + 1) -- much closer
    # to the IRLS optimum than beta=0, so IRLS converges in a handful of iterations.
    y_log = np.log(y + 1.0)
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (y_log * w)
    beta = np.linalg.solve(XtWX + np.diag(penalty), XtWy)

    for _ in range(MAX_IRLS_ITER):
        eta = np.clip(X @ beta, -20, 20)
        mu = np.maximum(np.exp(eta), 1e-6)
        z = eta + (y - mu) / mu
        irls_w = w * mu
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
    defense = {t: float(beta[n_g + 1 + n_p + i]) for t, i in tidx.items()}
    share_coef = float(beta[n_g + 1 + n_p + n_t]) if has_share else None
    share_fallback: dict[str, float] = {}
    if has_share:
        share_fallback = {g: float(df.loc[df["position_group"] == g, "trailing_share"].mean())
                          for g in groups}
    extra_coefs = {col: float(beta[extra_start + i]) for i, col in enumerate(extra_covariates)}
    extra_fallback = {col: {g: float(df.loc[df["position_group"] == g, col].mean()) for g in groups}
                      for col in extra_covariates}
    group_counts = df["position_group"].value_counts().to_dict()
    intercept_fallback = float(np.average([position_intercept[g] for g in groups],
                                          weights=[group_counts[g] for g in groups]))

    final_eta = np.clip(X @ beta, -20, 20)
    final_mu = np.maximum(np.exp(final_eta), 1e-6)
    pearson_sq = (y - final_mu) ** 2 / final_mu
    group = df["position_group"].to_numpy()
    dispersion_global = max(1.0, float(np.average(pearson_sq, weights=w)))
    dispersion: dict[str, float] = {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() >= model.MIN_GROUP_RESIDUALS:
            dispersion[g] = max(1.0, float(np.average(pearson_sq[mask], weights=w[mask])))

    player_name = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    position_group = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()
    game_counts = df["player_id"].value_counts().to_dict()

    return RateRatings(
        stat=stat, players=players, player_name=player_name, position_group=position_group,
        ability=ability, teams=teams, defense=defense,
        position_intercept=position_intercept, intercept_fallback=intercept_fallback,
        share_coef=share_coef, share_fallback=share_fallback,
        home_field=home_field, dispersion=dispersion, dispersion_global=dispersion_global,
        as_of=as_of, n_games=n, game_counts=game_counts,
        extra_coefs=extra_coefs, extra_fallback=extra_fallback,
    )


def nb_params(lam: float | np.ndarray, dispersion: float | np.ndarray) -> tuple:
    """Reparameterize a (mean, quasi-Poisson dispersion) pair as a negative binomial's
    (r, p), matching the mean/variance of Var = dispersion * lam (NB2 parameterization).
    Only meaningful for dispersion > 1.0 -- callers gate on that themselves (see
    fantasy.py's _sample_quasi_poisson and rate_backtest.py's _nll/calibration for the
    dispersion <= 1.0 Poisson-branch check that stays at each call site).
    """
    r = lam / (dispersion - 1.0)
    p = r / (r + lam)
    return r, p


def predicted_rate(r: RateRatings, player_id: str, position_group: str, opponent_team: str,
                   home: bool, trailing_share: float | None = None,
                   extra_values: dict[str, float] | None = None) -> tuple[float, float]:
    """(lambda, dispersion): lambda is the expected count (Poisson mean); dispersion is
    the quasi-Poisson variance-inflation factor (Var = dispersion * lambda), always >= 1.
    Same unknown-player/unknown-group fallback treatment as model.predicted_distribution().
    """
    ability = r.ability.get(player_id)
    if ability is None:
        ability = 0.0
        if player_id not in r.prior_players:
            r.prior_players.append(player_id)
    defense = r.defense.get(opponent_team, 0.0)
    intercept = r.position_intercept.get(position_group, r.intercept_fallback)
    eta = intercept + ability + defense + (r.home_field if home else 0.0)
    if r.share_coef is not None:
        effective_share = (trailing_share if trailing_share is not None
                           else r.share_fallback.get(position_group, 0.0))
        eta += r.share_coef * effective_share
    extra_values = extra_values or {}
    for col, coef in r.extra_coefs.items():
        value = extra_values.get(col)
        if value is None:
            value = r.extra_fallback.get(col, {}).get(position_group, 0.0)
        eta += coef * value
    lam = float(np.exp(np.clip(eta, -20, 20)))
    disp = r.dispersion.get(position_group, r.dispersion_global)
    return lam, disp
