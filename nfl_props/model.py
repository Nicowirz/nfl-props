"""Weighted ridge regression on log(yards + OFFSET): position-group baseline + player
ability + opponent defense + home field, with exponential recency decay. Residuals are
normal (per position group), giving each player-game a log-normal yardage distribution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

STAT_COLUMN = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards"}
QUALIFY_COLUMN = {"pass_yds": "attempts", "rush_yds": "carries", "rec_yds": "targets",
                  "receptions": "targets"}  # reused by rate_model.py's usage-share covariate
QUALIFY_MIN = {"pass_yds": 10, "rush_yds": 5, "rec_yds": 2}
# position_group values that can plausibly produce this stat -- verified live 2026-09-13
# against real data. Bounds which players/rows enter the fit (a defensive lineman's rows
# are irrelevant to rush_yds). rec_yds deliberately excludes DB/DL despite each having a
# handful of real qualifying rows (a pick-six or gadget-play catch): those positions
# otherwise contribute ~15,000 structurally-always-zero rows (receiving yards is
# essentially impossible for a defensive player by rule) that swamp the real WR/TE/RB/QB
# signal and dilute the shared opponent-defense/home-field fit -- confirmed live, this
# alone flipped rec_yds from beating the backtest baseline to losing to it. rush_yds
# keeps WR (real jet-sweep/gadget-carry signal, not structural noise -- and needed for
# players like a return specialist that Kalshi lists a real rush_yds market for).
RELEVANT_POSITIONS = {
    "pass_yds": {"QB"},
    "rush_yds": {"QB", "RB", "TE", "WR"},
    "rec_yds": {"QB", "RB", "TE", "WR"},
}

# Validated against real 3-season data, 2026-09-14, 1-year backtest lookback (the same
# window the pace-adjustment feature was validated against). GATE PASSED for both stats:
#   rec_yds   NLL 0.8702 -> 0.8412  (~3.3% better)
#   rush_yds  NLL 0.1347 -> 0.1204  (~10.6% better)
# (Baseline "without" values must be computed with `game_id` dropped from the input --
# walk_forward() auto-activates the covariate whenever `game_id` is present, which real
# data always has, so a naive without/with comparison against real data silently
# compares the covariate against itself. See docs/superpowers/plans/2026-09-14-nfl-
# props-usage-share-covariate.md's Task 5 ledger entry for the full story.)
SHARE_STATS = {"rec_yds", "rush_yds"}  # stats with a real usage-share covariate; the
                                       # underlying numerator column is QUALIFY_COLUMN[stat]
                                       # (targets for rec_yds, carries for rush_yds) --
                                       # pass_yds has no analog and is deliberately absent.

# Validated against real 3-season data, 2026-09-16, 1-year backtest lookback, n=693 both
# arms (a true apples-to-apples comparison: WIDE_SIGMA_STATS temporarily disabled
# in-process to get a WITHOUT baseline on the identical data window, since a stale
# README snapshot at a different n is not a valid comparison). GATE FAILED:
#   NLL              1.3020 -> 1.2374  (~5% better -- criterion 1 PASSES)
#   top-decile bucket  2.89% -> 2.45%  (moved AWAY from the 10% target -- criterion 2 FAILS)
# Root cause: sigma_low_sample["QB"] came back 65% wider than the normal sigma["QB"]
# (1.3249 vs 0.7998 in one snapshot fit), affecting 29% of QBs (24/83) in that fit --
# a large, broadly-applied effect, not a narrow one targeting only genuine blowout
# games. A materially wider sigma compresses PIT values toward 0.5 for every low-sample
# prediction, which shrinks the already-too-small top-decile bucket further even as it
# improves average NLL (dominated by the bulk of ordinary predictions) and shrinks the
# already-too-large bottom bucket. The top-decile gap itself was a 3-prediction
# difference (17 vs. 20 of 693), on a data window whose baseline top-decile rate
# (~2.5-2.9%) was already far below the ~6.6% baseline the original low-sample-blowout
# diagnostic was built against -- so the gate result is real (criterion 2 genuinely
# failed on this window) but should be read as "this specific window/mechanism
# combination didn't resolve it," not as strong evidence the underlying diagnostic
# (backup QBs having underestimated blowout games) was wrong. A narrower mechanism
# (capped widening, a heavier-tailed distribution instead of a wider normal one, or
# triggering only on a documented role change) is a real candidate follow-up, out of
# this plan's scope.
#
# Per the plan's explicit gate, this stays inert by DEFAULT everywhere -- predict,
# parlay, best-bet never pass `stat` into predicted_distribution(), and backtest.py's
# walk_forward() only does so when called with wide_sigma=True (cli.py's `backtest
# --wide-sigma` flag; off by default). This mechanism is tested, working,
# currently-inert infrastructure (like pace.py's own rejected feature), gated behind a
# diagnostic-only flag for anyone who wants to re-run the validation later -- not wired
# into any live command's default behavior.
WIDE_SIGMA_STATS = {"pass_yds"}  # stats where a low-sample player's sigma is widened,
                                 # measured from real low-sample residuals -- rec_yds/
                                 # rush_yds already calibrate correctly at the top decile
                                 # (confirmed live 2026-09-16) and are deliberately absent.
                                 # NOTE: as calibrated, this does not currently improve
                                 # top-decile calibration (see comment above) and is only
                                 # ever activated via `backtest --wide-sigma` -- see the
                                 # gate result.

OFFSET = 10.0             # log(yards + OFFSET) stays finite even for a slightly negative rushing game
NEW_PLAYER_GAMES = 4      # fewer qualifying games than this -> flagged as low-sample in output
MIN_GROUP_RESIDUALS = 30  # fewer residuals than this in a position group -> fall back to sigma_global
GROUP_INTERCEPT_REG = 0.05  # near-zero: only guards against a singular solve if a group has 0 rows
                           # in a given fit window, not meant to meaningfully shrink real estimates


def _safe_log_yards(yards) -> np.ndarray:
    """log(yards + OFFSET), with yards floored so the log argument never drops to zero
    or below. A handful of real games have extreme negative yardage (a fumbled lateral,
    a punter's fake-punt carry) that would otherwise NaN out and poison the whole ridge
    fit; those are treated as equivalent to this floor rather than crashing.
    """
    raw = np.asarray(yards, dtype=float)
    return np.log(np.maximum(raw, 1.0 - OFFSET) + OFFSET)


def _weighted_share(shares: np.ndarray, days_ago: np.ndarray, halflife_days: float) -> float:
    w = 0.5 ** (days_ago / halflife_days)
    return float(np.sum(w * shares) / np.sum(w))


def add_trailing_share(stats: pd.DataFrame, stat: str, halflife_days: float = 180.0) -> pd.DataFrame:
    """Add a "trailing_share" column: each row's recency-weighted average share of its
    team's targets (rec_yds) or carries (rush_yds) over that PLAYER's own strictly-prior
    games only -- a row's own game never contributes to its own trailing_share, so this
    is leakage-safe by construction regardless of when it's later used.

    Team totals are summed across the whole roster for that game_id, not filtered to
    RELEVANT_POSITIONS -- a real usage share is a fraction of the team's true offensive
    output. Below NEW_PLAYER_GAMES prior games, trailing_share falls back to the
    row-count-weighted average among that position group's players who DO have enough
    history, mirroring how `Ratings.intercept_fallback` already handles the same class
    of "not enough of this specific player's own data" problem.

    Returns rows sorted by (player_id, date), not the caller's original order --
    callers that rely on positional/chronological indexing into the return value must
    re-sort first.
    """
    if "game_id" not in stats.columns:
        raise ValueError("add_trailing_share requires a 'game_id' column")
    share_col = QUALIFY_COLUMN[stat]
    df = stats.copy()
    team_total = df.groupby(["game_id", "team"])[share_col].transform("sum")
    df["_share"] = np.where(team_total > 0, df[share_col] / team_total, 0.0)
    df = df.sort_values(["player_id", "date"])

    trailing = np.full(len(df), np.nan)
    n_prior = np.zeros(len(df), dtype=int)
    row_pos = {idx: i for i, idx in enumerate(df.index)}
    for player_id, group in df.groupby("player_id", sort=False):
        dates = group["date"].to_numpy()
        shares = group["_share"].to_numpy(dtype=float)
        for i, idx in enumerate(group.index):
            pos = row_pos[idx]
            n_prior[pos] = i
            if i == 0:
                continue
            days_ago = (dates[i] - dates[:i]).astype("timedelta64[D]").astype(float)
            trailing[pos] = _weighted_share(shares[:i], days_ago, halflife_days)

    df = df.assign(trailing_share=trailing, _n_prior=n_prior)

    # Unlike the per-row computation above (leakage-safe by construction: a row's own
    # game never contributes to its own trailing_share), this group-level fallback is a
    # pooled statistic over the ENTIRE input frame. When called from backtest.py's
    # walk_forward(), that frame includes rows inside the current walk-forward TEST
    # window, not just the training window -- so a handful of low-history rows' fallback
    # values technically see a sliver of future data. Measured impact is negligible
    # (roughly 1/10,000 of one share value per contributing row) and is mostly absorbed
    # by the position-group intercept term elsewhere in the fit; this is an accepted,
    # measured-negligible exception to the leakage-safety principle above, not an
    # oversight.
    enough = df[df["_n_prior"] >= NEW_PLAYER_GAMES]
    group_fallback = enough.groupby("position_group")["trailing_share"].mean()
    overall_fallback = float(enough["trailing_share"].mean()) if len(enough) else 0.0
    low = df["_n_prior"] < NEW_PLAYER_GAMES
    df.loc[low, "trailing_share"] = df.loc[low, "position_group"].map(group_fallback).fillna(overall_fallback)
    df["trailing_share"] = df["trailing_share"].fillna(overall_fallback)

    return df.drop(columns=["_share", "_n_prior"])


def current_trailing_share(stats: pd.DataFrame, stat: str, player_id: str,
                           as_of: date | None = None, halflife_days: float = 180.0) -> float | None:
    """The live-prediction analog of add_trailing_share()'s per-row computation:
    `player_id`'s recency-weighted usage share over ALL of his games strictly before
    `as_of` (default: the most recent date in `stats`). Returns None if he has fewer
    than NEW_PLAYER_GAMES such games -- the caller applies the group-average fallback
    via predicted_distribution(), same as an unknown player's ability already defaults
    to 0.0 there.
    """
    share_col = QUALIFY_COLUMN[stat]
    player_rows = stats[stats["player_id"] == player_id]
    if player_rows.empty:
        return None
    team_total = stats.groupby(["game_id", "team"])[share_col].transform("sum")
    share = np.where(team_total.loc[player_rows.index] > 0,
                     player_rows[share_col] / team_total.loc[player_rows.index], 0.0)
    df = player_rows.assign(_share=share)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else df["date"].max() + pd.Timedelta(days=1)
    df = df[df["date"] < as_of_ts].sort_values("date")
    if len(df) < NEW_PLAYER_GAMES:
        return None
    days_ago = (as_of_ts - df["date"]).dt.days.to_numpy(dtype=float)
    return _weighted_share(df["_share"].to_numpy(dtype=float), days_ago, halflife_days)


@dataclass
class Ratings:
    stat: str
    players: list[str]
    player_name: dict[str, str]
    position_group: dict[str, str]
    ability: dict[str, float]
    teams: list[str]
    defense: dict[str, float]
    position_intercept: dict[str, float]  # baseline log(yards + OFFSET) per position group
    intercept_fallback: float             # row-count-weighted average, for a group unseen at fit time
    share_coef: float | None                # fitted usage-share sensitivity; None if not applicable
    share_fallback: dict[str, float]        # per-group average trailing_share at fit time; {} if share_coef is None
    home_field: float
    sigma: dict[str, float]
    sigma_global: float
    sigma_low_sample: dict[str, float]  # per position group, for WIDE_SIGMA_STATS only; {} otherwise
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
    """Fit ratings on every game from a stat-relevant position (RELEVANT_POSITIONS), not
    only games that clear QUALIFY_MIN, strictly before `as_of` (default: all rows).

    Restricting the fit to qualifying games only would systematically overstate every
    player's true output: a qualifying game is, by construction, one where the player
    got real volume, which correlates with getting more yards. Verified live
    2026-09-13: QB rushing rows with carries >= 5 average 36.2 yards vs. 16.0 across ALL
    of their rush-relevant games -- fitting on qualifying rows only taught the model
    that a typical QB rushing performance is more than double what it actually is.
    QUALIFY_MIN still gates the min_games sanity check below (enough real
    productive-game signal to fit reliably), and separately in backtest.py decides
    which games get SCORED (did the model predict well when a player had a real role)
    -- but it no longer decides which games train the fit.
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

    y = _safe_log_yards(df[STAT_COLUMN[stat]].to_numpy(dtype=float))
    days_ago = (pd.Timestamp(as_of) - df["date"]).dt.days.to_numpy(dtype=float)
    w = 0.5 ** (days_ago / halflife_days)

    # design matrix columns: [position-group intercept(n_g), home, player(n_p), opponent-defense(n_t)].
    # A per-group intercept (one-hot, no separate shared constant column -- that would be
    # exactly collinear with the group columns' sum) replaces a single global intercept so
    # that a low-sample player's ability, once shrunk toward 0 by ridge regularization,
    # falls back to THEIR position group's own baseline rather than a baseline dominated by
    # whichever group has the most qualifying rows (e.g. a scrambling QB's rush_yds rating
    # used to default toward a full-workload RB's baseline when his own carries were too
    # sparse to fit an ability of his own).
    has_share = stat in SHARE_STATS and "trailing_share" in df.columns
    n_extra = 1 if has_share else 0

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

    penalty = np.zeros(X.shape[1])
    penalty[:n_g] = GROUP_INTERCEPT_REG  # group intercepts: minimally regularized (see constant above)
    penalty[n_g + 1:n_g + 1 + n_p + n_t] = reg  # player ability + opponent defense; home field
                                                # (index n_g) and the share column (if present,
                                                # last index) both stay unregularized
    XtWX = X.T @ (X * w[:, None])
    XtWy = X.T @ (y * w)
    b = np.linalg.solve(XtWX + np.diag(penalty), XtWy)

    position_intercept = {g: float(b[i]) for g, i in gidx.items()}
    home_field = float(b[n_g])
    ability = {p: float(b[n_g + 1 + i]) for p, i in pidx.items()}
    defense = {t: float(b[n_g + 1 + n_p + i]) for t, i in tidx.items()}
    share_coef = float(b[n_g + 1 + n_p + n_t]) if has_share else None
    share_fallback: dict[str, float] = {}
    if has_share:
        share_fallback = {g: float(df.loc[df["position_group"] == g, "trailing_share"].mean())
                          for g in groups}
    group_counts = df["position_group"].value_counts().to_dict()
    intercept_fallback = float(np.average([position_intercept[g] for g in groups],
                                          weights=[group_counts[g] for g in groups]))

    resid = y - X @ b
    sigma_global = float(np.sqrt(np.average(resid ** 2, weights=w)))
    group = df["position_group"].to_numpy()
    sigma: dict[str, float] = {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() >= MIN_GROUP_RESIDUALS:
            sigma[g] = float(np.sqrt(np.average(resid[mask] ** 2, weights=w[mask])))

    game_counts = df["player_id"].value_counts().to_dict()

    sigma_low_sample: dict[str, float] = {}
    if stat in WIDE_SIGMA_STATS:
        is_low_sample = df["player_id"].map(game_counts).to_numpy() < NEW_PLAYER_GAMES
        for g in np.unique(group):
            mask = (group == g) & is_low_sample
            if mask.sum() >= MIN_GROUP_RESIDUALS:
                sigma_low_sample[g] = float(np.sqrt(np.average(resid[mask] ** 2, weights=w[mask])))
            elif g in sigma:
                sigma_low_sample[g] = sigma[g]
            else:
                sigma_low_sample[g] = sigma_global

    player_name = df.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    position_group = df.drop_duplicates("player_id").set_index("player_id")["position_group"].to_dict()

    return Ratings(
        stat=stat, players=players, player_name=player_name, position_group=position_group,
        ability=ability, teams=teams, defense=defense,
        position_intercept=position_intercept, intercept_fallback=intercept_fallback,
        share_coef=share_coef, share_fallback=share_fallback,
        home_field=home_field, sigma=sigma, sigma_global=sigma_global,
        sigma_low_sample=sigma_low_sample, as_of=as_of,
        n_games=n, game_counts=game_counts,
    )


def predicted_distribution(r: Ratings, player_id: str, position_group: str, opponent_team: str,
                           home: bool, trailing_share: float | None = None,
                           stat: str = "") -> tuple[float, float]:
    """(mu, sigma) of log(yards + OFFSET) for a player-game. Unknown players get their
    position group's average ability (0.0, relative to that group's own intercept) and
    are flagged in `r.prior_players`, same treatment epl-parlay gives a club with no
    fitted history. A position group never seen at fit time falls back to
    `r.intercept_fallback` (the row-count-weighted average across fitted groups).

    `trailing_share` is ignored entirely when `r.share_coef is None` (a stat with no
    usage-share covariate, or a fit that didn't have the trailing_share column
    available). When `r.share_coef` IS set and `trailing_share` is omitted (None), the
    position group's fitted average trailing_share (`r.share_fallback`) is used instead
    -- the same "assume average when we don't know" treatment ability already gets for
    an unknown player.

    `stat` selects a widened `sigma_low_sample` for a low-sample player (fewer than
    NEW_PLAYER_GAMES games at fit time, including an entirely unknown player) when
    `stat in WIDE_SIGMA_STATS` -- omitting `stat` (the default, "") never matches
    WIDE_SIGMA_STATS, reproducing the exact sigma this function always returned before
    this parameter existed.
    """
    ability = r.ability.get(player_id)
    if ability is None:
        ability = 0.0
        if player_id not in r.prior_players:
            r.prior_players.append(player_id)
    defense = r.defense.get(opponent_team, 0.0)
    intercept = r.position_intercept.get(position_group, r.intercept_fallback)
    mu = intercept + ability + defense + (r.home_field if home else 0.0)
    if r.share_coef is not None:
        effective_share = (trailing_share if trailing_share is not None
                           else r.share_fallback.get(position_group, 0.0))
        mu += r.share_coef * effective_share
    if stat in WIDE_SIGMA_STATS and r.game_counts.get(player_id, 0) < NEW_PLAYER_GAMES:
        sigma = r.sigma_low_sample.get(position_group, r.sigma.get(position_group, r.sigma_global))
    else:
        sigma = r.sigma.get(position_group, r.sigma_global)
    return mu, sigma
