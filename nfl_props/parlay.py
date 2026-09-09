"""Leg evaluation and parlay construction.

Legs are treated as independent (product of probabilities), unlike epl-parlay's shared
score-matrix joint probability -- there is no single joint distribution across different
players the way there is for one match's goals. Two legs from the same game may in reality
be correlated (game script, pace); that correlation is not modeled here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .markets import devig, prob_over

ASSUMED_OVERROUND = 1.05  # used when only one side of a line is supplied

STAT_WORDS = {
    "pass_yds": "pass_yds", "passing": "pass_yds", "pass": "pass_yds", "passing_yards": "pass_yds",
    "rush_yds": "rush_yds", "rushing": "rush_yds", "rush": "rush_yds", "rushing_yards": "rush_yds",
    "rec_yds": "rec_yds", "receiving": "rec_yds", "rec": "rec_yds", "receiving_yards": "rec_yds",
}


@dataclass(frozen=True)
class Leg:
    player: str
    stat: str        # 'pass_yds' | 'rush_yds' | 'rec_yds'
    selection: str    # 'over' | 'under'
    line: float
    odds: float       # decimal

    @property
    def label(self) -> str:
        return f"{self.player} {self.stat} {self.selection} {self.line}"

    def key(self) -> tuple:
        return (self.player, self.stat, self.line)


@dataclass
class LegEval:
    leg: Leg
    p_model: float
    p_book: float | None   # devigged book probability (None if other side unknown)
    p: float               # blended probability actually used
    devig_exact: bool

    @property
    def implied(self) -> float:
        return 1.0 / self.leg.odds

    @property
    def edge(self) -> float:
        return self.p - self.implied

    @property
    def ev(self) -> float:
        return self.p * self.leg.odds - 1


@dataclass
class Parlay:
    legs: tuple[LegEval, ...]
    prob: float
    odds: float

    @property
    def ev(self) -> float:
        return self.prob * self.odds - 1

    @property
    def kelly(self) -> float:
        return max(0.0, (self.prob * self.odds - 1) / (self.odds - 1))

    @property
    def label(self) -> str:
        return " + ".join(le.leg.label for le in self.legs)


def evaluate_legs(legs: Iterable[Leg], mu_sigma: dict[tuple[str, str], tuple[float, float]],
                  market_weight: float = 0.0) -> list[LegEval]:
    """Model probability, devigged book probability, and the blend for each leg.

    market_weight in [0, 1]: 0 = pure model, 1 = pure (devigged) book.
    """
    legs = list(legs)
    by_key: dict[tuple, dict[str, float]] = {}
    for lg in legs:
        by_key.setdefault(lg.key(), {})[lg.selection] = lg.odds
    out = []
    for lg in legs:
        mu, sigma = mu_sigma[(lg.player, lg.stat)]
        p_over = prob_over(mu, sigma, lg.line)
        p_model = p_over if lg.selection == "over" else 1 - p_over
        book = by_key[lg.key()]
        if "over" in book and "under" in book:
            p_over_book, p_under_book = devig([book["over"], book["under"]])
            p_book = p_over_book if lg.selection == "over" else p_under_book
            exact = True
        else:
            p_book, exact = (1.0 / lg.odds) / ASSUMED_OVERROUND, False
        p = (1 - market_weight) * p_model + market_weight * p_book
        out.append(LegEval(lg, p_model, p_book if exact else None, p, exact))
    return out


def build_parlays(evals: list[LegEval], min_legs: int = 2, max_legs: int = 4,
                  max_candidates: int = 12, min_edge: float = 0.0, top: int = 20) -> list[Parlay]:
    """Enumerate parlays from the best-edge legs and rank them by expected value."""
    cands = sorted((le for le in evals if le.edge > min_edge), key=lambda le: -le.edge)
    cands = cands[:max_candidates]
    parlays: list[Parlay] = []
    for k in range(min_legs, max_legs + 1):
        for combo in combinations(cands, k):
            if _contradictory(combo):
                continue
            p = 1.0
            odds = 1.0
            for le in combo:
                p *= le.p
                odds *= le.leg.odds
            parlays.append(Parlay(tuple(combo), p, odds))
    parlays.sort(key=lambda pl: -pl.ev)
    return parlays[:top]


def _contradictory(combo: Iterable[LegEval]) -> bool:
    """Two legs of the same player/stat/line in one parlay (e.g. over 250.5 + under 250.5),
    or two legs on the same player/stat at different lines -- correlated or potentially
    mutually exclusive (e.g. over 300 + under 200 for the same player's yardage)."""
    seen_keys = set()
    seen_player_stat = set()
    for le in combo:
        k = le.leg.key()
        ps = (le.leg.player, le.leg.stat)
        if k in seen_keys or ps in seen_player_stat:
            return True
        seen_keys.add(k)
        seen_player_stat.add(ps)
    return False


def legs_from_csv(path: str) -> list[Leg]:
    """Read a user's odds file: player,stat,line,over_odds,under_odds (decimal or American)."""
    import pandas as pd

    from .markets import parse_odds

    df = pd.read_csv(path, dtype={"over_odds": str, "under_odds": str})
    missing = {"player", "stat", "line"} - set(df.columns)
    if missing:
        raise ValueError(f"odds file is missing columns: {sorted(missing)}")
    legs: list[Leg] = []
    for _, row in df.iterrows():
        stat = STAT_WORDS.get(str(row["stat"]).strip().lower())
        if stat is None:
            raise ValueError(f"unknown stat {row['stat']!r} (use pass_yds, rush_yds or rec_yds)")
        player = str(row["player"]).strip()
        line = float(row["line"])
        if line <= 0:
            raise ValueError(f"line must be positive, got {line!r} for player {player!r}")
        if "over_odds" in df.columns and pd.notna(row.get("over_odds")):
            legs.append(Leg(player, stat, "over", line, parse_odds(row["over_odds"])))
        if "under_odds" in df.columns and pd.notna(row.get("under_odds")):
            legs.append(Leg(player, stat, "under", line, parse_odds(row["under_odds"])))
    if not legs:
        raise ValueError("odds file had no usable over_odds/under_odds values")
    return legs
