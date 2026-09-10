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


from .game_markets import prob_margin_over, prob_total_over

GAME_MARKET_WORDS = {
    "moneyline": "moneyline", "ml": "moneyline",
    "spread": "spread", "ats": "spread",
    "total": "total", "totals": "total", "ou": "total", "o/u": "total",
}


@dataclass(frozen=True)
class GameLeg:
    home_team: str
    away_team: str
    market: str         # 'moneyline' | 'spread' | 'total'
    selection: str       # 'home' | 'away' (moneyline/spread) | 'over' | 'under' (total)
    line: float | None    # None for moneyline; the spread_line or total_line otherwise
    odds: float           # decimal

    @property
    def label(self) -> str:
        matchup = f"{self.away_team} @ {self.home_team}"
        if self.market == "moneyline":
            pick = self.home_team if self.selection == "home" else self.away_team
            return f"{matchup}: {pick} ML"
        if self.market == "spread":
            # self.line is nflverse's home-perspective convention (positive = home
            # favored). Standard bettor notation shows the FAVORITE with a minus sign,
            # so the home side's displayed number is the negation of self.line while
            # the away side's is self.line unchanged.
            pick = self.home_team if self.selection == "home" else self.away_team
            value = -self.line if self.selection == "home" else self.line
            return f"{matchup}: {pick} {value:+g}"
        return f"{matchup}: {self.selection} {self.line}"

    def key(self) -> tuple:
        return (self.home_team, self.away_team, self.market, self.line)


@dataclass
class GameLegEval:
    leg: GameLeg
    p_model: float
    p_book: float | None
    p: float
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
class GameParlay:
    legs: tuple[GameLegEval, ...]
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


def game_leg_model_prob(leg: GameLeg, margin_mu: float, margin_sigma: float,
                        total_mu: float, total_sigma: float) -> float:
    """P(this leg's selection wins) under the fitted margin/total distributions.

    spread_line and moneyline use nflverse's convention: positive means the HOME team is
    favored by that many points (margin = home_score - away_score). Home covers/wins iff
    margin > line (line = 0.0 for moneyline).
    """
    if leg.market in ("moneyline", "spread"):
        line = 0.0 if leg.market == "moneyline" else leg.line
        p_home = prob_margin_over(margin_mu, margin_sigma, line)
        return p_home if leg.selection == "home" else 1 - p_home
    p_over = prob_total_over(total_mu, total_sigma, leg.line)
    return p_over if leg.selection == "over" else 1 - p_over


def evaluate_game_legs(legs: Iterable[GameLeg],
                       distributions: dict[tuple[str, str], tuple[float, float, float, float]],
                       market_weight: float = 0.0) -> list[GameLegEval]:
    """Model probability, devigged book probability, and the blend for each leg.

    `distributions` maps (home_team, away_team) -> (margin_mu, margin_sigma, total_mu,
    total_sigma), precomputed by the caller -- this module never imports game_model.py
    directly, same decoupling as evaluate_legs/model.py.
    """
    legs = list(legs)
    by_key: dict[tuple, dict[str, float]] = {}
    for lg in legs:
        by_key.setdefault(lg.key(), {})[lg.selection] = lg.odds
    out = []
    for lg in legs:
        margin_mu, margin_sigma, total_mu, total_sigma = distributions[(lg.home_team, lg.away_team)]
        p_model = game_leg_model_prob(lg, margin_mu, margin_sigma, total_mu, total_sigma)
        book = by_key[lg.key()]
        sides = ("home", "away") if lg.market in ("moneyline", "spread") else ("over", "under")
        if sides[0] in book and sides[1] in book:
            p0, p1 = devig([book[sides[0]], book[sides[1]]])
            p_book = p0 if lg.selection == sides[0] else p1
            exact = True
        else:
            p_book, exact = (1.0 / lg.odds) / ASSUMED_OVERROUND, False
        p = (1 - market_weight) * p_model + market_weight * p_book
        out.append(GameLegEval(lg, p_model, p_book if exact else None, p, exact))
    return out


def build_game_parlays(evals: list[GameLegEval], min_legs: int = 2, max_legs: int = 4,
                       max_candidates: int = 12, min_edge: float = 0.0, top: int = 20) -> list[GameParlay]:
    """Enumerate parlays from the best-edge legs and rank them by expected value."""
    cands = sorted((le for le in evals if le.edge > min_edge), key=lambda le: -le.edge)
    cands = cands[:max_candidates]
    parlays: list[GameParlay] = []
    for k in range(min_legs, max_legs + 1):
        for combo in combinations(cands, k):
            if _game_contradictory(combo):
                continue
            p = 1.0
            odds = 1.0
            for le in combo:
                p *= le.p
                odds *= le.leg.odds
            parlays.append(GameParlay(tuple(combo), p, odds))
    parlays.sort(key=lambda pl: -pl.ev)
    return parlays[:top]


def _game_contradictory(combo: Iterable[GameLegEval]) -> bool:
    """Two legs on the same game (any market) are excluded from one parlay -- moneyline,
    spread, and total outcomes for one game are all correlated with each other (e.g. a
    moneyline pick is a near-subset of that team covering a positive spread), not
    independent."""
    seen = set()
    for le in combo:
        k = (le.leg.home_team, le.leg.away_team)
        if k in seen:
            return True
        seen.add(k)
    return False


def game_legs_from_csv(path: str) -> list[GameLeg]:
    """Read a user's odds file: home_team,away_team,market,selection,line,odds."""
    import pandas as pd

    from .markets import parse_odds

    df = pd.read_csv(path, dtype={"odds": str})
    missing = {"home_team", "away_team", "market", "selection", "odds"} - set(df.columns)
    if missing:
        raise ValueError(f"odds file is missing columns: {sorted(missing)}")
    legs: list[GameLeg] = []
    for _, row in df.iterrows():
        market = GAME_MARKET_WORDS.get(str(row["market"]).strip().lower())
        if market is None:
            raise ValueError(f"unknown market {row['market']!r} (use moneyline, spread or total)")
        sel = str(row["selection"]).strip().lower()
        if market in ("moneyline", "spread"):
            if sel not in ("home", "away"):
                raise ValueError(f"selection {row['selection']!r} must be home or away for {market}")
        else:
            if sel not in ("over", "under"):
                raise ValueError(f"selection {row['selection']!r} must be over or under for total")
        line = float(row["line"]) if "line" in df.columns and pd.notna(row.get("line")) else None
        if market != "moneyline" and line is None:
            raise ValueError(f"{market} leg needs a line: {row.to_dict()}")
        legs.append(GameLeg(str(row["home_team"]).strip(), str(row["away_team"]).strip(),
                            market, sel, line, parse_odds(row["odds"])))
    return legs
