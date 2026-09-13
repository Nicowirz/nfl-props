"""Read-only Kalshi market data for NFL player-yardage props.

No authentication used or needed: Kalshi's `GET /events` and `GET /markets` endpoints
declare `security: []` in their own API spec (verified live 2026-09-13). This module
never places an order and never touches an authenticated Kalshi endpoint.

Event ticker format and field names verified live against real, currently-trading
markets on 2026-09-13 -- see docs/superpowers/specs/2026-09-13-nfl-props-kalshi-
integration-design.md.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from .parlay import Leg

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "kalshi"
MAX_AGE_MINUTES = 5.0

SERIES = {"pass_yds": "KXNFLPASSYDS", "rush_yds": "KXNFLRSHYDS", "rec_yds": "KXNFLRECYDS"}

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def event_ticker(series: str, gameday: date, away_team: str, home_team: str) -> str:
    """Build a Kalshi event ticker from a game's own schedule fields.

    Format verified live: {SERIES}-{YY}{MON}{DD}{AWAY}{HOME}, e.g.
    KXNFLRECYDS-26SEP13GBMIN.
    """
    return f"{series}-{gameday.strftime('%y')}{_MONTHS[gameday.month - 1]}{gameday.day:02d}{away_team}{home_team}"


def parse_player_name(title: str) -> str | None:
    """Extract the player's name from a market title, e.g. "Keon Coleman: 40+
    receiving yards" -> "Keon Coleman". Returns None if the title isn't in that shape."""
    if ":" not in title:
        return None
    name = title.split(":", 1)[0].strip()
    return name or None


def market_prices(market: dict) -> tuple[float, float, float] | None:
    """Extract (line, over_odds, under_odds) from one Kalshi market dict.

    line comes straight from floor_strike (already the threshold-minus-0.5 value this
    project's prob_over expects). over/under decimal odds come from the yes/no ask
    prices (dollar-denominated implied probabilities) via 1/price. Kalshi's live API
    returns these dollar prices as strings (e.g. "0.3800"), not numbers -- verified
    live 2026-09-13 -- so they're coerced through float() here rather than compared
    directly. Returns None if the market is missing any of these fields, or a price
    isn't a usable positive number.
    """
    floor_strike = market.get("floor_strike")
    yes_ask = market.get("yes_ask_dollars")
    no_ask = market.get("no_ask_dollars")
    if floor_strike is None or yes_ask is None or no_ask is None:
        return None
    try:
        yes_ask, no_ask = float(yes_ask), float(no_ask)
    except (TypeError, ValueError):
        return None
    if yes_ask <= 0 or no_ask <= 0:
        return None
    return float(floor_strike), 1.0 / yes_ask, 1.0 / no_ask


def legs_from_event(event: dict, stat: str) -> list[Leg]:
    """Parse every usable market in one Kalshi event's `markets` list into over/under
    Leg pairs. `event` is the *inner* event object (i.e. the value of the top-level
    "event" key Kalshi's `GET /events/{ticker}` response wraps everything in -- verified
    live 2026-09-13 -- not the raw response); fetch_prop_legs unwraps that layer.

    Player names are exactly as Kalshi writes them (title-cased, un-matched to any
    roster) -- matching against a roster's full_name happens one layer up, in
    fetch_prop_legs, since this function has no roster to match against.
    """
    legs: list[Leg] = []
    for market in event.get("markets", []):
        # A market for a game that already happened within the requested week (e.g. a
        # Thursday-night game, queried on the following Sunday) comes back "finalized"
        # with degenerate yes_ask == no_ask == 1.0 prices -- verified live 2026-09-13
        # against a real settled Sam Darnold passing-yards market. Only "active" markets
        # carry a real, currently-tradeable price.
        if market.get("status") != "active":
            continue
        title = market.get("title", "")
        player = parse_player_name(title)
        if player is None:
            continue
        prices = market_prices(market)
        if prices is None:
            continue
        line, over_odds, under_odds = prices
        legs.append(Leg(player, stat, "over", line, over_odds))
        legs.append(Leg(player, stat, "under", line, under_odds))
    return legs


def _fetch_event(event_ticker_str: str, refresh: bool) -> dict | None:
    """Fetch one event (with nested markets), cached on disk for MAX_AGE_MINUTES.

    Returns None if Kalshi has no such event (404 -- no market for this game/stat this
    week, which is common: not every game/stat combination is always listed).
    """
    cache = CACHE_DIR / f"{event_ticker_str}.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < MAX_AGE_MINUTES * 60
    if refresh or not fresh:
        url = f"{BASE_URL}/events/{event_ticker_str}?with_nested_markets=true"
        req = urllib.request.Request(url, headers={"User-Agent": "nfl-props/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                cache.write_bytes(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                if cache.exists():
                    cache.unlink()
                return None
            raise
    return json.loads(cache.read_text(encoding="utf-8"))


def fetch_prop_legs(games: pd.DataFrame, roster: pd.DataFrame, refresh: bool = False) -> tuple[list[Leg], int]:
    """Fetch Kalshi's live player-yardage-prop markets for a week's games.

    `games` must have home_team/away_team/gameday columns (e.g. the DataFrame from
    cli._upcoming_games). Player names from Kalshi are matched against `roster`'s
    full_name column case-insensitively -- an unmatched name is dropped, not a hard
    failure, since this pulls many players automatically (unlike a user-typed --odds
    CSV). Returns (legs, skipped_count).
    """
    roster_names = {n.lower(): n for n in roster["full_name"].dropna().unique()}
    legs: list[Leg] = []
    skipped = 0
    for _, g in games.iterrows():
        gameday = pd.Timestamp(g["gameday"]).date()
        for stat, series in SERIES.items():
            ticker = event_ticker(series, gameday, g["away_team"], g["home_team"])
            raw = _fetch_event(ticker, refresh)
            if raw is None:
                continue
            event = raw.get("event", {})
            for leg in legs_from_event(event, stat):
                matched = roster_names.get(leg.player.lower())
                if matched is None:
                    skipped += 1
                    continue
                legs.append(Leg(matched, leg.stat, leg.selection, leg.line, leg.odds))
    return legs, skipped
