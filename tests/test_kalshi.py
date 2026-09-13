from datetime import date

import pytest

from nfl_props.kalshi import event_ticker, legs_from_event, market_prices, parse_player_name


def test_event_ticker_matches_verified_live_format():
    # KXNFLRECYDS-26SEP13GBMIN: Packers @ Vikings, 2026-09-13, verified live.
    assert event_ticker("KXNFLRECYDS", date(2026, 9, 13), "GB", "MIN") == "KXNFLRECYDS-26SEP13GBMIN"


def test_parse_player_name_from_real_title_shape():
    assert parse_player_name("Keon Coleman: 40+ receiving yards") == "Keon Coleman"


def test_parse_player_name_none_without_colon():
    assert parse_player_name("garbage title with no colon") is None


def test_market_prices_converts_dollar_prices_to_decimal_odds():
    market = {"title": "Keon Coleman: 40+ receiving yards", "floor_strike": 39.5,
             "yes_ask_dollars": 0.08, "no_ask_dollars": 0.95}
    line, over_odds, under_odds = market_prices(market)
    assert line == pytest.approx(39.5)
    assert over_odds == pytest.approx(1 / 0.08)
    assert under_odds == pytest.approx(1 / 0.95)


def test_market_prices_accepts_string_dollar_prices():
    # Kalshi's live API returns yes_ask_dollars/no_ask_dollars as strings (e.g.
    # "0.3800"), not numbers -- verified live 2026-09-13.
    market = {"floor_strike": 14.5, "yes_ask_dollars": "0.6400", "no_ask_dollars": "0.3800"}
    line, over_odds, under_odds = market_prices(market)
    assert line == pytest.approx(14.5)
    assert over_odds == pytest.approx(1 / 0.64)
    assert under_odds == pytest.approx(1 / 0.38)


@pytest.mark.parametrize("missing_field", ["floor_strike", "yes_ask_dollars", "no_ask_dollars"])
def test_market_prices_none_when_field_missing(missing_field):
    market = {"title": "x", "floor_strike": 39.5, "yes_ask_dollars": 0.08, "no_ask_dollars": 0.95}
    del market[missing_field]
    assert market_prices(market) is None


def test_market_prices_none_when_price_is_zero():
    market = {"floor_strike": 39.5, "yes_ask_dollars": 0.0, "no_ask_dollars": 0.95}
    assert market_prices(market) is None


def test_legs_from_event_builds_over_and_under_pair_per_market():
    event = {"markets": [
        {"title": "Keon Coleman: 40+ receiving yards", "floor_strike": 39.5, "status": "active",
         "yes_ask_dollars": 0.08, "no_ask_dollars": 0.95},
        {"title": "Josh Oliver: 25+ receiving yards", "floor_strike": 24.5, "status": "active",
         "yes_ask_dollars": 0.30, "no_ask_dollars": 0.73},
    ]}
    legs = legs_from_event(event, "rec_yds")
    assert len(legs) == 4
    coleman_over = next(lg for lg in legs if lg.player == "Keon Coleman" and lg.selection == "over")
    assert coleman_over.stat == "rec_yds"
    assert coleman_over.line == pytest.approx(39.5)
    assert coleman_over.odds == pytest.approx(1 / 0.08)


def test_legs_from_event_skips_markets_with_unparseable_title():
    event = {"markets": [{"title": "no colon here", "floor_strike": 10.5, "status": "active",
                          "yes_ask_dollars": 0.5, "no_ask_dollars": 0.5}]}
    assert legs_from_event(event, "rush_yds") == []


def test_legs_from_event_skips_markets_missing_prices():
    event = {"markets": [{"title": "A Player: 10+ rushing yards", "floor_strike": 9.5, "status": "active"}]}
    assert legs_from_event(event, "rush_yds") == []


def test_legs_from_event_skips_non_active_markets():
    # A market for a game already played within the requested week comes back
    # "finalized" with degenerate yes_ask == no_ask == 1.0 prices -- verified live
    # 2026-09-13 against a real settled market. Must be excluded, not parsed as a leg.
    event = {"markets": [{"title": "Sam Darnold: 150+ passing yards", "floor_strike": 149.5,
                          "status": "finalized", "yes_ask_dollars": "1.0000", "no_ask_dollars": "1.0000"}]}
    assert legs_from_event(event, "pass_yds") == []
