"""Command line: update | ratings | predict | parlay | backtest."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import backtest, data, game_backtest, game_model, kalshi, model, pace, tracking
from .game_markets import moneyline_prob
from .markets import fair_odds, mean_yards, median_yards, parse_odds, prob_over, to_american
from .parlay import (GameLeg, Leg, build_game_parlays, build_parlays, evaluate_game_legs,
                     evaluate_legs, game_legs_from_csv, legs_from_csv)

STATS = ("pass_yds", "rush_yds", "rec_yds")
POSITION_GROUPS = {"pass_yds": {"QB"}, "rush_yds": {"RB", "FB", "QB"}, "rec_yds": {"WR", "TE", "RB"}}

DEFAULTS = {"seasons": 3, "halflife": 180.0, "reg": 5.0, "market_weight": 0.5,
           "game_halflife": 365.0, "game_reg": 3.0}


def _load(args) -> pd.DataFrame:
    return data.load_player_stats(args.seasons, refresh=args.refresh)


def cmd_update(args):
    stats = data.load_player_stats(args.seasons, refresh=True)
    games = data.load_games(refresh=True)
    season = data.current_season_start()
    roster = data.load_rosters(season, refresh=True)
    print(f"{len(stats)} player-games, {len(games)} scheduled games, "
          f"{len(roster)} roster entries for {season}")


def cmd_ratings(args):
    stats = _load(args)
    for stat in STATS:
        fit_stats = (model.add_trailing_share(stats, stat, halflife_days=args.halflife)
                    if stat in model.SHARE_STATS else stats)
        r = model.fit(fit_stats, stat, halflife_days=args.halflife, reg=args.reg)
        print(f"\n=== {stat} (as of {r.as_of}, {r.n_games} games) ===")
        print(r.table().head(args.top).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


def _upcoming_games(args) -> pd.DataFrame:
    games = data.load_games(refresh=args.refresh)
    upcoming = games[(games["season"] == args.season) & (games["week"] == args.week)]
    if upcoming.empty:
        sys.exit(f"no scheduled games for season {args.season} week {args.week}")
    return upcoming


def _roster_for_stat(roster: pd.DataFrame, stat: str) -> pd.DataFrame:
    return roster[roster["position"].isin(POSITION_GROUPS[stat]) & (roster["status"] == "ACT")]


def cmd_predict(args):
    stats = _load(args)
    games = _upcoming_games(args)
    roster = data.load_rosters(args.season, args.week, refresh=args.refresh)
    roster_names = {p["player_id"]: p["full_name"] for _, p in roster.iterrows()}
    for stat in STATS:
        fit_stats = (model.add_trailing_share(stats, stat, halflife_days=args.halflife)
                    if stat in model.SHARE_STATS else stats)
        r = model.fit(fit_stats, stat, halflife_days=args.halflife, reg=args.reg)
        cand = _roster_for_stat(roster, stat)
        rows = []
        for _, g in games.iterrows():
            for team, opp, home in ((g["home_team"], g["away_team"], True),
                                    (g["away_team"], g["home_team"], False)):
                for _, p in cand[cand["team"] == team].iterrows():
                    ts = (model.current_trailing_share(stats, stat, p["player_id"], halflife_days=args.halflife)
                         if stat in model.SHARE_STATS else None)
                    mu, sigma = model.predicted_distribution(r, p["player_id"], p["position"], opp, home,
                                                             trailing_share=ts, stat=stat)
                    med = median_yards(mu)
                    if pd.isna(med):
                        # Upstream model.fit() can produce NaN ratings for a stat if a
                        # single row's raw stat value breaks the log(yards + OFFSET)
                        # assumption (e.g. a real receiving_yards value below -OFFSET).
                        # Skip rather than crash on round(nan); see task-6-report.md.
                        continue
                    line = round(med / args.line_step) * args.line_step
                    rows.append({
                        "player": p["full_name"], "team": team, "opp": opp,
                        "mean": mean_yards(mu, sigma), "median": med,
                        "line": line, "p_over": prob_over(mu, sigma, line),
                    })
        print(f"\n=== {stat} projections, week {args.week} ===")
        if not rows:
            print("  no candidates found on either roster")
            continue
        out = pd.DataFrame(rows).sort_values("mean", ascending=False)
        print(out.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        if r.prior_players:
            names = [roster_names.get(p, p) for p in r.prior_players]
            shown = ", ".join(names[:10])
            if len(names) > 10:
                shown += f" (+{len(names) - 10} more)"
            print(f"  no history: {shown}")


def _mu_sigma_for_legs(legs: list[Leg], stats: pd.DataFrame, roster: pd.DataFrame,
                       games: pd.DataFrame, args) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    fitted: dict[str, model.Ratings] = {}
    for lg in legs:
        if lg.stat not in fitted:
            fit_stats = (model.add_trailing_share(stats, lg.stat, halflife_days=args.halflife)
                        if lg.stat in model.SHARE_STATS else stats)
            fitted[lg.stat] = model.fit(fit_stats, lg.stat, halflife_days=args.halflife, reg=args.reg)
        r = fitted[lg.stat]
        row = roster[roster["full_name"].str.lower() == lg.player.lower()]
        if row.empty:
            sys.exit(f"player not found on the season {args.season} week {args.week} roster: {lg.player!r}")
        row = row.iloc[0]
        team = row["team"]
        g = games[(games["home_team"] == team) | (games["away_team"] == team)]
        if g.empty:
            sys.exit(f"{team} has no scheduled game in season {args.season} week {args.week}")
        g = g.iloc[0]
        home = bool(g["home_team"] == team)
        opp = g["away_team"] if home else g["home_team"]
        ts = (model.current_trailing_share(stats, lg.stat, row["player_id"], halflife_days=args.halflife)
             if lg.stat in model.SHARE_STATS else None)
        mu, sigma = model.predicted_distribution(r, row["player_id"], row["position"], opp, home,
                                                 trailing_share=ts, stat=lg.stat)
        out[(lg.player, lg.stat)] = (mu, sigma)
    return out


def _feed_prop_legs(games: pd.DataFrame, roster: pd.DataFrame, args) -> list[Leg]:
    legs, skipped = kalshi.fetch_prop_legs(games, roster, refresh=args.refresh)
    matched = len({lg.player for lg in legs})
    print(f"Kalshi: {matched} props matched, {skipped} skipped (no roster match)\n")
    return legs


def cmd_parlay(args):
    stats = _load(args)
    games = _upcoming_games(args)
    roster = data.load_rosters(args.season, args.week, refresh=args.refresh)
    legs = [] if args.no_feed_odds else _feed_prop_legs(games, roster, args)
    if args.odds:
        user = legs_from_csv(args.odds)
        user_keys = {lg.key() for lg in user}
        legs = [lg for lg in legs if lg.key() not in user_keys] + user
    if not legs:
        sys.exit("no legs: the Kalshi feed had no odds and no --odds file was given")
    mu_sigma = _mu_sigma_for_legs(legs, stats, roster, games, args)
    evals = evaluate_legs(legs, mu_sigma, market_weight=args.market_weight)
    evals.sort(key=lambda e: -e.edge)

    print(f"Market weight {args.market_weight:.2f}\n")
    print(f"{'leg':45} {'odds':>6} {'amer':>6} {'p':>6} {'model':>6} {'book':>6} {'edge':>7}")
    for e in evals:
        book = f"{e.p_book:6.1%}" if e.p_book is not None else "   n/a"
        flag = "" if e.devig_exact else " *"
        print(f"{e.leg.label:45} {e.leg.odds:6.2f} {to_american(e.leg.odds):>6} {e.p:6.1%} "
              f"{e.p_model:6.1%} {book} {e.edge:+7.1%}{flag}")
    if any(not e.devig_exact for e in evals):
        print("  * other side not supplied; book prob approximated with a 5% margin")

    parlays = build_parlays(evals, min_legs=args.min_legs, max_legs=args.max_legs,
                            max_candidates=args.candidates, min_edge=args.min_edge, top=args.top)
    print(f"\nTop parlays ({args.min_legs}-{args.max_legs} legs, edge > {args.min_edge:.1%}):")
    if not parlays:
        print("  none: no leg clears the edge threshold. Try --min-edge 0.")
        return
    print(f"{'#':>2} {'odds':>8} {'amer':>7} {'p':>6} {'EV':>7} {'kelly':>6}  legs")
    for i, pl in enumerate(parlays, 1):
        print(f"{i:2d} {pl.odds:8.2f} {to_american(pl.odds):>7} {pl.prob:6.1%} {pl.ev:+7.1%} "
              f"{pl.kelly:6.1%}  {pl.label}")
    print("\nLegs are treated as independent even within the same game; real correlation "
          "(e.g. game script) is not modeled.")


def cmd_backtest(args):
    stats = data.load_player_stats(args.seasons, refresh=args.refresh)
    games = data.load_games(refresh=args.refresh) if args.pace_adjust else None
    if args.pace_adjust:
        coefs = ", ".join(f"{s}={v:.3f}" for s, v in pace.SENSITIVITY.items())
        print(f"Pace adjustment: ON ({coefs})")
    else:
        print("Pace adjustment: OFF")
    for stat in STATS:
        if args.start:
            start = pd.Timestamp(args.start).date()
        else:
            start = (stats["date"].max() - pd.Timedelta(days=365 * args.test_seasons)).date()
        preds = backtest.walk_forward(stats, stat, start, halflife_days=args.halflife, reg=args.reg,
                                      pace_adjust=args.pace_adjust, games=games,
                                      game_halflife_days=args.game_halflife, game_reg=args.game_reg,
                                      wide_sigma=args.wide_sigma)
        s = backtest.summarize(preds)
        print(f"\n=== {stat}: {s['n']} predictions from {start} ===")
        print(f"NLL model {s['nll_model']:.4f} vs baseline (season-to-date average) "
              f"{s['nll_baseline']:.4f} (lower is better)")
        print("Calibration (PIT should be ~10% per bucket if well-calibrated):")
        print(backtest.calibration(preds).to_string(float_format=lambda v: f"{v:.3f}"))


def cmd_game_ratings(args):
    games = data.load_games(refresh=args.refresh)
    mr = game_model.fit_margin(games, halflife_days=args.game_halflife, reg=args.game_reg)
    sr = game_model.fit_score(games, halflife_days=args.game_halflife, reg=args.game_reg)
    current_season = games["season"].max()
    current_teams = (set(games.loc[games["season"] == current_season, "home_team"])
                     | set(games.loc[games["season"] == current_season, "away_team"]))
    print(f"\n=== margin power ratings (as of {mr.as_of}, {mr.n_games} games, "
          f"home field {mr.home_field:+.2f}) ===")
    m_table = mr.table()
    m_table = m_table[m_table["team"].isin(current_teams)]
    print(m_table.head(args.top).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\n=== scoring/allowed ratings (as of {sr.as_of}, {sr.n_games} games) ===")
    s_table = sr.table()
    s_table = s_table[s_table["team"].isin(current_teams)]
    print(s_table.head(args.top).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


def _upcoming_completed_games(args) -> pd.DataFrame:
    games = data.load_games(refresh=args.refresh)
    upcoming = games[(games["season"] == args.season) & (games["week"] == args.week)]
    if upcoming.empty:
        sys.exit(f"no scheduled games for season {args.season} week {args.week}")
    return games, upcoming


def cmd_game_predict(args):
    games, upcoming = _upcoming_completed_games(args)
    mr = game_model.fit_margin(games, halflife_days=args.game_halflife, reg=args.game_reg)
    sr = game_model.fit_score(games, halflife_days=args.game_halflife, reg=args.game_reg)
    print(f"\n=== game predictions, week {args.week} ===")
    for _, g in upcoming.iterrows():
        margin_mu, margin_sigma = game_model.predicted_margin(mr, g["home_team"], g["away_team"],
                                                              neutral=bool(g["neutral"]))
        total_mu, total_sigma = game_model.predicted_total(sr, g["home_team"], g["away_team"])
        p_home = moneyline_prob(margin_mu, margin_sigma)
        print(f"\n{g['away_team']} @ {g['home_team']}")
        print(f"  moneyline: {g['home_team']} {p_home:5.1%} ({to_american(fair_odds(p_home))})  "
              f"{g['away_team']} {1 - p_home:5.1%} ({to_american(fair_odds(1 - p_home))})")
        print(f"  margin: {g['home_team']} by {margin_mu:+.1f} (sigma {margin_sigma:.1f})")
        print(f"  total: {total_mu:.1f} (sigma {total_sigma:.1f})")
    if mr.prior_teams or sr.prior_teams:
        names = sorted(set(mr.prior_teams) | set(sr.prior_teams))
        print(f"\nno history: {', '.join(names)}")


def _feed_game_legs(g: pd.Series) -> list[GameLeg]:
    legs = []
    if pd.notna(g.get("home_moneyline")):
        legs.append(GameLeg(g["home_team"], g["away_team"], "moneyline", "home", None,
                            parse_odds(g["home_moneyline"])))
    if pd.notna(g.get("away_moneyline")):
        legs.append(GameLeg(g["home_team"], g["away_team"], "moneyline", "away", None,
                            parse_odds(g["away_moneyline"])))
    if pd.notna(g.get("spread_line")) and pd.notna(g.get("home_spread_odds")):
        legs.append(GameLeg(g["home_team"], g["away_team"], "spread", "home", float(g["spread_line"]),
                            parse_odds(g["home_spread_odds"])))
    if pd.notna(g.get("spread_line")) and pd.notna(g.get("away_spread_odds")):
        legs.append(GameLeg(g["home_team"], g["away_team"], "spread", "away", float(g["spread_line"]),
                            parse_odds(g["away_spread_odds"])))
    if pd.notna(g.get("total_line")) and pd.notna(g.get("over_odds")):
        legs.append(GameLeg(g["home_team"], g["away_team"], "total", "over", float(g["total_line"]),
                            parse_odds(g["over_odds"])))
    if pd.notna(g.get("total_line")) and pd.notna(g.get("under_odds")):
        legs.append(GameLeg(g["home_team"], g["away_team"], "total", "under", float(g["total_line"]),
                            parse_odds(g["under_odds"])))
    return legs


def cmd_game_bets(args):
    games, upcoming = _upcoming_completed_games(args)
    mr = game_model.fit_margin(games, halflife_days=args.game_halflife, reg=args.game_reg)
    sr = game_model.fit_score(games, halflife_days=args.game_halflife, reg=args.game_reg)

    legs = [] if args.no_feed_odds else [lg for _, g in upcoming.iterrows() for lg in _feed_game_legs(g)]
    if args.odds:
        user = game_legs_from_csv(args.odds)
        user_keys = {lg.key() for lg in user}
        legs = [lg for lg in legs if lg.key() not in user_keys] + user
    if not legs:
        sys.exit("no legs: the feed had no odds and no --odds file was given")

    distributions = {}
    for _, g in upcoming.iterrows():
        margin_mu, margin_sigma = game_model.predicted_margin(mr, g["home_team"], g["away_team"],
                                                              neutral=bool(g["neutral"]))
        total_mu, total_sigma = game_model.predicted_total(sr, g["home_team"], g["away_team"])
        distributions[(g["home_team"], g["away_team"])] = (margin_mu, margin_sigma, total_mu, total_sigma)

    for lg in legs:
        if (lg.home_team, lg.away_team) not in distributions:
            valid = ", ".join(f"{h} vs {a}" for h, a in distributions) or "(none)"
            sys.exit(f"no scheduled game found for {lg.home_team} vs {lg.away_team} in "
                     f"season {args.season} week {args.week}. Valid games: {valid}")

    evals = evaluate_game_legs(legs, distributions, market_weight=args.market_weight)
    evals.sort(key=lambda e: -e.edge)
    print(f"Market weight {args.market_weight:.2f}\n")
    print(f"{'leg':40} {'odds':>6} {'amer':>6} {'p':>6} {'model':>6} {'book':>6} {'edge':>7}")
    for e in evals:
        book = f"{e.p_book:6.1%}" if e.p_book is not None else "   n/a"
        flag = "" if e.devig_exact else " *"
        print(f"{e.leg.label:40} {e.leg.odds:6.2f} {to_american(e.leg.odds):>6} {e.p:6.1%} "
              f"{e.p_model:6.1%} {book} {e.edge:+7.1%}{flag}")
    if any(not e.devig_exact for e in evals):
        print("  * other side not supplied; book prob approximated with a 5% margin")

    parlays = build_game_parlays(evals, min_legs=args.min_legs, max_legs=args.max_legs,
                                 max_candidates=args.candidates, min_edge=args.min_edge, top=args.top)
    print(f"\nTop parlays ({args.min_legs}-{args.max_legs} legs, edge > {args.min_edge:.1%}):")
    if not parlays:
        print("  none: no leg clears the edge threshold. Try --min-edge 0.")
        return
    print(f"{'#':>2} {'odds':>8} {'amer':>7} {'p':>6} {'EV':>7} {'kelly':>6}  legs")
    for i, pl in enumerate(parlays, 1):
        print(f"{i:2d} {pl.odds:8.2f} {to_american(pl.odds):>7} {pl.prob:6.1%} {pl.ev:+7.1%} "
              f"{pl.kelly:6.1%}  {pl.label}")
    print("\nNo named provider for the feed's reference line (see README) -- treat as a "
          "consensus/reference price, not proven beatable.")
    print("Legs from the same game are excluded from parlays entirely (moneyline, spread, "
          "and total outcomes for one game are correlated, not independent).")


def cmd_game_backtest(args):
    games = data.load_games(refresh=args.refresh)
    completed = games.dropna(subset=["home_score", "away_score"])
    if args.start:
        start = pd.Timestamp(args.start).date()
    else:
        start = (completed["gameday"].max() - pd.Timedelta(days=365 * args.test_seasons)).date()

    margin_preds = game_backtest.walk_forward_margin(games, start, halflife_days=args.game_halflife,
                                                      reg=args.game_reg)
    s = game_backtest.summarize(margin_preds, "actual_margin")
    print(f"\n=== margin: {s['n']} predictions from {start} ===")
    print(f"NLL model {s['nll_model']:.4f} vs baseline (home-field-only) {s['nll_baseline']:.4f} "
          f"(lower is better)")
    print("Calibration:")
    print(game_backtest.calibration(margin_preds, "actual_margin").to_string(float_format=lambda v: f"{v:.3f}"))

    total_preds = game_backtest.walk_forward_total(games, start, halflife_days=args.game_halflife,
                                                    reg=args.game_reg)
    s = game_backtest.summarize(total_preds, "actual_total")
    print(f"\n=== totals: {s['n']} predictions from {start} ===")
    print(f"NLL model {s['nll_model']:.4f} vs baseline (league-average) {s['nll_baseline']:.4f} "
          f"(lower is better)")
    print("Calibration:")
    print(game_backtest.calibration(total_preds, "actual_total").to_string(float_format=lambda v: f"{v:.3f}"))


def cmd_best_bet(args):
    games, upcoming = _upcoming_completed_games(args)

    # Game market: single highest-edge leg against the real games.csv reference line.
    mr = game_model.fit_margin(games, halflife_days=args.game_halflife, reg=args.game_reg)
    sr = game_model.fit_score(games, halflife_days=args.game_halflife, reg=args.game_reg)
    game_legs = [lg for _, g in upcoming.iterrows() for lg in _feed_game_legs(g)]
    distributions = {}
    for _, g in upcoming.iterrows():
        margin_mu, margin_sigma = game_model.predicted_margin(mr, g["home_team"], g["away_team"],
                                                              neutral=bool(g["neutral"]))
        total_mu, total_sigma = game_model.predicted_total(sr, g["home_team"], g["away_team"])
        distributions[(g["home_team"], g["away_team"])] = (margin_mu, margin_sigma, total_mu, total_sigma)
    game_evals = evaluate_game_legs(game_legs, distributions, market_weight=args.market_weight) if game_legs else []

    print(f"=== Best bet of the week (season {args.season}, week {args.week}) ===\n")
    log_rows = []
    print("Game market:")
    if game_evals:
        best_game = max(game_evals, key=lambda e: e.edge)
        print(f"  {best_game.leg.label}")
        print(f"  odds {best_game.leg.odds:.2f} ({to_american(best_game.leg.odds)})  "
              f"model {best_game.p_model:6.1%}  edge {best_game.edge:+.1%}  EV {best_game.ev:+.1%}")
        log_rows.append({
            "season": args.season, "week": args.week, "market": best_game.leg.market,
            "subject": f"{best_game.leg.away_team}@{best_game.leg.home_team}",
            "selection": best_game.leg.selection,
            "line": best_game.leg.line if best_game.leg.line is not None else 0.0,
            "odds": best_game.leg.odds, "p_model": best_game.p_model, "edge": best_game.edge,
        })
    else:
        print("  no legs available (games.csv had no odds for this week).")

    # Player prop: single highest-edge leg, from Kalshi's live feed by default (or
    # --odds, which overrides/supplements it; --no-feed-odds disables the feed).
    print("\nPlayer prop:")
    stats = _load(args)
    roster = data.load_rosters(args.season, args.week, refresh=args.refresh)
    prop_legs = [] if args.no_feed_odds else _feed_prop_legs(upcoming, roster, args)
    if args.odds:
        user = legs_from_csv(args.odds)
        user_keys = {lg.key() for lg in user}
        prop_legs = [lg for lg in prop_legs if lg.key() not in user_keys] + user
    if not prop_legs:
        print("  no player props available -- the Kalshi feed had none for this week "
              "(pass --odds file.csv to supply your own).")
    else:
        mu_sigma = _mu_sigma_for_legs(prop_legs, stats, roster, upcoming, args)
        prop_evals = evaluate_legs(prop_legs, mu_sigma, market_weight=args.market_weight)
        best_prop = max(prop_evals, key=lambda e: e.edge)
        print(f"  {best_prop.leg.label}")
        print(f"  odds {best_prop.leg.odds:.2f} ({to_american(best_prop.leg.odds)})  "
              f"model {best_prop.p_model:6.1%}  edge {best_prop.edge:+.1%}  EV {best_prop.ev:+.1%}")
        log_rows.append({
            "season": args.season, "week": args.week, "market": best_prop.leg.stat,
            "subject": best_prop.leg.player, "selection": best_prop.leg.selection,
            "line": best_prop.leg.line, "odds": best_prop.leg.odds,
            "p_model": best_prop.p_model, "edge": best_prop.edge,
        })

    if args.log and log_rows:
        tracking.log_picks(log_rows)
        print(f"\nLogged {len(log_rows)} pick(s) to {tracking.LOG_PATH} -- run "
              f"`grade` after these games complete to see how they did.")


def cmd_grade(args):
    stats = data.load_player_stats(args.seasons, refresh=args.refresh)
    games = data.load_games(refresh=args.refresh)
    df = tracking.grade_log(stats, games)
    if df.empty:
        print(f"no picks logged yet at {tracking.LOG_PATH} -- run `best-bet --log` first")
        return
    display = df.copy()
    for col in ("odds", "p_model", "edge"):
        display[col] = display[col].astype(float)
    print(display.to_string(index=False, formatters={
        "odds": "{:.2f}".format, "p_model": "{:.1%}".format, "edge": "{:+.1%}".format,
    }))
    s = tracking.summarize(df)
    print(f"\nRecord: {s['wins']}-{s['losses']}-{s['pushes']} (W-L-P), "
          f"{s['pending']} pending, win rate {s['win_rate']:.1%}, "
          f"flat-1-unit-stake ROI {s['roi']:+.1%}")


def main(argv=None):
    # Shared as a parent parser (not just added to `p`) so these options are accepted
    # both before AND after the subcommand, e.g. both `--week 2 predict` and
    # `predict --week 2` work -- argparse subparsers don't inherit a plain parent's
    # options once the subcommand token is consumed. See task-6-report.md self-review.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seasons", type=int, default=DEFAULTS["seasons"], help="seasons of history to fit on")
    common.add_argument("--halflife", type=float, default=DEFAULTS["halflife"], help="time-decay half-life in days")
    common.add_argument("--reg", type=float, default=DEFAULTS["reg"], help="ridge shrinkage on player/defense ratings")
    common.add_argument("--refresh", action="store_true", help="re-download data")
    common.add_argument("--season", type=int, default=data.current_season_start(), help="NFL season year")
    common.add_argument("--week", type=int, default=1, help="week number within --season")
    common.add_argument("--game-halflife", type=float, default=DEFAULTS["game_halflife"],
                        help="time-decay half-life in days for the game-outcome models")
    common.add_argument("--game-reg", type=float, default=DEFAULTS["game_reg"],
                        help="ridge shrinkage on team power/scoring/allowed ratings")

    p = argparse.ArgumentParser(prog="nfl_props", description="NFL yardage-props model", parents=[common])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("update", parents=[common], help="download latest stats, schedules, rosters").set_defaults(fn=cmd_update)

    rt = sub.add_parser("ratings", parents=[common], help="player ability + opponent defense ratings")
    rt.add_argument("--top", type=int, default=20)
    rt.set_defaults(fn=cmd_ratings)

    pr = sub.add_parser("predict", parents=[common], help="projected yardage distribution for --week")
    pr.add_argument("--top", type=int, default=10)
    pr.add_argument("--line-step", type=float, default=5.0)
    pr.set_defaults(fn=cmd_predict)

    pl = sub.add_parser("parlay", parents=[common], help="rank legs from Kalshi's live feed and build parlays")
    pl.add_argument("--odds", help="CSV: player,stat,line,over_odds,under_odds (overrides/supplements the Kalshi feed)")
    pl.add_argument("--no-feed-odds", action="store_true", help="ignore Kalshi's live prop feed")
    pl.add_argument("--market-weight", type=float, default=DEFAULTS["market_weight"])
    pl.add_argument("--min-legs", type=int, default=2)
    pl.add_argument("--max-legs", type=int, default=4)
    pl.add_argument("--min-edge", type=float, default=0.02)
    pl.add_argument("--candidates", type=int, default=12)
    pl.add_argument("--top", type=int, default=15)
    pl.set_defaults(fn=cmd_parlay)

    bt = sub.add_parser("backtest", parents=[common], help="walk-forward evaluation vs. a naive baseline")
    bt.add_argument("--start", help="YYYY-MM-DD; default = start of the last --test-seasons seasons")
    bt.add_argument("--test-seasons", type=int, default=1)
    bt.add_argument("--pace-adjust", action="store_true",
                    help="apply the game-pace adjustment (pace.pace_adjust) before scoring, "
                         "for comparison against the unadjusted model")
    bt.add_argument("--wide-sigma", action="store_true",
                    help="apply the low-sample QB sigma widening (model.WIDE_SIGMA_STATS) before "
                         "scoring, for comparison against the unwidened model -- diagnostic only, "
                         "a real-data validation gate failed for this mechanism; see model.py")
    bt.set_defaults(fn=cmd_backtest)

    gr = sub.add_parser("game-ratings", parents=[common], help="team power + scoring/allowed ratings")
    gr.add_argument("--top", type=int, default=32)
    gr.set_defaults(fn=cmd_game_ratings)

    gp = sub.add_parser("game-predict", parents=[common], help="moneyline/spread/total fair probabilities for --week")
    gp.set_defaults(fn=cmd_game_predict)

    gb = sub.add_parser("game-bets", parents=[common], help="edge vs. the real reference line, plus ranked parlays")
    gb.add_argument("--odds", help="CSV: home_team,away_team,market,selection,line,odds")
    gb.add_argument("--no-feed-odds", action="store_true", help="ignore games.csv's built-in reference line")
    gb.add_argument("--market-weight", type=float, default=DEFAULTS["market_weight"])
    gb.add_argument("--min-legs", type=int, default=2)
    gb.add_argument("--max-legs", type=int, default=4)
    gb.add_argument("--min-edge", type=float, default=0.02)
    gb.add_argument("--candidates", type=int, default=12)
    gb.add_argument("--top", type=int, default=15)
    gb.set_defaults(fn=cmd_game_bets)

    gbt = sub.add_parser("game-backtest", parents=[common], help="walk-forward evaluation for margin and totals")
    gbt.add_argument("--start", help="YYYY-MM-DD; default = start of the last --test-seasons seasons")
    gbt.add_argument("--test-seasons", type=int, default=1)
    gbt.set_defaults(fn=cmd_game_backtest)

    bb = sub.add_parser("best-bet", parents=[common],
                        help="single highest-edge pick, one per market (game + player prop)")
    bb.add_argument("--odds", help="CSV: player,stat,line,over_odds,under_odds (overrides/supplements the Kalshi feed)")
    bb.add_argument("--no-feed-odds", action="store_true", help="ignore Kalshi's live prop feed")
    bb.add_argument("--market-weight", type=float, default=DEFAULTS["market_weight"])
    bb.add_argument("--log", action="store_true", help="append this week's pick(s) to data/picks_log.csv for grading later")
    bb.set_defaults(fn=cmd_best_bet)

    gd = sub.add_parser("grade", parents=[common],
                        help="grade logged best-bet picks against real results and show the running record")
    gd.set_defaults(fn=cmd_grade)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
