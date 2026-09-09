"""Command line: update | ratings | predict | parlay | backtest."""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from . import backtest, data, model
from .markets import mean_yards, median_yards, prob_over, to_american
from .parlay import Leg, build_parlays, evaluate_legs, legs_from_csv

STATS = ("pass_yds", "rush_yds", "rec_yds")
POSITION_GROUPS = {"pass_yds": {"QB"}, "rush_yds": {"RB", "FB", "QB"}, "rec_yds": {"WR", "TE", "RB"}}

DEFAULTS = {"seasons": 3, "halflife": 180.0, "reg": 5.0, "market_weight": 0.5}


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
        r = model.fit(stats, stat, halflife_days=args.halflife, reg=args.reg)
        print(f"\n=== {stat} (as of {r.as_of}, {r.n_games} qualifying games) ===")
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
    for stat in STATS:
        r = model.fit(stats, stat, halflife_days=args.halflife, reg=args.reg)
        cand = _roster_for_stat(roster, stat)
        rows = []
        for _, g in games.iterrows():
            for team, opp, home in ((g["home_team"], g["away_team"], True),
                                    (g["away_team"], g["home_team"], False)):
                for _, p in cand[cand["team"] == team].iterrows():
                    mu, sigma = model.predicted_distribution(r, p["player_id"], p["position"], opp, home)
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
                        f"p_over_{line:g}": prob_over(mu, sigma, line),
                    })
        print(f"\n=== {stat} projections, week {args.week} ===")
        if not rows:
            print("  no candidates found on either roster")
            continue
        out = pd.DataFrame(rows).sort_values("mean", ascending=False)
        print(out.head(args.top).to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        if r.prior_players:
            names = ", ".join(r.player_name.get(p, p) for p in r.prior_players)
            print(f"  no history: {names}")


def _mu_sigma_for_legs(legs: list[Leg], stats: pd.DataFrame, roster: pd.DataFrame,
                       games: pd.DataFrame, args) -> dict[tuple[str, str], tuple[float, float]]:
    out = {}
    fitted: dict[str, model.Ratings] = {}
    for lg in legs:
        if lg.stat not in fitted:
            fitted[lg.stat] = model.fit(stats, lg.stat, halflife_days=args.halflife, reg=args.reg)
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
        mu, sigma = model.predicted_distribution(r, row["player_id"], row["position"], opp, home)
        out[(lg.player, lg.stat)] = (mu, sigma)
    return out


def cmd_parlay(args):
    stats = _load(args)
    games = _upcoming_games(args)
    roster = data.load_rosters(args.season, args.week, refresh=args.refresh)
    legs = legs_from_csv(args.odds)
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
    for stat in STATS:
        if args.start:
            start = pd.Timestamp(args.start).date()
        else:
            start = (stats["date"].max() - pd.Timedelta(days=365 * args.test_seasons)).date()
        preds = backtest.walk_forward(stats, stat, start, halflife_days=args.halflife, reg=args.reg)
        s = backtest.summarize(preds)
        print(f"\n=== {stat}: {s['n']} predictions from {start} ===")
        print(f"NLL model {s['nll_model']:.4f} vs baseline (season-to-date average) "
              f"{s['nll_baseline']:.4f} (lower is better)")
        print("Calibration (PIT should be ~10% per bucket if well-calibrated):")
        print(backtest.calibration(preds).to_string(float_format=lambda v: f"{v:.3f}"))


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

    pl = sub.add_parser("parlay", parents=[common], help="rank legs from --odds and build parlays")
    pl.add_argument("--odds", required=True, help="CSV: player,stat,line,over_odds,under_odds")
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
    bt.set_defaults(fn=cmd_backtest)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
