# NFL props model

A log-linear yardage model for NFL passing, rushing, and receiving props. Fits each player an
"ability" and each opponent a "defense allowed" rating on `log(yards + 10)` by weighted ridge
regression (recency-weighted, shrinkage-regularized), giving every player-game a log-normal
yardage distribution. From that distribution: `P(over/under any line)`, edge and expected value
against a supplied price, and ranked parlays.

Standalone Python project, sibling to `epl-parlay/`. Neither touches the Next.js app in the
parent folder or each other.

## Setup

```
cd nfl-props
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
```

Data is downloaded from [nflverse](https://github.com/nflverse/nflverse-data) on first use and
cached in `data/` (finished seasons forever, the current season refreshed every few hours).

## Commands

All commands: `.venv\Scripts\python -m nfl_props <command>`.

| command | what it does |
|---|---|
| `update` | force re-download of stats, schedules, rosters |
| `ratings` | player ability + opponent defense ratings per stat category |
| `predict --week N` | projected yardage distribution for that week's matchups |
| `parlay [--odds file.csv]` | rank legs by edge against live Kalshi prices (no file needed to start), build ranked parlays; `--odds` overrides/supplements the feed, `--no-feed-odds` disables it |
| `backtest` | walk-forward evaluation: log-likelihood and calibration vs. a naive baseline |

### Typical week

```
.venv\Scripts\python -m nfl_props --season 2026 --week 3 predict --top 10
.venv\Scripts\python -m nfl_props --season 2026 --week 3 parlay
```

### Player prop odds: Kalshi feed (default) or your own CSV

`parlay` and `best-bet` price player props against
[Kalshi](https://kalshi.com)'s live, real-money markets for NFL passing/rushing/receiving
yardage by default -- no `--odds` file needed to get started, the same way `game-bets`
already gets a free reference line from `games.csv`. Kalshi's read (market-data) endpoints
need no API key; this project only ever reads prices, it never places an order.

- Prints a one-line match summary before the table, e.g.
  `Kalshi: 174 props matched, 176 skipped (no roster match)` -- an unmatched Kalshi player
  name is skipped, not a hard failure, since the feed pulls in many players automatically.
- `--no-feed-odds` ignores the live feed entirely.
- `--odds file.csv` still works: it overrides any feed leg sharing the same
  player/stat/line and adds anything the feed doesn't have.
- Real-time only -- not wired into `backtest`, which needs historical closing lines.
- Some Kalshi contracts are thin (wide bid/ask spreads on far-out-of-the-money lines);
  treat those edges with extra skepticism, the tool doesn't filter them out.

### Odds file (`--odds`, optional -- overrides/supplements the Kalshi feed)

```
player,stat,line,over_odds,under_odds
Patrick Mahomes,pass_yds,275.5,-115,-105
Christian McCaffrey,rush_yds,84.5,-110,-110
```

- `stat`: `pass_yds`, `rush_yds`, `rec_yds` (aliases like `passing`, `rushing`, `receiving` work too)
- odds: decimal (`1.91`) or American (`+185`, `-110`)
- supply both sides when you can; the tool then removes the bookmaker margin exactly. A
  one-sided line is marked with `*` in `parlay` output and assumes a 5% margin.

## What the backtest says, and how to use this honestly

Walk-forward over the last completed season, refit weekly, the model never sees the game it
predicts. Lower NLL (negative log-likelihood) is better. Run with
`.venv\Scripts\python -m nfl_props backtest --test-seasons 1` on 2026-09-09:

| stat | model NLL | season-to-date-average baseline NLL |
|---|---|---|
| pass_yds | 0.4980 (589 predictions, from 2025-02-08) | 0.5875 |
| rush_yds | 0.7642 (1122 predictions, from 2025-02-08) | 0.8443 |
| rec_yds | 0.9152 (3405 predictions, from 2025-02-08) | 1.0096 |

The model beats the naive baseline on all three stats: about 15% lower NLL on pass_yds, and
roughly 9-10% lower on rush_yds and rec_yds. No NaNs or fit failures in this run.

Note: the baseline is given the model's own fitted `sigma` for this comparison (not an
independently estimated one), so this NLL gap measures whether knowing the opponent
improves the *mean* projection, not whether either distribution's width is well
calibrated — the calibration table below is what actually validates that.

Calibration (PIT buckets, each should hold ~10% of predictions if well-calibrated):

```
=== pass_yds ===
                n  mean_pit
(-0.001, 0.1]  68     0.030
(0.1, 0.2]     49     0.155
(0.2, 0.3]     47     0.247
(0.3, 0.4]     52     0.349
(0.4, 0.5]     49     0.445
(0.5, 0.6]     57     0.551
(0.6, 0.7]     80     0.649
(0.7, 0.8]     71     0.748
(0.8, 0.9]     77     0.852
(0.9, 1.0]     39     0.937

=== rush_yds ===
                 n  mean_pit
(-0.001, 0.1]  128     0.044
(0.1, 0.2]     102     0.151
(0.2, 0.3]     122     0.249
(0.3, 0.4]      97     0.352
(0.4, 0.5]      95     0.453
(0.5, 0.6]     102     0.551
(0.6, 0.7]     114     0.651
(0.7, 0.8]     118     0.755
(0.8, 0.9]     126     0.848
(0.9, 1.0]     118     0.954

=== rec_yds ===
                 n  mean_pit
(-0.001, 0.1]  439     0.042
(0.1, 0.2]     327     0.150
(0.2, 0.3]     277     0.250
(0.3, 0.4]     307     0.349
(0.4, 0.5]     324     0.449
(0.5, 0.6]     344     0.549
(0.6, 0.7]     343     0.649
(0.7, 0.8]     334     0.752
(0.8, 0.9]     360     0.851
(0.9, 1.0]     350     0.946
```

Bucket counts as a share of that stat's total predictions: rush_yds is close to uniform
(8.5%-11.4% per bucket against a 10% target). rec_yds is close too (8.1%-12.9%), with a mild
excess in the bottom decile. pass_yds is the least even: the bottom bucket runs a bit hot
(11.5%) and the top bucket runs cold (39/589 = 6.6%), i.e. the model is slightly
under-confident about the very highest passing-yardage games relative to what actually
happened. None of the three are badly miscalibrated, but pass_yds is worth revisiting with
more seasons of data before leaning on it hard at the tails.

The Kalshi feed (see above) is **live-only, not historical**, so this backtest validates model
*calibration* against actual outcomes and a naive opponent-blind baseline — it does not (and
cannot, without recorded historical prices) prove the model beats Kalshi's closing price the
way `epl-parlay`'s backtest can against Bet365. Treat a reported "edge" in `parlay`
output as *the feed's price minus the model's fair number*, not a proven inefficiency.

## Known limitations

- **No injury/inactive-list awareness.** `predict`/`parlay` filter to roster `status == "ACT"`
  as of the roster snapshot for that week, which is not the same as the gameday-inactive list
  (published ~90 minutes before kickoff). Check inactives yourself before betting.
- **Kalshi is real-time only.** See the backtest section above -- historical closing prices
  aren't available the same way, so the live feed isn't wired into `backtest`.
- **Legs are independent.** Two legs from the same game (e.g. a QB's pass yards and his WR1's
  receiving yards) are priced as if uncorrelated, even though game script correlates them in
  reality. Not modeled in this version.
- **Small per-season sample.** ~17 games/player/season converges slower than a sport with a
  longer season; ridge shrinkage leans harder on the league-average prior early each season,
  and a player with fewer than 4 qualifying games is flagged `low_sample` in `ratings` output.
- **No starter/volume/depth-chart awareness.** Every active-status player is projected
  as if he holds his typical full workload — the tool doesn't know a backup QB will
  barely play behind a healthy starter (e.g. a real run projected backup Joe Flacco at
  252.4 pass yards right alongside starter Joe Burrow's 262.4 in the same game). A
  player's ability rating reflects what he did in HIS OWN past qualifying games, not his
  current depth-chart role. Check depth charts yourself before trusting a projection.

## Model

- `log(yards + 10) ~ intercept + player_ability + opponent_defense + home_field`, one
  independent fit per stat category (`pass_yds`, `rush_yds`, `rec_yds`).
- Weighted ridge regression (closed-form normal equations), games weighted by
  `0.5^(days_ago / halflife_days)`. Player and opponent-defense coefficients are L2-penalized
  (not the intercept or home-field term), which is what shrinks low-sample players toward the
  league average automatically.
- Residual standard deviation (`sigma`) is estimated per position group where at least 30
  qualifying residuals are available, falling back to one global value per stat category
  otherwise.
- Games only count toward a stat's fit if the player met a minimum usage threshold that
  season: 10 attempts (pass_yds), 5 carries (rush_yds), 2 targets (rec_yds) — excludes
  injury-shortened or token appearances from the fit.
- Defaults (`--halflife 180`, `--reg 5.0`) are starting points, not yet grid-searched the way
  epl-parlay's were. The Task 7 backtest run above used these defaults as-is; no grid search was
  performed in this task, so they have not been tuned against the walk-forward NLL. That's a
  reasonable follow-up before relying on the model's tails.

## Layout

```
nfl_props/data.py       download + cache nflverse player stats, schedules, rosters
nfl_props/model.py      ridge fit per stat category, shrinkage, recency weighting
nfl_props/markets.py    log-normal yardage distribution -> P(over/under line), devig, odds formats
nfl_props/parlay.py     leg evaluation (edge, EV, Kelly), parlay enumeration
nfl_props/backtest.py   walk-forward evaluation, calibration
nfl_props/cli.py        commands
tests/                  pytest
```

## Game outcomes (moneyline / spread / totals)

A team-level companion to the player-props model above: a margin model (single power
rating per team) for moneyline and spread, and a score model (scoring-rate + allowed-rate
per team) for totals. Both are weighted ridge regressions using the same fitting mechanics
as the player-yardage model, just at team level.

**Kept separate from the player-props `parlay` command in this version** — a game leg and
a player-prop leg can't be combined into one parlay ticket yet.

### Commands

| command | what it does |
|---|---|
| `game-ratings` | team power ratings (margin) + scoring/allowed ratings (totals) |
| `game-predict --week N` | moneyline/spread/total fair probabilities for that week's games |
| `game-bets --week N [--odds file.csv]` | edge vs. the real reference line built into `games.csv`, no CSV required to start; `--odds` overrides with your own prices |
| `game-backtest` | walk-forward evaluation vs. a naive baseline (home-field-only for margin, league-average for totals) |
| `best-bet --week N [--odds file.csv]` | the single highest-edge pick in each market: one game bet (from the real reference line) and, if `--odds` is given, one player prop |

### The reference line, and its sign convention

`games.csv` (from nflverse) already carries a real `spread_line`/`total_line`/moneyline
for every game. **This is not attributed to a specific sportsbook in nflverse's own data
dictionary** — treat it as a reference/consensus price, not a proven-beatable line the way
`epl-parlay`'s labeled Bet365 feed is.

**`spread_line`'s sign convention is the opposite of common bettor intuition**: a
*positive* number means the home team is favored by that many points (a negative number
means the away team is favored). This tool's `spread` market follows that same convention
throughout — the home team covers iff `home_score - away_score > spread_line`.

### Odds file (for `game-bets --odds`)

```
home_team,away_team,market,selection,line,odds
SEA,NE,moneyline,home,,+150
SEA,NE,moneyline,away,,-180
SEA,NE,spread,home,3.0,-110
SEA,NE,spread,away,3.0,-110
SEA,NE,total,over,44.5,-110
SEA,NE,total,under,44.5,-110
```

- `market`: `moneyline` (or `ml`), `spread` (or `ats`), `total` (or `totals`/`ou`/`o/u`)
- `selection`: `home`/`away` for moneyline and spread, `over`/`under` for total
- `line`: blank for moneyline; the spread line (home-perspective, same sign convention as
  above) for spread; the total line for total
- odds: decimal or American, same parsing as the player-props odds file

### What the game backtest says

Walk-forward over the last completed season, refit weekly, the model never sees the game
it predicts. Run with `.venv\Scripts\python -m nfl_props game-backtest --test-seasons 1`
on 2026-09-10:

| market | model NLL | baseline NLL | baseline |
|---|---|---|---|
| margin | 4.0209 (270 predictions, from 2025-09-09) | 4.0980 | home-field-only |
| totals | 4.0093 (270 predictions, from 2025-09-09) | 4.0582 | league-average |

The model beats the naive baseline on both markets, but by a much smaller margin than the
player-props model above: about 1.9% lower NLL on margin, and about 1.2% lower NLL on
totals. No NaNs or fit failures in this run.

Calibration (PIT buckets, each should hold ~10% of predictions if well-calibrated):

```
=== margin: 270 predictions from 2025-09-09 ===
                n  mean_pit
(-0.001, 0.1]  33     0.054
(0.1, 0.2]     24     0.150
(0.2, 0.3]     32     0.259
(0.3, 0.4]     28     0.345
(0.4, 0.5]     22     0.456
(0.5, 0.6]     31     0.538
(0.6, 0.7]     23     0.660
(0.7, 0.8]     24     0.751
(0.8, 0.9]     19     0.841
(0.9, 1.0]     34     0.955

=== totals: 270 predictions from 2025-09-09 ===
                n  mean_pit
(-0.001, 0.1]  25     0.050
(0.1, 0.2]     29     0.141
(0.2, 0.3]     24     0.261
(0.3, 0.4]     29     0.356
(0.4, 0.5]     25     0.463
(0.5, 0.6]     30     0.550
(0.6, 0.7]     24     0.645
(0.7, 0.8]     25     0.751
(0.8, 0.9]     27     0.856
(0.9, 1.0]     32     0.952
```

Bucket counts as a share of the 270 predictions: totals is close to uniform (8.9%-11.9% per
bucket against a 10% target: 25, 29, 24, 29, 25, 30, 24, 25, 27, 32). margin is less even —
the bottom bucket runs a bit hot (33/270 = 12.2%) and the top bucket does too (34/270 =
12.6%), while the 0.8-0.9 bucket runs cold (19/270 = 7.0%), suggesting the margin model's
predictive distribution is a little too narrow in the tails on this one-season sample.
Neither market is badly miscalibrated, but with only 270 predictions from a single season,
both the NLL gap and the calibration read here should be treated as a first look rather
than a settled result — worth re-checking once more seasons of walk-forward data are
available, and before leaning on the margin model's tails.

Same honesty caveat as the player-props backtest: the reference line has no named
provider, so this validates model *calibration*, not proven edge over a real sportsbook's
closing line.

Note: the baseline is scored using the model's own fitted sigma (not an independently
estimated one), so the reported NLL gap overstates the model's advantage somewhat — it
measures whether knowing team identity improves the *mean* projection, not a clean
apples-to-apples comparison of two independently-tuned distributions.

### Known limitations (game outcomes)

- **Odds source has no named provider.** Treat `games.csv`'s lines as reference/consensus,
  not necessarily beatable.
- **No injury/lineup awareness**, same as the player-props model.
- **Single power rating for margin** means the model can't separately say "great offense,
  bad defense" — only the net effect on margin is identifiable from margin data alone.
- **No starter/depth-chart awareness** carries over conceptually here too: a team missing
  its starting QB isn't reflected in its power rating until enough post-injury games
  accumulate to shift the recency-weighted fit.
- **Pushes aren't modeled.** Both markets use a continuous probability distribution, so
  an exact push (final margin/total lands precisely on the line) is assigned zero
  probability, even though real lines are frequently exact integers (~55.6% of
  spread lines) and real push rates on common numbers like a 3-point margin are
  non-trivial (~8.1% of games finish with an exact 3-point margin; ~4.8% with an exact
  7-point margin). This modestly inflates the favored side's computed probability on
  integer lines — treat edges on exact-integer lines with extra skepticism.
