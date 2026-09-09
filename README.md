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
| `parlay --odds file.csv` | rank legs by edge against your prices, build ranked parlays |
| `backtest` | walk-forward evaluation: log-likelihood and calibration vs. a naive baseline |

### Typical week

```
.venv\Scripts\python -m nfl_props --season 2026 --week 3 predict --top 10
.venv\Scripts\python -m nfl_props --season 2026 --week 3 parlay --odds my_odds.csv
```

### Odds file

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

There is **no free historical player-prop odds feed**, so this backtest validates model
*calibration* against actual outcomes and a naive opponent-blind baseline — it does not (and
cannot, without your own historical prices) prove the model beats a real sportsbook's closing
line the way `epl-parlay`'s backtest can against Bet365. Treat a reported "edge" in `parlay`
output as *your price minus the model's fair number*, not a proven inefficiency.

## Known limitations

- **No injury/inactive-list awareness.** `predict`/`parlay` filter to roster `status == "ACT"`
  as of the roster snapshot for that week, which is not the same as the gameday-inactive list
  (published ~90 minutes before kickoff). Check inactives yourself before betting.
- **No historical prop-line data.** See the backtest section above.
- **Legs are independent.** Two legs from the same game (e.g. a QB's pass yards and his WR1's
  receiving yards) are priced as if uncorrelated, even though game script correlates them in
  reality. Not modeled in this version.
- **Small per-season sample.** ~17 games/player/season converges slower than a sport with a
  longer season; ridge shrinkage leans harder on the league-average prior early each season,
  and a player with fewer than 4 qualifying games is flagged `low_sample` in `ratings` output.

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
