# Baseline metrics — current model snapshot

**Commit:** `ec2aaf9` (master, 2026-09-18) — includes the sigma-contamination fix,
now live by default in `predict`/`parlay`/`best-bet`. The same-game joint-correlation
Monte Carlo module (`joint_sim.py`) is also merged but is inert (unimported by any live
command — see `docs/superpowers/specs/2026-09-17-nfl-props-joint-correlation-sim-design.md`),
so it does not affect any number below.
**Data window:** 1-year walk-forward lookback, 3 seasons of real nflverse data, run 2026-09-18
**Method:** `backtest.walk_forward()` — true walk-forward, refit every `(season, week)` using
only data strictly before that date, scored against a naive season-to-date-average baseline.
`pass_yds` additionally uses `wide_sigma=True` (the low-sample-aware `sigma_normal_only`
split), matching what `predict`/`parlay`/`best-bet` now always do live for that stat —
`rush_yds`/`rec_yds` are unaffected (`model.WIDE_SIGMA_STATS = {"pass_yds"}` only).

This file exists so future model changes can be compared against a fixed, real,
reproducible reference point rather than a shifting one — see `docs/superpowers/specs/`
in the parent repo for the design history behind every constant referenced below.

## pass_yds (n=656, wide_sigma=True)

| Metric | Value |
|---|---|
| NLL (model / baseline) | 1.2330 / 1.3274 |
| MAE | 77.78 yds |
| RMSE | 96.70 yds |
| Brier (own median line, own p_over) | 0.2497 |
| Calibration: bottom decile / top decile | 12.0% / 3.7% (target ~10% each) |

**History:** the original snapshot of this stat (master `5f91564`, before the fix)
measured NLL 1.3020/1.3378, top-decile calibration 2.9% (n=693) — see
`docs/superpowers/specs/2026-09-16-nfl-props-low-sample-sigma-design.md` for the full
design history. The `nfl-props-sigma-contamination-fix` branch (a bug in the original
`WIDE_SIGMA_STATS` implementation's sigma-pool contamination) is now merged and live:
top-decile calibration moved from 2.9%→3.7% toward the 10% target, and NLL improved
1.3020→1.2330. Top-decile calibration remains below target — this stat's low-sample-QB
tail is measurably improved, not fully solved.

## rush_yds (n=6066)

| Metric | Value |
|---|---|
| NLL (model / baseline) | 0.1124 / 0.1517 |
| MAE | 6.71 yds |
| RMSE | 16.11 yds |
| Brier (own median line, own p_over) | 0.2818 |
| Calibration: bottom decile / top decile | 4.8% / 7.5% (target ~10% each) |

## rec_yds (n=6066)

| Metric | Value |
|---|---|
| NLL (model / baseline) | 0.8555 / 0.9145 |
| MAE | 13.41 yds |
| RMSE | 22.18 yds |
| Brier (own median line, own p_over) | 0.2587 |
| Calibration: bottom decile / top decile | 9.3% / 11.1% (target ~10% each) |

(`rush_yds`/`rec_yds` numbers are essentially unchanged from the original snapshot — the
small shifts are the 1-year rolling window advancing by a day, not a model change.)

## Brier score methodology caveat

No historical Kalshi (or any) price archive exists for this project, so there is no real
market line to score a Brier score against. The number above is constructed instead
against the model's **own** predicted median, rounded to the nearest 5 yards (matching
`predict`'s own `--line-step` convention) — this is a real proper-scoring-rule check of
the model's own stated `p_over`, but it is an artificial construction, not a market-edge
metric. Treat these Brier numbers as a **relative** benchmark for comparing future model
versions against this exact same methodology, not as an absolute judgment of real
betting quality — the NLL and calibration numbers above, and the real tracked
`best-bet --log` / `grade` record (see README), are the more trustworthy signals for that.

## targets (opportunity model, first validation)

**Method:** `rate_backtest.walk_forward()` — true walk-forward, refit every `(season,
week)` using only data strictly before that date, scored against a naive season-to-date-
average baseline. Model: `rate_model.fit_poisson()`'s existing quasi-Poisson IRLS fit
(player ability + opponent defense + home field + trailing target-share covariate),
wired to the `targets` stat for the first time in this plan (see
`docs/superpowers/plans/2026-09-22-nfl-props-opportunity-model-targets.md`).

| Metric | Value |
|---|---|
| NLL (model / baseline) | 1.699028790877028 / 1.841225562020044 |
| n | 6387 |

Calibration (PIT buckets, each should hold ~10% of predictions if well-calibrated):

```
                 n  mean_pit
(-0.001, 0.1]  746  0.050632
(0.1, 0.2]     711  0.150296
(0.2, 0.3]     692  0.249419
(0.3, 0.4]     681  0.348691
(0.4, 0.5]     611  0.448234
(0.5, 0.6]     628  0.547358
(0.6, 0.7]     547  0.651389
(0.7, 0.8]     552  0.750083
(0.8, 0.9]     547  0.850367
(0.9, 1.0]     672  0.954086
```

**Read honestly:** The targets model beats the naive season-to-date baseline by 0.14 NLL
(~7.7% improvement), with calibration buckets well-distributed around the target 10%—
this validates the quasi-Poisson fit for targets as a solid foundation for future
ablations (snap_share, air_yards_share, team_pass_volume composition).

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
from nfl_props import data, rate_backtest
stats = data.load_player_stats(seasons=3)
start = (stats['date'].max() - timedelta(days=365)).date()
df = rate_backtest.walk_forward(stats, 'targets', start, halflife_days=180.0, reg=5.0, min_games=200)
print(rate_backtest.summarize(df))
print(rate_backtest.calibration(df))
"
```

## Reproducing this baseline

```bash
cd nfl-props
.venv\Scripts\python -m nfl_props backtest --test-seasons 1 --wide-sigma
```

(`--wide-sigma` only affects `pass_yds`, matching what `predict`/`parlay`/`best-bet` now
do by default for that stat. The MAE/RMSE/Brier figures above required a small one-off
script beyond what `backtest` prints by default; `backtest.walk_forward()` +
`backtest.summarize()` + `backtest.calibration()` are the only building blocks used — no
code changes.)
