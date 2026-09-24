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

**Commit:** `47729de` (master, 2026-09-22). **Data window:** 3 seasons of real nflverse
data, 1-year walk-forward lookback anchored to the most recent date in the loaded stats
at run time, run 2026-09-22. **Method:** `rate_backtest.walk_forward()` (not
`backtest.walk_forward()`, which the rest of this file uses) — true walk-forward, refit
every `(season, week)` using only data strictly before that date, scored against a naive
season-to-date-average baseline. Model: `rate_model.fit_poisson()`'s existing quasi-Poisson
IRLS fit (player ability + opponent defense + home field + trailing target-share
covariate), wired to the `targets` stat for the first time in this plan (see
`docs/superpowers/plans/2026-09-22-nfl-props-opportunity-model-targets.md`). **Inert:** no
CLI command (`predict`/`parlay`/`best-bet`/`fantasy-predict`) reads this model yet — this
is a research checkpoint, not a shipping decision, matching how this file already treats
`joint_sim.py` above.

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

### snap_share ablation

**Commit:** `3386819` (master, run 2026-09-22) -- the commit that added
`data.stats_with_trailing_snap_share()`; the result below is not reproducible at any
earlier commit. **Method:** same `rate_backtest.walk_forward()` setup as the `targets`
section above, with one additional step: `stats = data.stats_with_trailing_snap_share(stats,
seasons=3)` before calling `walk_forward`, comparing with vs. without `trailing_snap_share`
as an `extra_covariates` column, per
`docs/superpowers/plans/2026-09-22-nfl-props-opportunity-model-snap-share.md`. Join
coverage: only 20/13,334 relevant rows (3/7,123 qualifying rows) had no snap-count/
crosswalk match and fell back to `offense_pct=0.0` -- not a material share of the input.

| Run | n | NLL model | NLL baseline |
|---|---|---|---|
| Base (trailing target-share only) | 6387 | 1.699028790877028 | 1.841225562020044 |
| + snap_share | 6387 | 1.6892137371486382 | 1.901954381214095 |

Note: `NLL baseline` is **not** a fixed reference across the two rows -- `rate_backtest`'s
naive baseline is derived from the same fitted position-group intercepts the model uses,
which shift when an uncentered covariate (`trailing_snap_share`, mean ~0.51) is added.
Only the model-vs-model `NLL model` comparison (1.699029 vs. 1.689214) is a valid read
here; the baseline column moving is an artifact of the covariate addition, not evidence
the edge over baseline widened.

**Read honestly:** Adding `trailing_snap_share` as an extra covariate improved NLL model
from 1.699029 to 1.689214 -- a real but small gain of ~0.00982 nats (~0.6% relative
improvement). That is a much smaller effect than the usage-share covariate's validated
gains on `rec_yds`/`rush_yds` (3.3%/10.6%, see README's "Usage-share covariate" section)
and is in the same rough magnitude as noise from one backtest run (single walk-forward
window, no significance test performed here). It is a real, directionally positive
result, not a failure like `--pace-adjust`'s -- but on its own it is too marginal to
justify making `trailing_snap_share` a default covariate for the `targets` model without
further validation (e.g. across a longer window, or with a significance check) first.
This ablation result alone does not ship anything live -- no CLI command reads either of
these covariate configurations yet.

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
from nfl_props import data, rate_backtest

stats = data.load_player_stats(seasons=3)
stats = data.stats_with_trailing_snap_share(stats, seasons=3)
start = (stats['date'].max() - timedelta(days=365)).date()

base = rate_backtest.walk_forward(stats, 'targets', start, halflife_days=180.0, reg=5.0, min_games=200)
print('Base:', rate_backtest.summarize(base))

with_snap = rate_backtest.walk_forward(stats, 'targets', start, halflife_days=180.0, reg=5.0, min_games=200,
                                       extra_covariates=['trailing_snap_share'])
print('+ snap_share:', rate_backtest.summarize(with_snap))
"
```

## team pass volume (TeamPassVolume, first validation)

**Commit:** `4d617de` (master, run 2026-09-23). **Method:**
`game_backtest.walk_forward_pass_volume()` — true walk-forward, refit every `(season,
week)` using only data strictly before that date, scored against a naive league-
expanding-average-pass_oe-so-far baseline (ignores team identity). Model:
`game_model.fit_pass_volume()` (built in
docs/superpowers/plans/2026-09-22-nfl-props-opportunity-model-targets.md, previously only
synthetic-recovery-tested) — one weighted-ridge rating per team on play-by-play-derived
pass-rate-over-expected, mirroring `fit_margin`'s structure.

| Metric | Value |
|---|---|
| NLL (model / baseline) | 3.2332461232319463 / 3.2579245739983143 |
| n (team-games) | 568 |

Calibration (PIT buckets, each should hold ~10% of predictions if well-calibrated):

```
                n  mean_pit
(-0.001, 0.1]  66  0.057429
(0.1, 0.2]     67  0.149551
(0.2, 0.3]     54  0.249168
(0.3, 0.4]     50  0.348307
(0.4, 0.5]     54  0.447089
(0.5, 0.6]     59  0.553336
(0.6, 0.7]     43  0.649720
(0.7, 0.8]     47  0.754352
(0.8, 0.9]     56  0.846659
(0.9, 1.0]     72  0.950343
```

**Read honestly:** The model beats the naive league-average baseline by a small margin
(0.0247 nats, ~0.76% relative NLL improvement) — a real but weak result. For honest
context: every other model-vs-naive-baseline relative NLL gain recorded in this file is
substantially larger (`pass_yds` 7.1%, `rush_yds` 25.9%, `rec_yds` 6.5%, `targets` 7.7%),
so 0.76% is the weakest result in this file by that comparison, not a comparable one to
any single-covariate ablation (which measures a different thing: model-vs-model, not
model-vs-baseline). Because `base_sigma` equals `model_sigma` for every row in this
backtest, the NLL gap is mathematically a rescaled mean-squared-error reduction — computed
directly: MSE model = 37.488805256378086, MSE baseline = 39.183462183288206, a real
4.3% reduction in squared error versus the naive baseline. This
validation establishes that `fit_pass_volume()` extracts real, non-zero signal from
play-by-play pass-rate data in a walk-forward setting, but the effect is weak by this
project's own established standards and does not on its own justify using it as a
covariate. **Inert:** no CLI command reads `fit_pass_volume` -- `game-backtest` still runs
margin and totals only. This result alone does not ship anything live and does not decide
whether `TeamPassVolume` becomes a covariate for the `targets` model -- that is a
separate, later decision requiring its own real backtest, per
`docs/superpowers/plans/2026-09-22-nfl-props-team-pass-volume-validation.md`.

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
from nfl_props import data, game_backtest

games = data.load_games()
pbp = data.load_pbp(seasons=3)
team_pass_rate = data.aggregate_pbp_team_pass_rate(pbp)
dated = team_pass_rate.merge(games[['game_id', 'gameday']], on='game_id')
start = (dated['gameday'].max() - timedelta(days=365)).date()

preds = game_backtest.walk_forward_pass_volume(games, team_pass_rate, start,
                                               halflife_days=365.0, reg=3.0, min_games=100)
print(game_backtest.summarize(preds, 'actual_pass_oe'))
print(game_backtest.calibration(preds, 'actual_pass_oe'))
"
```

## CatchRate | Targets (first validation)

**Commit:** `96ca34b` (master). **Data window:** 3 seasons of real nflverse data, 1-year
walk-forward lookback anchored to the most recent date in `data.load_targets()`'s output
at run time, run 2026-09-23. **Method:** `catch_backtest.walk_forward()` -- true
walk-forward, refit every `(season, week)` using only data strictly before that date,
scored against a per-player season-to-date completion-rate baseline (blended
with the fitted model's own position-group base rate for players with no prior targets
that season). Model: `catch_model.fit_catch_rate()`'s new logistic IRLS fit (player
ability + opponent defense + home field + per-target aDOT, i.e. `air_yards`), built in
`docs/superpowers/plans/2026-09-23-nfl-props-catch-rate-model.md`. **Inert:** no CLI
command (`predict`/`parlay`/`best-bet`/`fantasy-predict`) reads this model -- this is a
research checkpoint, not a shipping decision, matching how this file already treats
`targets` and `team pass volume` above.

| Metric | Value |
|---|---|
| NLL (model / baseline) | 0.5864556345856085 / 0.6539519937394961 |
| n | 17257 |

Calibration (predicted-probability bin, equal-width at 0.1 each, vs. actual catch rate in that bin):

```
                  n  mean_predicted  actual_rate
(-0.001, 0.1]    20        0.083763     0.150000
(0.1, 0.2]      227        0.164089     0.264317
(0.2, 0.3]      342        0.249610     0.307018
(0.3, 0.4]      510        0.355746     0.352941
(0.4, 0.5]      949        0.457194     0.429926
(0.5, 0.6]     1875        0.555717     0.537067
(0.6, 0.7]     3471        0.656518     0.631518
(0.7, 0.8]     6465        0.752066     0.756226
(0.8, 0.9]     3379        0.833125     0.815626
(0.9, 1.0]       19        0.909892     0.684211
```

**Read honestly:** The model beats the naive per-player season-to-date baseline by
0.0675 nats (0.6539519937394961 - 0.5864556345856085 = 0.06749635915388763), a
~10.32% relative NLL improvement (`(0.6539519937394961 - 0.5864556345856085) /
0.6539519937394961 = 0.10321301838675183`). This is a real win, and by this file's own
comparison standard it is the strongest first-validation result recorded here so far:
larger than `targets`' ~7.7% and much larger than `team pass volume`'s ~0.76% -- all
three are model-vs-baseline NLL comparisons (not covariate-ablation percentages like
`snap_share`'s ~0.6%, which measures a different thing, model-vs-model), so this
comparison is apples-to-apples with those two, not with the ablation figure. Calibration
is good but not uniform: the two largest bins by far, `(0.7, 0.8]` (n=6465, ~37% of all
predictions) and `(0.8, 0.9]` (n=3379, ~20%), track closely -- 0.752066 predicted vs.
0.756226 actual, and 0.833125 predicted vs. 0.815626 actual, respectively -- and
`(0.3, 0.4]` (n=510) is also tight (0.355746 vs. 0.352941). But the low-probability bins
show a real, sample-size-backed divergence: `(0.1, 0.2]` (n=227) predicts 0.164089 but
actually catches at 0.264317 (+0.10), and `(0.2, 0.3]` (n=342) predicts 0.249610 against
an actual 0.307018 (+0.057) -- both bins have enough rows that this looks like systematic
underprediction in the low-catch-probability range, not noise. The largest single
divergence is `(0.9, 1.0]` (n=19, actual 0.684211 vs. predicted 0.909892, -0.226), but
with only 19 rows out of 17257 it is too small to read as a calibration problem rather
than noise. `(-0.001, 0.1]` (n=20, actual 0.150000 vs. predicted 0.083763) is similarly
too small a sample to be meaningful on its own. Net: the model wins clearly on NLL, and
calibration is solid in the bulk of the probability mass (0.7-0.9, ~57% of all
predictions) with a real low-probability underprediction bias worth investigating in a
future pass, not a tail artifact to ignore.

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
from nfl_props import data, catch_backtest

games = data.load_games()
pbp = data.load_pbp(seasons=3)
stats = data.load_player_stats(seasons=3)
targets = data.load_targets(pbp, stats, games)
start = (targets['date'].max() - timedelta(days=365)).date()

preds = catch_backtest.walk_forward(targets, start, halflife_days=180.0, reg=5.0, min_targets=200)
print(catch_backtest.summarize(preds))
print(catch_backtest.calibration(preds))
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
