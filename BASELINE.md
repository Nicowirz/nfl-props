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

### position-split opponent-defense ablation

**Commit:** `7353ff8` (master). **Method:** same `catch_backtest.walk_forward()` setup as
the base `CatchRate | Targets` section above, with one additional step:
`position_split_defense=True`, per
`docs/superpowers/plans/2026-09-23-nfl-props-catch-rate-position-split-defense.md` -- the
spec's own next Stage 1 ablation step (`## Ablation plan`). Base is re-run under the same
commit/data window for a same-run, apples-to-apples comparison rather than reused from
the prior section's recorded number.

| Run | n | NLL model | NLL baseline |
|---|---|---|---|
| Base (single opponent scalar) | `17257` | `0.5864556345856085` | `0.6539519937394961` |
| + position_split_defense | `17257` | `0.5874359273471375` | `0.6539746941168135` |

Note: `NLL baseline` is **not** necessarily comparable across the two rows -- as the
`snap_share ablation` subsection above already establishes for the identical reason, the
baseline is derived from the model's own fitted position-group intercepts, which can
shift between configurations. Only the model-vs-model `NLL model` comparison is a valid
read here.

Calibration under `+ position_split_defense` (predicted-probability bin, equal-width at
0.1 each, vs. actual catch rate in that bin):

```
                  n  mean_predicted  actual_rate
(-0.001, 0.1]    21        0.083708     0.142857
(0.1, 0.2]      223        0.163567     0.282511
(0.2, 0.3]      341        0.248359     0.299120
(0.3, 0.4]      516        0.354765     0.344961
(0.4, 0.5]      962        0.457086     0.432432
(0.5, 0.6]     1871        0.555871     0.541956
(0.6, 0.7]     3498        0.656895     0.628931
(0.7, 0.8]     6308        0.751021     0.757134
(0.8, 0.9]     3477        0.833784     0.814783
(0.9, 1.0]       40        0.908739     0.700000
```

**Read honestly:** Adding `position_split_defense` made NLL model *worse*, not better --
it moved from 0.5864556345856085 (Base) to 0.5874359273471375 (+ position_split_defense),
a change of +0.0009802927615290002 nats. Using the required formula, `(nll_model_base -
nll_model_split) / nll_model_base = (0.5864556345856085 - 0.5874359273471375) /
0.5864556345856085 = -0.0016715548520932508`, i.e. a **-0.167% relative change** -- a
real regression, not an improvement. In magnitude this is smaller than `snap_share`'s
~0.6% ablation gain (the only other model-vs-model ablation percentage recorded in this
file), and unlike `snap_share` it is negative: the extra per-(team, position_group)
defense coefficients did not extract useful signal here and instead cost a small amount
of fit quality, consistent with the dimensionality increase this ablation introduces
(far more coefficients than the single-scalar defense term, fit on the same walk-forward
data). Note also that this configuration REPLACES the pooled team-level defense term
entirely (there is no team-level defense term left when `position_split_defense=True` --
each `(team, position)` cell is shrunk toward 0 independently, not toward a team mean),
not merely augments it with a position-specific adjustment; a pooled-plus-interaction
variant was not tested here and remains an open question. This result is not strong
enough, on its own, to justify making `position_split_defense=True` the default for
`CatchRate` -- it is a loss, not a marginal win, so per this project's gate discipline it
does not ship. The calibration table also shows a small corroborating signal: the
`(0.1, 0.2]` bin's underprediction gap widens from +0.100 in the base run (predicted
0.164089 vs. actual 0.264317) to +0.119 here (predicted 0.163567 vs. actual 0.282511),
consistent with the split configuration's NLL loss rather than contradicting it.
**Inert:** no CLI command reads either configuration of this model -- this ablation
result alone does not change any live behavior.

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

base = catch_backtest.walk_forward(targets, start, halflife_days=180.0, reg=5.0, min_targets=200)
print('Base:', catch_backtest.summarize(base))

split = catch_backtest.walk_forward(targets, start, halflife_days=180.0, reg=5.0, min_targets=200,
                                    position_split_defense=True)
print('+ position_split_defense:', catch_backtest.summarize(split))
print(catch_backtest.calibration(split))
"
```

## Yards | Reception (first validation)

**Commit:** `4db3b1e` (master). **Data window:** 3 seasons of real nflverse data, 1-year
walk-forward lookback anchored to the most recent date among COMPLETED targets
(`data.load_targets()`'s output filtered to `complete == 1`) at run time, run 2026-09-23.
**Method:** `yards_backtest.walk_forward()` -- true walk-forward, refit every `(season,
week)` using only data strictly before that date, scored against a per-player
season-to-date-average log-yards baseline (falling back to the fitted model's own
position-group base rate for a player with no prior catches that season). Model:
`yards_model.fit_yards_per_catch()`'s new log-normal ridge fit (player ability + opponent
defense + home field), one row per COMPLETED catch rather than per player-game, built in
`docs/superpowers/plans/2026-09-23-nfl-props-yards-per-reception-model.md`. This predicts
a DIFFERENT quantity than the `rec_yds` section above (yards on ONE catch, not total game
yards) -- the two NLL numbers are not comparable to each other. **Inert:** no CLI command
(`predict`/`parlay`/`best-bet`/`fantasy-predict`) reads this model -- this is a research
checkpoint, not a shipping decision.

| Metric | Value |
|---|---|
| NLL (model / baseline) | 0.4988037610419862 / 0.5528300974432662 |
| n | 11613 |

Calibration (PIT buckets, each should hold ~10% of predictions if well-calibrated):

```
                  n  mean_pit
(-0.001, 0.1]   840  0.048694
(0.1, 0.2]     1135  0.151657
(0.2, 0.3]     1483  0.250104
(0.3, 0.4]     1393  0.348460
(0.4, 0.5]     1345  0.448341
(0.5, 0.6]     1199  0.549150
(0.6, 0.7]     1020  0.648751
(0.7, 0.8]     1020  0.749349
(0.8, 0.9]      928  0.847431
(0.9, 1.0]     1250  0.958758
```

**Read honestly:** The model beats the naive per-player season-to-date-average baseline
by 0.05402633640127996 nats (0.5528300974432662 - 0.4988037610419862 =
0.05402633640127996), a ~9.77% relative NLL improvement ((0.5528300974432662 -
0.4988037610419862) / 0.5528300974432662 = 0.09772683623981666). This is a real win, and
by this file's own comparison standard it lands as the second-strongest first-validation
model-vs-baseline result recorded here: below `CatchRate`'s ~10.32% but above `targets`'
~7.7% and far above `team pass volume`'s ~0.76% -- all model-vs-baseline NLL comparisons
(not covariate-ablation percentages like `snap_share`'s ~0.6% or the position-split
ablation's -0.167%, which measure a different thing, model-vs-model). Calibration is not
uniform: the bottom decile, `(-0.001, 0.1]` (n=840 of 11613, 7.23% of all predictions vs.
the ~10% target), is meaningfully under-filled -- fewer catches land in the model's
lowest-PIT bucket than a well-calibrated model would produce. The top decile, `(0.9,
1.0]` (n=1250, 10.76% of all predictions), tracks the ~10% target closely. The middle of
the table shows real but modest drift in both directions: `(0.2, 0.3]` (n=1483, 12.77%)
and `(0.3, 0.4]` (n=1393, 12.00%) both run above the 10% target, while `(0.6, 0.7]` and
`(0.7, 0.8]` (n=1020 each, 8.78%) and `(0.8, 0.9]` (n=928, 7.99%) all run below it. Net:
the model wins clearly on NLL, by a margin in line with this file's other strong
first-validation results, but calibration diverges from uniform in the bottom decile and
across the 0.2-0.9 middle buckets -- a real, sample-size-backed pattern worth
investigating in a future pass, not evidence the model is broken given the clear NLL win.

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
from nfl_props import data, yards_backtest

games = data.load_games()
pbp = data.load_pbp(seasons=3)
stats = data.load_player_stats(seasons=3)
targets = data.load_targets(pbp, stats, games)
catches = targets[targets['complete'] == 1.0].copy()
start = (catches['date'].max() - timedelta(days=365)).date()

preds = yards_backtest.walk_forward(catches, start, halflife_days=180.0, reg=5.0, min_catches=200)
print(yards_backtest.summarize(preds))
print(yards_backtest.calibration(preds))
"
```

### position-split opponent-defense ablation

**Commit:** `4c40b9d` (master). **Method:** same `yards_backtest.walk_forward()` setup as
the base `Yards | Reception` section above, with one additional step:
`position_split_defense=True`, per
`docs/superpowers/plans/2026-09-24-nfl-props-yards-position-split-defense.md` -- the
spec's own position-split yards-allowed-per-catch term. Base is re-run under the same
commit/data window for a same-run, apples-to-apples comparison rather than reused from
the prior section's recorded number.

| Run | n | NLL model | NLL baseline |
|---|---|---|---|
| Base (single opponent scalar) | `11613` | `0.4988037610419862` | `0.5528300974432662` |
| + position_split_defense | `11613` | `0.5009750302937114` | `0.5534564466519641` |

Note: `NLL baseline` is **not** necessarily comparable across the two rows -- as both the
`snap_share ablation` and `CatchRate` position-split ablation subsections above already
establish for the identical reason, the baseline is derived from the model's own fitted
position-group intercepts, which can shift between configurations. Only the model-vs-model
`NLL model` comparison is a valid read here.

Calibration under `+ position_split_defense` (PIT buckets, each should hold ~10% of
predictions if well-calibrated):

```
                  n  mean_pit
(-0.001, 0.1]   848  0.048375
(0.1, 0.2]     1165  0.151432
(0.2, 0.3]     1466  0.251518
(0.3, 0.4]     1388  0.348681
(0.4, 0.5]     1324  0.448626
(0.5, 0.6]     1185  0.548111
(0.6, 0.7]     1063  0.649905
(0.7, 0.8]      996  0.749783
(0.8, 0.9]      926  0.848765
(0.9, 1.0]     1252  0.959133
```

**Read honestly:** Adding `position_split_defense` made NLL model *worse*, not better --
it moved from 0.4988037610419862 (Base) to 0.5009750302937114 (+
position_split_defense), a change of +0.002171269251725172 nats. Using the required
formula, `(nll_model_base - nll_model_split) / nll_model_base = (0.4988037610419862 -
0.5009750302937114) / 0.4988037610419862 = -0.004352952847006316`, i.e. a **-0.435%
relative change** -- a real regression, not an improvement. In magnitude this loss is
larger than `CatchRate`'s own position-split ablation loss (-0.167%) and runs the
opposite direction from the `snap_share` ablation's gain (+~0.6%) -- the only other
model-vs-model ablation percentages recorded in this file -- not to be confused with this
section's own ~9.77% model-vs-baseline figure above, which measures a different thing.
As with `CatchRate`, this configuration REPLACES the pooled team-level defense term
entirely (there is no team-level defense term left when `position_split_defense=True` --
each `(team, position_group)` cell is shrunk toward 0 independently, not toward a team
mean), not merely augments it with a position-specific adjustment; a pooled-plus-
interaction variant was not tested here and remains an open question. This result is not
strong enough, on its own, to justify making `position_split_defense=True` the default
for `Yards | Reception` -- it is a loss, not a marginal win, so per this project's gate
discipline it does not ship. **Inert:** no CLI command reads either configuration of this
model -- this ablation result alone does not change any live behavior.

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
from nfl_props import data, yards_backtest

games = data.load_games()
pbp = data.load_pbp(seasons=3)
stats = data.load_player_stats(seasons=3)
targets = data.load_targets(pbp, stats, games)
catches = targets[targets['complete'] == 1.0].copy()
start = (catches['date'].max() - timedelta(days=365)).date()

base = yards_backtest.walk_forward(catches, start, halflife_days=180.0, reg=5.0, min_catches=200)
print('Base:', yards_backtest.summarize(base))

split = yards_backtest.walk_forward(catches, start, halflife_days=180.0, reg=5.0, min_catches=200,
                                    position_split_defense=True)
print('+ position_split_defense:', yards_backtest.summarize(split))
print(yards_backtest.calibration(split))
"
```

## rec_yds point-estimate composition (first validation)

**Commit:** `4c995a6` (master). **Data window:** 3 seasons of real nflverse data, 1-year
walk-forward lookback, run 2026-09-24. **Method:** a one-off script (not a reusable backtest module --
matching this file's own "Reproducing this baseline" precedent for MAE/RMSE-style
metrics) refitting `rate_model`/`catch_model`/`yards_model` weekly and composing
`composition.predicted_rec_yds_point_estimate()` for every real WR/TE/RB/QB player-game
in the window, scored by MAE/RMSE against real actual `receiving_yards` -- compared, in
the SAME script and window, against `model.py`'s existing direct `rec_yds` fit (its
median prediction, `exp(model_mu) - OFFSET`, via `backtest.walk_forward()`), per the
spec's own "Baselines to compare against" items 3 and 5. Built in
`docs/superpowers/plans/2026-09-24-nfl-props-rec-yards-point-estimate-composition.md`.
**Inert:** no CLI command reads this composition -- this is a research checkpoint on
whether Stage 1's decomposition beats the existing model, not a shipping decision.

| Model | n | MAE | RMSE |
|---|---|---|---|
| Composition (Targets x CatchRate x Yards\|Reception) | `4490` | `17.970320761320977` | `25.44482184833516` |
| Existing direct model, full population (median prediction) | `6387` | `13.463997241868164` | `22.288854954881792` |
| Existing direct model, **matched subset** (same `4490` rows as composition -- apples-to-apples) | `4490` | `17.740159480038965` | `26.343101223202684` |
| Existing direct model, matched subset, **mean prediction** (like-for-like estimator) | `4490` | `17.930798126483605` | `25.563644681842522` |

**Correction (post-review):** the first version of this section compared the composition
(`n=4490`) against the direct model's **full-population** numbers (`n=6387`) and called
that an "unambiguous loss." That framing was wrong -- it compared two different
populations. The `1897`-row gap is not a random subsample: **100% of the skipped rows
have `targets == 0` in that specific game** (the player was on the field/roster for a
game where the offense simply never threw to him -- a run-heavy or game-script-driven
week, unrelated to career history or experience; the original write-up's claim that these
were disproportionately "rookies, first appearances, low-usage players" was not checked
against the data and is not correct), and **99.89% of those rows have `actual_yards == 0`**
(mean `0.009488666315234581` -- essentially always exactly zero). The direct model's own
MAE on just that skipped slice alone is `3.3427698041312643` -- a near-free,
trivially-predictable slice that pulls its full-population average down relative to what
it actually achieves on the harder population the composition was scored on. Restricting
the direct model to the *same* `4490` `(player_id, date)` rows the composition faced (no
free zero-target games on either side) gives the row above: MAE `17.740159480038965`,
RMSE `26.343101223202684` -- this matched-subset row, not the full-population row, is the
honest apples-to-apples comparison and the one the "Read honestly" paragraph below is
based on. The full-population row is kept in the table above for transparency but should
**not** be read as the composition's real relative performance.

**A second correction (same review pass):** the matched-subset comparison above still
compares two different kinds of point prediction -- the composition returns a MEAN
(`markets.mean_yards`), while the direct model's own comparison figure is its MEDIAN
(`exp(model_mu) - OFFSET`). MAE is minimized by a median predictor and RMSE by a mean
predictor, so this estimator mismatch mechanically favors a median predictor on MAE and a
mean predictor on RMSE, regardless of which underlying model is better. Scoring the direct
model on its own mean instead (same ratings, same matched rows, same run) gives MAE
`17.930798126483605` and RMSE `25.563644681842522` -- a like-for-like comparison. Under
that comparison, `rel_mae = (17.930798126483605 - 17.970320761320977) / 17.930798126483605
= -0.0022041759969957673` (composition **0.22% worse**, essentially a tie) and `rel_rmse =
(25.563644681842522 - 25.44482184833516) / 25.563644681842522 = 0.004648117863716067`
(composition **0.46% better**, a real but very small edge). The honest read of a properly
estimator-matched comparison is a **near-exact dead heat on both metrics** -- not the
"narrowly loses MAE, really wins RMSE (+3.41%)" framing the mean-vs-median comparison above
implies.

**Read honestly:** The primary, estimator-matched comparison (composition mean vs. direct
model mean, `n=4490` on both sides) is a genuine **near-exact dead heat**: composition MAE
is `0.22%` worse and RMSE is `0.46%` better than the direct model's own mean prediction --
both differences are small enough, on a window this size, to read as noise rather than a
real edge either way. For context, the mean-vs-median comparison (composition mean vs. the
direct model's median, its comparison figure elsewhere in this file) shows a larger spread
-- composition MAE `1.30%` worse, RMSE `3.41%` better -- but that spread is now understood
to be partly an artifact of comparing two different estimator types (MAE favors a median
predictor, RMSE favors a mean predictor), not a clean reflection of relative model quality.
Quoting the spec's own gate directly: "If Model 6 doesn't beat Model 5 ... it is not
shipped." Neither comparison constitutes a clear pass of that gate -- the composition does
not clearly beat the direct model on either metric, under either estimator pairing -- but
this is also not a clear loss: the small mean-vs-mean RMSE edge (`+0.46%`) is real, just
small, and the MAE gap is close enough to call a tie. This is a materially different, and
much closer, result than the original write-up's false "unambiguous loss, no winning
framing" claim. (Note for the reader: this stage validates the POINT ESTIMATE only, not the
full NLL/calibration criteria the spec's gate literally names for the eventual full Monte
Carlo model -- MAE/RMSE is the honest analog available at this stage, not a substitute for
the real gate check the full Monte Carlo plan will need to run. A near-dead-heat
point-estimate result is genuinely ambiguous input for that later decision, not a clear
green light or a clear stop.)

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
import numpy as np
import pandas as pd
from nfl_props import backtest, catch_model, composition, data, model, rate_model, yards_model

games = data.load_games()
pbp = data.load_pbp(seasons=3)
stats = data.load_player_stats(seasons=3)
targets = data.load_targets(pbp, stats, games)
catches = targets[targets['complete'] == 1.0].copy()

stats_share = model.add_trailing_share(stats, 'targets')
targets_ay = model.add_trailing_air_yards(targets)
game_ay = targets_ay.drop_duplicates(['player_id', 'game_id'])[['player_id', 'game_id', 'trailing_air_yards']]

relevant = stats_share[stats_share['position_group'].isin(model.RELEVANT_POSITIONS['rec_yds'])].sort_values('date')
relevant = relevant.merge(game_ay, on=['player_id', 'game_id'], how='left')

start = (stats['date'].max() - timedelta(days=365)).date()
start_ts = pd.Timestamp(start)

rows = []
skipped_no_air_yards = 0
rate_r = catch_r = yards_r = None
fit_week = None
for _, g in relevant.iterrows():
    if g['date'] < start_ts:
        continue
    week_key = (g['season'], g['week'])
    if week_key != fit_week:
        as_of = g['date'].date()
        rate_r = rate_model.fit_poisson(stats_share, 'targets', as_of=as_of, halflife_days=180.0, reg=5.0, min_games=200)
        catch_r = catch_model.fit_catch_rate(targets, as_of=as_of, halflife_days=180.0, reg=5.0, min_targets=200)
        yards_r = yards_model.fit_yards_per_catch(catches, as_of=as_of, halflife_days=180.0, reg=5.0, min_catches=200)
        fit_week = week_key
    ay = g['trailing_air_yards']
    if pd.isna(ay):
        skipped_no_air_yards += 1
        continue  # no targets in this game at all -- nothing to average; there is no
                  # safe air_yards fallback (see composition.py's module docstring)
    point_est = composition.predicted_rec_yds_point_estimate(
        rate_r, catch_r, yards_r, g['player_id'], g['position_group'], g['opponent_team'],
        bool(g['home']), trailing_air_yards=float(ay), trailing_share=g.get('trailing_share'))
    rows.append({'actual': g['receiving_yards'], 'point_est': point_est, 'player_id': g['player_id'], 'date': g['date']})

comp_df = pd.DataFrame(rows)
comp_mae = float(np.mean(np.abs(comp_df['actual'] - comp_df['point_est'])))
comp_rmse = float(np.sqrt(np.mean((comp_df['actual'] - comp_df['point_est']) ** 2)))
print('Composition:', {'n': len(comp_df), 'skipped_no_air_yards': skipped_no_air_yards, 'MAE': comp_mae, 'RMSE': comp_rmse})

direct_preds = backtest.walk_forward(stats, 'rec_yds', start, halflife_days=180.0, reg=5.0, min_games=200)
direct_median = np.exp(direct_preds['model_mu']) - model.OFFSET
direct_preds = direct_preds.assign(direct_median=direct_median)
direct_mae = float(np.mean(np.abs(direct_preds['actual_yards'] - direct_preds['direct_median'])))
direct_rmse = float(np.sqrt(np.mean((direct_preds['actual_yards'] - direct_preds['direct_median']) ** 2)))
print('Direct model (full population):', {'n': len(direct_preds), 'MAE': direct_mae, 'RMSE': direct_rmse})

# Matched-subset comparison (the honest apples-to-apples read): restrict the direct
# model's own already-computed predictions to the SAME (player_id, date) rows the
# composition was actually scored on, since the full population above includes 1897
# zero-target-that-game rows the composition skipped (no trailing_air_yards) that are
# trivially easy for the direct model (actual_yards == 0 for 99.89% of them).
comp_keys = comp_df[['player_id', 'date']].drop_duplicates()
matched = direct_preds.merge(comp_keys, on=['player_id', 'date'], how='inner')
matched_mae = float(np.mean(np.abs(matched['actual_yards'] - matched['direct_median'])))
matched_rmse = float(np.sqrt(np.mean((matched['actual_yards'] - matched['direct_median']) ** 2)))
print('Direct model (matched subset):', {'n': len(matched), 'MAE': matched_mae, 'RMSE': matched_rmse})

matched_mean = np.exp(matched['model_mu'] + matched['model_sigma'] ** 2 / 2) - model.OFFSET
matched_mean_mae = float(np.mean(np.abs(matched['actual_yards'] - matched_mean)))
matched_mean_rmse = float(np.sqrt(np.mean((matched['actual_yards'] - matched_mean) ** 2)))
print('Direct model (matched subset, mean):', {'n': len(matched), 'MAE': matched_mean_mae, 'RMSE': matched_mean_rmse})
"
```

## rec_yds full Monte Carlo composition (first validation)

**Commit:** `6a7fc3e` (master; fixes a units-mismatch baseline bug and a shared-seed
decorrelation bug found by a final whole-branch review of this plan -- see below).
**Data window:** 3 seasons of real nflverse data, 1-year walk-forward lookback, run
2026-09-25. **Method:** `composition_backtest.walk_forward()` refits
`rate_model`/`catch_model`/`yards_model` weekly, draws 10,000 Monte Carlo samples per
in-window WR/TE/RB/QB player-game via `composition.simulate_rec_yds()`, moment-matches a
log-normal to each sample in `log(receiving_yards + OFFSET)` space, and scores real
observed `receiving_yards` against that fit -- compared, in the same script and window,
against `model.py`'s existing direct `rec_yds` fit via `backtest.walk_forward()`
(existing, unchanged), per the spec's own "Baselines to compare against" items 5 and 6
and the spec's literal gate: "If Model 6 doesn't beat Model 5 on out-of-sample NLL and
calibration, it is not shipped." Built in
`docs/superpowers/plans/2026-09-24-nfl-props-rec-yards-monte-carlo-composition.md`.
**Inert:** no CLI command reads this composition -- this is a research checkpoint on
whether Stage 1's full decomposition beats the existing model, not a shipping decision.

**This section replaces, not merely amends, an earlier committed write-up (commit
`add0e8e`) that claimed a flat "49.6% relative NLL loss, worse than its own baseline."**
A final whole-branch review found that headline was **substantially an artifact**: (1) a
real bug in `composition_backtest.py`'s baseline computation (it approximated a
per-*game* log-yards baseline using `yards_model`'s per-*catch* efficiency parameters --
a units mismatch), and (2) 11 QB player-games out of 4275 (0.26% of the population)
carried a wildly disproportionate share of the total NLL gap, driven by a known,
documented-but-not-fixed methodology issue (zero-inflated Monte Carlo samples collapsing
the moment-matched log-normal's fitted sigma). This rewrite fixes bug (1), decorrelates
a second real bug (every row previously reused the identical RNG seed), documents
issue (2) as a confirmed, deliberately-not-fixed structural limitation, and reports the
resulting corrected numbers honestly below -- including a genuinely surprising
consequence of the seed-decorrelation fix explained in detail further down.

| Model | n | NLL (model) | NLL (season-to-date baseline) |
|---|---|---|---|
| Monte Carlo composition (Targets x CatchRate x Yards\|Reception) | `4275` | `2.2351073116720697` | `1.3400655041895395` |
| Existing direct model, full population | `6074` | `0.8681852592317548` | `0.9406648684926634` |
| Existing direct model, **matched subset** (same `4275` rows the composition actually scored -- apples-to-apples) | `4275` | `1.250694422136362` | `1.403724673315056` |

**The corrected baseline retracts the ORIGINAL committed bug, but not the headline
claim.** The old committed `mc_nll_baseline` (`1.592887277155639`) was computed by
falling back to `yards_r.position_intercept`/`yards_r.sigma` -- the per-catch
yards-per-reception model's own fitted parameters -- to approximate a per-*game*
season-to-date baseline. `y` in this harness is `log(receiving_yards + OFFSET)` at the
**game** level, not the per-catch level, so that fallback was measuring the wrong thing.
The fix (`composition_backtest.py`, this commit) fits the existing, unchanged
game-level `model.fit(stats, "rec_yds", ...)` weekly, purely to source this fallback --
mirroring `backtest.py`'s own `walk_forward()` baseline PATTERN, though not an identical
fit: this call passes raw `stats` (no `trailing_share` column), whereas `backtest.py`'s
own equivalent call first runs `stats = model.add_trailing_share(stats, stat, ...)`
before fitting (since `rec_yds` is in `model.SHARE_STATS`), so `backtest.py`'s baseline
fit includes the trailing-share covariate and this composition's baseline fit does not.
This is not a bug -- the no-share intercept is arguably a more conservative baseline (it
makes the baseline harder to beat) -- but it does mean the two `nll_baseline` values
compared in this section are not from an identical fit: an identical-fit recomputation
(passing the `trailing_share`-augmented frame, already computed earlier in
`walk_forward()`, into this same `model.fit()` call) gives `nll_baseline=1.403724673315056`
instead of the committed `1.3400655041895395`. This affects `361` of `4275` rows (`8.4%`,
the intercept-fallback rows) with a real, material per-row difference -- e.g. the WR
fallback intercept is `3.2886` under the no-share fit used here vs. `2.5628` under
`backtest.py`'s exact-pattern fit. The real, corrected baseline actually reported on this
run is `nll_baseline=1.3400655041895395`, not the old, buggy `1.592887277155639`. **This
does retract the specific numeric claim "own baseline is 1.592887277155639"** -- but, as
shown in the table above, the Monte Carlo composition's own `nll_model`
(`2.2351073116720697`) is *still* worse than this corrected, tighter
baseline, by `0.8950418074825301` nats (`66.8%` relative) -- a **larger** margin than
the original (buggy-baseline) write-up's `17.4%`, not a smaller one. See "Why the
full-population number got worse, not better" below for why -- this is not a
contradiction of the baseline fix, it is a real, separately-caused, and informative
result.

**The full-population headline percentages above are themselves seed-dependent --
read them as one of several plausible single-seed draws, not a precise, stable
measurement.** A follow-up re-review re-ran this exact walk-forward at 4 independent
base seeds (all other code and data held identical to this run) and measured
`mc_summary['nll_model']` ranging from `1.8472` to `2.2351` across those 4 seeds -- a
`21%` range on the raw `nll_model` value alone. That range propagates to a
loss-vs-own-(corrected)-baseline range of `37.8%` to `66.8%`, and a loss-vs-direct-model
range of `47.7%` to `78.7%`. The committed run reported throughout this section
(`seed=2026`) is confirmed to be the **worst** (least favorable to the composition) of
the 4 seeds tested. The *sign* of the conclusion -- the composition loses to both its own
corrected baseline and the direct model on the full population -- is robust across all 4
seeds tested. The specific *magnitude* (`66.8%`, `78.7%`, and any numeric comparison
below to the original pre-fix write-up's `17.4%`/`49.6%`) is not: both this run's
percentages and the original write-up's percentages are single seed-dependent draws, not
precise, stable measurements, and any comparison between them should be read as a
comparison between two single draws, not as a precise change in magnitude.

| | seed range across 4 tested seeds | this committed run (`seed=2026`) |
|---|---|---|
| `nll_model` (full population) | `1.8472` -- `2.2351` | `2.2351` (worst of the 4) |
| Loss vs. own corrected baseline | `37.8%` -- `66.8%` | `66.8%` |
| Loss vs. direct model (matched) | `47.7%` -- `78.7%` | `78.7%` |

**Population check (unchanged from the original write-up -- this fix wave did not touch
the skip-population mechanism):** `direct_summary['n']` (`6074`) and `mc_summary['n']`
(`4275`) differ by `1799` rows -- `1799 / 6074 = 0.296180441224893` (29.62%) of the
direct model's full population. Of the `1799` skipped rows, `0.9988882712618121`
(99.89%) have `actual_yards == 0` in that specific game (mean `actual_yards` across the
skipped rows: `0.010005558643690939`). These are **player-games with zero targets
recorded in that specific game (a join-miss against the targets table), not new players
or low-history players** -- `model.add_trailing_air_yards` never returns NaN on its own
output; the only way `trailing_air_yards` is NaN in the merged walk-forward frame is a
player-game where the player recorded literally zero targets that specific game, so no
row exists for that `game_id` in the targets table and the left-merge produces NaN. A
gap this large requires the population-matched re-run used throughout this section (not
the raw, unequal-n row) as the trusted comparison.

**Position-split breakdown (freshly computed from this run's own predictions, not cited
from the whole-branch review):**

| Position | n | MC `nll_model` | Direct (matched) `nll_model` |
|---|---|---|---|
| QB | `11` | `489.83259280905554` | `105.9429617605317` |
| RB | `1064` | `0.9132808931908387` | `0.930345824040724` |
| TE | `1117` | `0.8767223893986158` | `0.8803224297199099` |
| WR | `2083` | `1.0638000274532793` | `1.060075835089116` |

`11` QB player-games are `11 / 4275 = 0.0025730994152046785` (`0.26%`) of the scored
population, but account for `5388.158520899611 / 9555.083757398097 = 0.5639048968804473`
(**56.4%**) of the composition's total summed NLL (`sum(row_nll)` across all `4275`
rows). Both models score badly on these same 11 rows (the direct model's own
`nll_model` on them, `105.9429617605317`, is two orders of magnitude worse than its
RB/TE/WR rows too) -- QBs are a genuinely hard, outlier-heavy, near-zero-usage
sub-population for *any* log-normal-style model in this window, real games included
(**three** of the 11 rows are negative-yardage plays -- `00-0023459` (`actual_yards=-9`),
`00-0033873` (`actual_yards=-10`), `00-0040234` (`actual_yards=-6`) -- though only the
`-10` row (`00-0033873`) is actually floored by `model._safe_log_yards`, which floors at
`1.0 - OFFSET = -9`; the `-9` and `-6` rows both sit above that floor and are not
floored). But the composition's failure on these 11 rows is roughly
`489.83259280905554 / 105.9429617605317 = 4.62x` worse than the direct model's, because
the Monte Carlo sample for a near-zero-catch-probability player is itself almost
entirely exact zeros, and moment-matching one smooth log-normal to that near-degenerate
sample collapses the fitted `sigma_hat` to a tiny value (`0.0136`-`0.0667` on these 11
rows) -- so any real deviation from the degenerate mode produces an enormous
`(y - mu)^2 / (2 * sigma^2)` penalty. This is the zero-inflation/moment-matching
mismatch documented as a confirmed, not-fixed structural limitation below.

**Ex-QB and WR-only aggregates (freshly re-derived from this run, not copied from the
whole-branch review's earlier numbers):**

| Population | n | MC `nll_model` | Direct (matched) `nll_model` | Read |
|---|---|---|---|---|
| Ex-QB (RB+TE+WR) | `4264` | `0.9772338734752551` | `0.980615871310295` | MC **better** by `0.0033819978350399` nats (`0.34%` relative) |
| WR-only (spec's actual scoped market) | `2083` | `1.0638000274532793` | `1.060075835089116` | MC **worse** by `0.0037241923641633257` nats (`0.35%` relative) |

Excluding the 11 QB rows, the composition is a statistical dead heat with the existing
direct model -- fractionally ahead ex-QB, fractionally behind on WR alone -- both
differences under half a percent. A follow-up re-review measured the REAL seed noise on
these two specific numbers directly, by re-running the full walk-forward at 4 independent
base seeds (all other code/data identical): the seed-to-seed noise on the ex-QB and
WR-only gaps is `0.087%`, roughly **4x smaller** than the `0.34%`/`0.35%` gaps themselves
-- so these gaps are NOT "inside sampling noise." The stronger and more honest read is
the opposite: both gaps are small in absolute terms **and** stable in sign across all 4
tested seeds (ex-QB: MC better by `0.32%`-`0.40%` across all seeds; WR-only: MC worse by
`0.33%`-`0.42%` across all seeds) -- a small, sign-stable gap is better evidence for a
genuine dead heat than a noise-based dismissal would have been, since it rules out "the
sign itself is just noise" rather than merely asserting the gap is too small to trust.
This matches, within that same seed-to-seed variation, the whole-branch review's own
earlier finding on the pre-fix run (ex-QB MC 0.31% better, WR-only 0.38% dead heat) --
these two aggregates are **stable** across both the original shared-seed run and this fix
wave's decorrelated-seed run, unlike the full-population number.

**Why the full-population number got worse, not better, after this fix wave (a real,
investigated finding, not a bug in this fix wave's own code):** the corrected baseline
(`1.3400655041895395`, vs. the old buggy `1.592887277155639`) is tighter and more
accurate, as intended. But `mc_summary['nll_model']` itself moved from the originally
committed `1.8707843655646479` to `2.2351073116720697` -- **worse**, not better --
because of Fix 2 (decorrelating the per-row Monte Carlo seed). Before this fix, every
scored row's `simulate_rec_yds()` call reused the exact same `seed=2026`, so every row's
Monte Carlo sampling error was driven by the identical underlying random stream. After
decorrelating (`seed + row_counter` per row), the same 11 QB rows' near-degenerate,
almost-all-zero samples are no longer all subject to the same shared-seed "luck" -- and
in this run, several of them landed on an even more extreme `sigma_hat` collapse than
the original shared-seed run happened to produce (the single worst row, player
`00-0040234`, has `model_sigma=0.013586` and contributes `2273.6` nats of NLL alone,
larger than any single-row blowup the whole-branch review's pre-fix investigation cited).
This is **additional, real confirmation that the zero-inflation moment-matching mismatch
(documented below) is a genuine instability, not a one-off artifact of a single shared
seed** -- decorrelating the seed did not fix or mask it, it exposed that the QB-row
failure mode is itself highly seed-sensitive on the sub-population where it occurs, while
leaving the much larger RB/TE/WR-only aggregates essentially unchanged. This is exactly
the behavior expected if the real, underlying problem is what limitation (2) below
describes, and rules out "the original 49.6% number was just an unlucky shared seed
happening to look bad" as the explanation -- the truth is closer to "the shared seed
happened to look *less* bad than average for this specific, structurally fragile
sub-population."

**Calibration (PIT deciles, model's own distribution):**

Monte Carlo composition (`n=4275`, this fixed run):
```
                 n  mean_pit
(-0.001, 0.1]  406  0.049932
(0.1, 0.2]     422  0.149645
(0.2, 0.3]     383  0.247763
(0.3, 0.4]     351  0.348259
(0.4, 0.5]     373  0.453466
(0.5, 0.6]     376  0.548610
(0.6, 0.7]     404  0.650118
(0.7, 0.8]     461  0.751005
(0.8, 0.9]     506  0.852450
(0.9, 1.0]     593  0.953377
```

Existing direct model, full population (`n=6074`, unchanged -- `backtest.py` was not
touched by this fix wave):
```
                 n  mean_pit
(-0.001, 0.1]  576  0.053340
(0.1, 0.2]     578  0.150158
(0.2, 0.3]     647  0.252241
(0.3, 0.4]     751  0.351025
(0.4, 0.5]     704  0.449163
(0.5, 0.6]     579  0.548553
(0.6, 0.7]     480  0.650764
(0.7, 0.8]     514  0.749635
(0.8, 0.9]     562  0.851457
(0.9, 1.0]     683  0.953254
```

Existing direct model, matched subset (`n=4275`, byte-for-byte identical to the
originally committed write-up -- confirms the direct-model side of this comparison is
completely unaffected by this fix wave, as expected):
```
                 n  mean_pit
(-0.001, 0.1]  412  0.049378
(0.1, 0.2]     337  0.147157
(0.2, 0.3]     361  0.249904
(0.3, 0.4]     352  0.350567
(0.4, 0.5]     365  0.452047
(0.5, 0.6]     399  0.551424
(0.6, 0.7]     402  0.650776
(0.7, 0.8]     452  0.750503
(0.8, 0.9]     526  0.851416
(0.9, 1.0]     669  0.953359
```

Per-bin `mean_pit` values sit close to their bin's own target center for both models
across every bucket (e.g. the `(0.4, 0.5]` bin: `0.453466` for the composition,
`0.452047` for the direct matched subset, both near `0.45`). Bin **counts** skew toward
the top decile for both (expected `427.5`/bin): the composition's `(0.9, 1.0]` bin holds
`593`, `38.7%` above expected; the direct matched subset's `(0.9, 1.0]` bin holds `669`,
`56.5%` above expected -- the direct model's raw bin-count skew is, if anything, larger.
Calibration does not offer the composition a compensating advantage against its NLL
loss, but it also does not make the direct model look cleanly better-calibrated either --
same read as the original write-up, unaffected by this fix wave.

**Confirmed, NOT-fixed structural limitations (documented here as open findings for a
future plan, per the final whole-branch review's ruling -- both are real, both are too
large in scope for this bounded fix wave):**

1. **Population-conditioning mismatch.** `composition.simulate_rec_yds()` draws Targets
   unconditionally (correct, general-purpose semantics for "what if this player played
   this game") -- but `composition_backtest.walk_forward()`'s scored population is
   *implicitly* conditioned on the player having recorded `>= 1` target that specific
   game, because a player-game only has a row in `game_ay` (and therefore a non-NaN
   `trailing_air_yards`) if `targets_df` has at least one row for that `(player_id,
   game_id)` (see the Population check above, and `composition_backtest.py`'s own
   module docstring). This means part of the MC sample's zero mass corresponds to a
   `targets == 0` outcome that is *structurally impossible* in the population actually
   being scored -- every scored row is, by construction, a game where the player did
   record at least one target. This inflates the zero-fraction (and therefore
   destabilizes the moment-matched `sigma_hat`) beyond what the real, conditional
   distribution would produce. Fixing this properly requires reworking how
   `trailing_air_yards` is looked up so it does not require a same-game target row (e.g.
   sourcing it from the player's most recent prior game instead of the current one) --
   a genuine redesign of the leakage-safe feature-lookup path, out of scope for this
   fix wave.
2. **Zero-inflation / single-log-normal moment-matching mismatch.** A meaningful
   fraction of any given player-game's Monte Carlo draws are exactly `0.0` (any draw
   with zero catches gives `total_yards == 0.0` exactly), most severely for
   near-zero-usage players (QBs in this window: see the position-split table above).
   `_moment_match_lognormal()` fits ONE smooth log-normal to this mixture distribution
   by taking the sample mean/std of `log(mc_sample + OFFSET)` -- a reasonable default,
   but a poor fit specifically on the affected rows, where it can produce an
   artificially tiny `sigma_hat` (see the QB rows above, `0.0136`-`0.0667`) that then
   produces enormous NLL penalties on any row where the real outcome deviates even
   slightly from the degenerate near-zero mode. A whole-branch review's own
   investigation (prior to this fix wave) found that alternative scorings of the exact
   same MC samples -- empirical CRPS and a mixed-measure log score with an explicit
   atom at zero -- show the composition competitive-to-better against the direct model
   (CRPS 12.8029 vs 12.8329, MC +0.23%; mixed-measure log score 1.0671 vs 1.2032, MC
   +11.3%), supporting the reading that the single-log-normal NLL metric itself, not
   necessarily the underlying composition, is the primary source of the catastrophic
   QB-row failure mode. Properly fixing this would mean replacing the single-log-normal
   moment-match with a real mixture-model likelihood (e.g. a point mass at zero plus a
   log-normal for the nonzero component) -- a genuine methodology redesign, out of scope
   for this fix wave.

**Read honestly, per the spec's literal gate ("If Model 6 doesn't beat Model 5 ... it is
not shipped"):** on the full, population-matched comparison (`n=4275` both sides), the
Monte Carlo composition does **not** beat the existing direct model -- `mc_nll_model`
(`2.2351073116720697`) is worse than `direct_matched_nll_model` (`1.250694422136362`) by
`0.9844128895357076` nats (`78.7%` relative), and it is also worse than its own
corrected, tighter baseline (`66.8%` relative, see above) -- both larger margins than the
original (buggy-baseline, shared-seed) write-up reported. As noted above, these specific
magnitudes are themselves seed-dependent (measured range `47.7%`-`78.7%` vs. direct,
`37.8%`-`66.8%` vs. own baseline, across 4 tested seeds) and the committed run sits at the
unfavorable end of that range -- the *sign* of "loses on the full population" is robust
across all 4 seeds tested, but this exact percentage, and its comparison to the original
write-up's percentage, should not be read as precise. The loss is driven almost entirely
by `11` QB rows (`0.26%` of the population, `56.4%` of the total NLL sum) whose failure mode is
the documented, deliberately-not-fixed zero-inflation/moment-matching mismatch (and, per
the investigation above, is itself unstable under seed decorrelation, which is why this
number moved in the "wrong" direction after two real bug fixes). Excluding those 11 QB
rows, the picture is materially different: the composition is a genuine, roughly
half-a-percent dead heat with the direct model both ex-QB (MC `0.34%` better) and on
WR-only, the spec's actual scoped market (MC `0.35%` worse) -- neither a win nor a loss
large enough to mean anything against real Monte Carlo/refit noise. **A dead heat is not
"beats Model 5."** Whether scored on the full population (a clear, large loss, now shown
to be concentrated in a known, documented, not-yet-fixed QB failure mode) or on the
spec's actual WR-scoped market (a dead heat, not a win), this result does not clear the
spec's gate either way: **Model 6 (this full Monte Carlo composition) is not shipped**,
per the spec's own literal language. This is a materially more honest and more useful
picture than the original write-up's flat "loses badly, worse than its own baseline"
framing -- the real story is "the composition is roughly competitive on its actual
scoped market, but a small, structurally-understood QB sub-population, plus a known
single-log-normal scoring mismatch on zero-inflated samples, currently make the full,
unscoped comparison look far worse than that" -- not "the whole approach is a large,
unqualified failure." Neither framing changes the shipping decision: this composition
does not ship as-is. The natural next steps -- reworking the trailing_air_yards lookup to
remove the population-conditioning mismatch, and/or replacing the single-log-normal
moment-match with a real zero-inflated mixture likelihood -- are both real, scoped
follow-up work, not attempted here.

**Reproducing this result:**

```bash
cd nfl-props
.venv\Scripts\python -c "
from datetime import timedelta
import numpy as np
import pandas as pd
from nfl_props import backtest, composition_backtest, data, model

games = data.load_games()
pbp = data.load_pbp(seasons=3)
stats = data.load_player_stats(seasons=3)
targets = data.load_targets(pbp, stats, games)

start = (stats['date'].max() - timedelta(days=365)).date()

mc_preds = composition_backtest.walk_forward(
    stats, targets, start, halflife_days=180.0, reg=5.0,
    min_games=200, min_targets=200, min_catches=200, n_draws=10_000, seed=2026)
mc_summary = composition_backtest.summarize(mc_preds)
mc_cal = composition_backtest.calibration(mc_preds)
print('Monte Carlo composition summarize():', mc_summary)
print('Monte Carlo composition calibration():')
print(mc_cal)

direct_preds = backtest.walk_forward(stats, 'rec_yds', start, halflife_days=180.0, reg=5.0, min_games=200)
direct_summary = backtest.summarize(direct_preds)
direct_cal = backtest.calibration(direct_preds)
print('Direct model summarize():', direct_summary)
print('Direct model calibration():')
print(direct_cal)
"
```

The population-matched direct-model subset (used for the trusted comparison above) is
computed by filtering `direct_preds` down to the same `(player_id, date)` pairs
`mc_preds` scored, then recomputing NLL/calibration with `backtest.py`'s own `_nll`/PIT
formulas on that subset:

```python
mc_keys = mc_preds[['player_id', 'date']].drop_duplicates()
matched = direct_preds.merge(mc_keys, on=['player_id', 'date'], how='inner')
```

The position-split, ex-QB, and WR-only breakdowns above are computed by adding a
`position_group` groupby on `mc_preds` and the `matched` direct-model subset, using each
module's own `_nll` formula (`composition_backtest._nll` / `backtest._nll`, same math)
per position group and on the `!= 'QB'` / `== 'WR'` filtered subsets:

```python
def row_nll(y, mu, sigma, eps=composition_backtest.EPS):
    sigma = np.maximum(sigma, eps)
    return 0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)

mc_preds['row_nll'] = row_nll(mc_preds['y'].to_numpy(), mc_preds['model_mu'].to_numpy(), mc_preds['model_sigma'].to_numpy())
matched['row_nll'] = row_nll(matched['y'].to_numpy(), matched['model_mu'].to_numpy(), matched['model_sigma'].to_numpy())

position_split = mc_preds.groupby('position_group')['row_nll'].agg(['size', 'mean'])
qb_frac_of_gap = mc_preds.loc[mc_preds['position_group'] == 'QB', 'row_nll'].sum() / mc_preds['row_nll'].sum()

exqb_mc = mc_preds[mc_preds['position_group'] != 'QB']
exqb_direct = matched[matched['position_group'] != 'QB']
wr_mc = mc_preds[mc_preds['position_group'] == 'WR']
wr_direct = matched[matched['position_group'] == 'WR']
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
