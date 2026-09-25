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

**Commit:** `6214e0a` (master). **Data window:** 3 seasons of real nflverse data, 1-year
walk-forward lookback, run 2026-09-25. **Method:** `composition_backtest.walk_forward()`
(Task 2) refits `rate_model`/`catch_model`/`yards_model` weekly, draws 10,000 Monte Carlo
samples per in-window WR/TE/RB/QB player-game via `composition.simulate_rec_yds()` (Task
1), moment-matches a log-normal to each sample in `log(receiving_yards + OFFSET)` space,
and scores real observed `receiving_yards` against that fit -- compared, in the same
script and window, against `model.py`'s existing direct `rec_yds` fit via
`backtest.walk_forward()` (existing, unchanged), per the spec's own "Baselines to compare
against" items 5 and 6 and the spec's literal gate: "If Model 6 doesn't beat Model 5 on
out-of-sample NLL and calibration, it is not shipped." Built in
`docs/superpowers/plans/2026-09-24-nfl-props-rec-yards-monte-carlo-composition.md`.
**Inert:** no CLI command reads this composition -- this is a research checkpoint on
whether Stage 1's full decomposition beats the existing model, not a shipping decision.

| Model | n | NLL (model) | NLL (season-to-date baseline) |
|---|---|---|---|
| Monte Carlo composition (Targets x CatchRate x Yards\|Reception) | `4275` | `1.8707843655646479` | `1.592887277155639` |
| Existing direct model, full population | `6074` | `0.8681852592317548` | `0.9406648684926634` |
| Existing direct model, **matched subset** (same `4275` rows the composition actually scored -- apples-to-apples) | `4275` | `1.250694422136362` | `1.403724673315056` |

**Population check (Step 2, required before trusting the comparison):**
`direct_summary['n']` (`6074`) and `mc_summary['n']` (`4275`) differ by `1799` rows --
`1799 / 6074 = 0.296180441224893` (29.62%) of the direct model's full population. This is
the same order of magnitude as the population gap this project already established once
before, in the `rec_yds point-estimate composition` section above (`1897 / 6387 = 0.297`,
29.7%), and the same mechanism: of the `1799` skipped rows, `0.9988882712618121` (99.89%)
have `actual_yards == 0` in that specific game (mean `actual_yards` across the skipped
rows: `0.010005558643690939`, essentially always exactly zero). These are **player-games
with zero targets recorded in that specific game (a join-miss against the targets
table), not new players or low-history players.** `model.add_trailing_air_yards` never
returns NaN on its own output -- a low-sample or brand-new player falls back to a
group/overall average; the ONLY way `trailing_air_yards` is NaN in the merged
walk-forward frame is a player-game where the player recorded literally zero targets in
that specific game, so no row exists for that `game_id` in the targets table and the
left-merge produces NaN. A 10-year veteran can trigger this on any game where they simply
weren't targeted -- this has nothing to do with career history. A gap this large (29.6%
of the population) clearly requires a population-matched re-run per this project's own
established discipline (first applied in the point-estimate composition plan's own Task
3 fix) before trusting the comparison -- the raw, unequal-n row above (`4275` vs `6074`)
is **not** the trusted comparison; the matched-subset row (`n=4275` on both sides,
restricting `direct_preds` to the exact `(player_id, date)` pairs `mc_preds` actually
scored) is the one the "Read honestly" paragraph below is based on.

**Calibration (PIT deciles, model's own distribution):**

Monte Carlo composition (`n=4275`):
```
                 n  mean_pit
(-0.001, 0.1]  400  0.049656
(0.1, 0.2]     420  0.149773
(0.2, 0.3]     390  0.247735
(0.3, 0.4]     338  0.347858
(0.4, 0.5]     381  0.452510
(0.5, 0.6]     377  0.548787
(0.6, 0.7]     404  0.650102
(0.7, 0.8]     456  0.750509
(0.8, 0.9]     516  0.852377
(0.9, 1.0]     593  0.953729
```

Existing direct model, full population (`n=6074`):
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

Existing direct model, matched subset (`n=4275`, the trusted apples-to-apples
comparison -- same `(player_id, date)` rows as the Monte Carlo composition table above):
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

**Read honestly:** On the trusted, population-matched comparison (`n=4275` on both
sides), the Monte Carlo composition does **not** beat the existing direct model on NLL --
it loses badly. `mc_nll_model` (`1.8707843655646479`) is **worse** than
`direct_matched_nll_model` (`1.250694422136362`) by `0.6200899434282858` nats, a
`0.6200899434282858 / 1.250694422136362 = 0.49579652107913375` (49.6%) relative
disadvantage for the composition -- an order of magnitude larger than any margin recorded
elsewhere in this file (the largest prior loss, `Yards | Reception`'s
`position_split_defense` ablation, was `-0.435%`). Worse still: the Monte Carlo
composition does not even beat its **own** season-to-date baseline on its own
population -- `mc_nll_model` (`1.8707843655646479`) is **worse** than `mc_nll_baseline`
(`1.592887277155639`) by `0.27789708840900884` nats
(`0.27789708840900884 / 1.592887277155639 = 0.17446123928194063`, 17.4% relative), the
only model-vs-own-baseline comparison in this entire file that comes out negative -- every
other first-validation entry above (`targets`, `team pass volume`, `CatchRate | Targets`,
`Yards | Reception`) beats its own baseline. By contrast, the direct model's matched-subset
result is a normal, real win over its own baseline: `direct_matched_nll_model`
(`1.250694422136362`) beats `direct_matched_nll_baseline` (`1.403724673315056`) by
`0.15303025117869384` nats (`10.9%` relative improvement), in line with this file's other
established results. (Note: `mc_nll_baseline` and `direct_matched_nll_baseline` are
**not** on the same baseline formula -- the Monte Carlo composition's baseline falls back
to `yards_model`'s per-reception position intercepts, per `composition_backtest.py`'s own
docstring, while the direct model's baseline falls back to `model.fit`'s own
directly-fit `rec_yds` intercepts -- so only the `nll_model` column, not the
`nll_baseline` column, is a valid head-to-head read between the two models, matching this
file's own established caveat for the `snap_share` and `position_split_defense`
ablations above.)

Calibration does not rescue this result, and does not show a clean advantage for either
side. Per-bin `mean_pit` values sit close to their bin's own target center for both
models across every bucket (e.g. the `(0.4, 0.5]` bin: `0.452510` for the composition,
`0.452047` for the direct matched subset, both near `0.45`) -- so neither model shows an
obvious, gross PIT-centering failure. But bin **counts** (each of the 10 bins should hold
`4275 / 10 = 427.5` rows if uniform) skew toward the top decile for both models: the
composition's `(0.9, 1.0]` bin holds `593` rows, `38.7%` above the expected `427.5`
(`(593 - 427.5) / 427.5 = 0.3871345029239766`), while its `(0.3, 0.4]` bin (the
low-count outlier) holds `338`, `20.9%` below expected
(`(338 - 427.5) / 427.5 = -0.20935672514619882`). The direct matched subset's skew is, if
anything, slightly **larger**: its `(0.9, 1.0]` bin holds `669`, `56.5%` above expected
(`(669 - 427.5) / 427.5 = 0.5649122807017544`), and its `(0.1, 0.2]` bin (its low-count
outlier) holds `337`, `21.2%` below expected
(`(337 - 427.5) / 427.5 = -0.21169590643274855`). Neither table is uniform, and the
direct model's raw bin-count skew is not better than the composition's -- but this is not
a redeeming factor for the composition: its catastrophic NLL loss is not explained away
by a calibration advantage, since there isn't one. Both models show the same qualitative
top-decile-heavy pattern in this window, and the direct model still wins decisively on
NLL despite its own top-decile skew being larger in raw counts.

Quoting the spec's own gate directly: "If Model 6 doesn't beat Model 5 ... it is not
shipped." **This result fails that gate, plainly and by a wide margin.** The Monte Carlo
composition loses to the existing direct model on NLL by a 49.6% relative margin (using
the trusted, population-matched `n=4275` comparison, not the raw unequal-n numbers), and
it does not offer a compensating calibration advantage to weigh against that loss --
calibration is, if anything, comparably imperfect on both sides. This is a clear, honest
loss, not a partial win or a close call: **Model 6 (this full Monte Carlo composition) is
not shipped, per the spec's own literal gate.** The scale of this loss (49.6% relative
NLL, and losing to its own trivial baseline by 17.4%) is different in kind from the
point-estimate composition's earlier near-dead-heat result on MAE/RMSE -- point-estimate
accuracy and full-distribution NLL are measuring different things, and this result shows
the full Monte Carlo composition's moment-matched log-normal fit is a substantially worse
probabilistic model of `receiving_yards` than the existing direct fit, not merely a
smaller-margin loss on a stricter metric. This is real information this project's gate
discipline requires acting on honestly: this composition does not ship, and the natural
next step is investigating why the moment-matched fit underperforms so badly (e.g.
whether the log-normal moment-match is a poor fit to the true MC sample shape, or whether
compounding three separately-fit sub-models inflates variance) -- not scoped or attempted
by this task.

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
