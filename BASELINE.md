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
| Existing direct model (median prediction) | `6387` | `13.463997241868164` | `22.288854954881792` |

**Read honestly:** The composition does NOT beat the existing direct model on MAE or on
RMSE -- it loses on both, clearly. Composition MAE is 17.970320761320977 vs. the direct
model's 13.463997241868164 (composition's error is 4.506323519452813 yds higher).
Composition RMSE is 25.44482184833516 vs. the direct model's 22.288854954881792
(composition's error is 3.1559668934533676 yds higher). Using the required formula,
`(direct_mae - comp_mae) / direct_mae = (13.463997241868164 - 17.970320761320977) /
13.463997241868164 = -0.33469432877182836`, i.e. the composition's MAE is **33.47%
worse** (relatively) than the direct model's, not better. Likewise `(direct_rmse -
comp_rmse) / direct_rmse = (22.288854954881792 - 25.44482184833516) / 22.288854954881792
= -0.14159394458987834`, i.e. the composition's RMSE is **14.16% worse** (relatively)
than the direct model's. This is a real, unambiguous loss on both metrics, not a mixed or
partial result -- there is no framing under which the composed point estimate beats the
model it was built to replace in this run. Quoting the spec's own gate directly: "If
Model 6 doesn't beat Model 5 ... it is not shipped." This result, on its own, **fails**
that gate -- the composition is not Model 6 in the spec's numbering, but the same
discipline applies identically: a composed estimate that loses to the model it was meant
to replace does not ship, and nothing here suggests otherwise. (Note for the reader: this
stage validates the POINT ESTIMATE only, not the full NLL/calibration criteria the spec's
gate literally names for the eventual full Monte Carlo model -- MAE/RMSE is the honest
analog available at this stage, not a substitute for the real gate check the full Monte
Carlo plan will need to run; a point-estimate loss of this size makes it very unlikely
that the fuller criteria would reverse the verdict, but that check was not run here.)

`n` differs meaningfully between the two rows: composition `n=4490` vs. direct model
`n=6387`, a difference of exactly `1897` -- which equals the real
`skipped_no_air_yards` count from Step 1 verbatim (`1897` of the `6387` relevant
player-games in the window, **29.70%**, had no prior target history and therefore no
safe `trailing_air_yards` value, so the composition skipped them per `composition.py`'s
own no-fallback design; the direct model's `backtest.walk_forward()` scored all `6387`).
This is a real, uncontrolled difference in the two rows' populations, not merely a
technicality: the skipped rows are disproportionately players with little or no prior
target history (rookies, first appearances, low-usage players), who plausibly have lower,
less variable `receiving_yards` outcomes that a season-to-date-style direct model can
predict cheaply -- if so, their presence in the direct model's `n=6387` but absence from
the composition's `n=4490` could inflate the apparent gap between the two models to some
unknown degree in the direct model's favor. This script did not re-run the direct model
restricted to the same `4490`-row subset the composition scored, so this possibility is
not resolved here and is flagged as an open limitation of this comparison, not something
to wave away. That said, the size of the observed gap -- a 33.47% relative MAE loss and a
14.16% relative RMSE loss -- is large enough that a same-`n` re-run explaining away the
entire result would require the skipped population to be carrying an unusually large
share of the direct model's apparent advantage; nothing in this run establishes that it
does, and the honest reading is that the composition currently underperforms the direct
model on real data, gate discipline says it is not shipped, and the population-mismatch
caveat is a real question for the next check, not a way to discount the loss.

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
        continue  # no target history at all for this player-game -- skip; there is no
                  # safe air_yards fallback (see composition.py's module docstring)
    point_est = composition.predicted_rec_yds_point_estimate(
        rate_r, catch_r, yards_r, g['player_id'], g['position_group'], g['opponent_team'],
        bool(g['home']), trailing_air_yards=float(ay), trailing_share=g.get('trailing_share'))
    rows.append({'actual': g['receiving_yards'], 'point_est': point_est})

comp_df = pd.DataFrame(rows)
comp_mae = float(np.mean(np.abs(comp_df['actual'] - comp_df['point_est'])))
comp_rmse = float(np.sqrt(np.mean((comp_df['actual'] - comp_df['point_est']) ** 2)))
print('Composition:', {'n': len(comp_df), 'skipped_no_air_yards': skipped_no_air_yards, 'MAE': comp_mae, 'RMSE': comp_rmse})

direct_preds = backtest.walk_forward(stats, 'rec_yds', start, halflife_days=180.0, reg=5.0, min_games=200)
direct_median = np.exp(direct_preds['model_mu']) - model.OFFSET
direct_mae = float(np.mean(np.abs(direct_preds['actual_yards'] - direct_median)))
direct_rmse = float(np.sqrt(np.mean((direct_preds['actual_yards'] - direct_median) ** 2)))
print('Direct model (median):', {'n': len(direct_preds), 'MAE': direct_mae, 'RMSE': direct_rmse})
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
