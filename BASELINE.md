# Baseline metrics — pre-improvement snapshot

**Commit:** `5f91564` (master, 2026-09-17)
**Data window:** 1-year walk-forward lookback, 3 seasons of real nflverse data, run 2026-09-17
**Method:** `backtest.walk_forward()` — true walk-forward, refit every `(season, week)` using
only data strictly before that date, scored against a naive season-to-date-average baseline.

This file exists so future model changes can be compared against a fixed, real,
reproducible reference point rather than a shifting one — see `docs/superpowers/specs/`
in the parent repo for the design history behind every constant referenced below.

## pass_yds (n=693)

| Metric | Value |
|---|---|
| NLL (model / baseline) | 1.3020 / 1.3378 |
| MAE | 78.21 yds |
| RMSE | 97.24 yds |
| Brier (own median line, own p_over) | 0.2496 |
| Calibration: bottom decile / top decile | 12.4% / 2.9% (target ~10% each) |

**Known weak spot:** top-decile calibration is well below target — the model is
under-confident about real explosive passing games, concentrated in low-snap/backup QBs
(see `docs/superpowers/specs/2026-09-16-nfl-props-low-sample-sigma-design.md`). A fix
exists and was validated (NLL 1.3020→1.2396, top-decile 2.89%→3.61%) but is **not yet
merged to master** — see branch `nfl-props-sigma-contamination-fix`.

## rush_yds (n=6375)

| Metric | Value |
|---|---|
| NLL (model / baseline) | 0.1195 / 0.1640 |
| MAE | 6.73 yds |
| RMSE | 16.06 yds |
| Brier (own median line, own p_over) | 0.2818 |
| Calibration: bottom decile / top decile | 4.8% / 7.5% (target ~10% each) |

## rec_yds (n=6375)

| Metric | Value |
|---|---|
| NLL (model / baseline) | 0.8417 / 0.9088 |
| MAE | 13.40 yds |
| RMSE | 22.16 yds |
| Brier (own median line, own p_over) | 0.2586 |
| Calibration: bottom decile / top decile | 9.3% / 11.1% (target ~10% each) |

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

## Reproducing this baseline

```bash
cd nfl-props
.venv\Scripts\python -m nfl_props backtest --test-seasons 1
```

(The MAE/RMSE/Brier figures above required a small one-off script beyond what `backtest`
prints by default; `backtest.walk_forward()` + `backtest.summarize()` +
`backtest.calibration()` are the only building blocks used — no code changes.)
