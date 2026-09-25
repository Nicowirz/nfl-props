import numpy as np
import pandas as pd

from nfl_props import composition_backtest, model


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _synthetic_multi_week(n_players=10, n_teams=4, weeks=20, targets_per_player_per_week=4, seed=0):
    """Builds BOTH a targets-shaped DataFrame (one row per target attempt, with
    game_id/complete/air_yards/receiving_yards) and a stats-shaped DataFrame (one row
    per player-game, with targets/receiving_yards), derived from the SAME underlying
    per-target draws so the two tables are internally consistent -- exactly matching
    how real nflverse data relates data.load_targets()'s output to
    data.load_player_stats()'s output. A real opponent-defense effect is baked into both
    the catch-rate and yards-per-catch generating process, so a baseline that ignores the
    opponent should score worse than the fitted composition (matching every sibling
    backtest test file's "model beats naive baseline on synthetic signal" convention).
    """
    rng = np.random.default_rng(seed)
    players = [f"P{i}" for i in range(n_players)]
    teams = [f"T{i}" for i in range(n_teams)]
    catch_ability = dict(zip(players, rng.normal(0, 0.3, n_players)))
    yards_ability = dict(zip(players, rng.normal(0, 0.3, n_players)))
    defense = dict(zip(teams, rng.normal(0, 0.2, n_teams)))
    catch_intercept, catch_home_field = 0.4, 0.05
    yards_intercept, yards_home_field, yards_sigma = 4.6, 0.05, 0.35
    player_team = {p: teams[i % n_teams] for i, p in enumerate(players)}
    target_rows = []
    game_counter = 0
    d = pd.Timestamp("2024-09-01")
    for week in range(1, weeks + 1):
        for p in players:
            team = player_team[p]
            opp = rng.choice([t for t in teams if t != team])
            home = bool(rng.integers(0, 2))
            game_id = f"g{game_counter}"
            game_counter += 1
            n_targets = max(1, int(rng.poisson(targets_per_player_per_week)))
            for _ in range(n_targets):
                air_yards = float(rng.normal(9.0, 6.0))
                catch_eta = (catch_intercept + catch_ability[p] + defense[opp] - 0.08 * air_yards
                            + (catch_home_field if home else 0.0))
                complete = float(rng.uniform() < _sigmoid(catch_eta))
                receiving_yards = 0.0
                if complete:
                    mu = (yards_intercept + yards_ability[p] + defense[opp]
                         + (yards_home_field if home else 0.0))
                    receiving_yards = max(0.0, np.exp(rng.normal(mu, yards_sigma)) - model.OFFSET)
                target_rows.append({
                    "player_id": p, "player_name": p, "position_group": "WR", "team": team,
                    "opponent_team": opp, "season": 2024, "week": week, "date": d, "home": home,
                    "game_id": game_id, "complete": complete, "air_yards": air_yards,
                    "receiving_yards": receiving_yards,
                })
        d += pd.Timedelta(days=7)
    targets_df = pd.DataFrame(target_rows)

    stats_df = (targets_df.groupby(
        ["player_id", "player_name", "position_group", "team", "opponent_team",
         "season", "week", "date", "home", "game_id"], as_index=False)
        .agg(targets=("complete", "size"), receiving_yards=("receiving_yards", "sum")))
    for col in ("passing_yards", "attempts", "rushing_yards", "carries"):
        stats_df[col] = 0.0
    # .groupby(...).agg(...) sorts by the group keys, not chronologically -- re-sort
    # back to chronological row order (a real bug caught by a probe run before this
    # plan was written: without this, `stats_df["date"].iloc[len(stats_df) // 2]` below
    # does not pick the true chronological midpoint).
    stats_df = stats_df.sort_values("date", kind="stable").reset_index(drop=True)
    return targets_df, stats_df


def test_walk_forward_returns_one_row_per_relevant_game_after_start():
    targets_df, stats_df = _synthetic_multi_week()
    start = stats_df["date"].iloc[len(stats_df) // 2].date()
    preds = composition_backtest.walk_forward(
        stats_df, targets_df, start, halflife_days=100_000, reg=0.05,
        min_games=50, min_targets=50, min_catches=50, n_draws=2_000, seed=7)
    expected_n = int((stats_df["date"] >= pd.Timestamp(start)).sum())
    assert len(preds) == expected_n
    assert set(preds.columns) >= {"date", "player_id", "position_group",
                                  "y", "model_mu", "model_sigma", "base_mu", "base_sigma"}


def test_summarize_model_beats_naive_baseline():
    targets_df, stats_df = _synthetic_multi_week()
    start = stats_df["date"].iloc[len(stats_df) // 2].date()
    preds = composition_backtest.walk_forward(
        stats_df, targets_df, start, halflife_days=100_000, reg=0.05,
        min_games=50, min_targets=50, min_catches=50, n_draws=2_000, seed=7)
    s = composition_backtest.summarize(preds)
    assert s["n"] == len(preds)
    assert s["nll_model"] < s["nll_baseline"]


def test_calibration_bins_cover_all_predictions():
    targets_df, stats_df = _synthetic_multi_week()
    start = stats_df["date"].iloc[len(stats_df) // 2].date()
    preds = composition_backtest.walk_forward(
        stats_df, targets_df, start, halflife_days=100_000, reg=0.05,
        min_games=50, min_targets=50, min_catches=50, n_draws=2_000, seed=7)
    table = composition_backtest.calibration(preds)
    assert table["n"].sum() == len(preds)


def test_walk_forward_skips_rows_with_no_trailing_air_yards_history():
    """A player-game with literally zero targets recorded in that specific game has no
    matching row in the targets table, so the left-merge against game_ay produces NaN
    for trailing_air_yards (add_trailing_air_yards itself never returns NaN on its own
    output -- see its docstring -- this is purely a join-miss) -- there is no safe
    fallback for aDOT (see composition.py's module docstring), so walk_forward must skip
    scoring that row rather than erroring or silently using a wrong fallback. This uses
    NEWP, a player present in stats_df (with targets=1 for that game) but with NO
    corresponding row in targets_df for that game_id, so game_ay has no entry for NEWP
    and the left merge produces trailing_air_yards == NaN for that row -- not a
    new-player/no-prior-history scenario.
    """
    targets_df, stats_df = _synthetic_multi_week(weeks=20)
    # Use a start date that's a few games in to allow fitting history, while keeping NEWP in-window with no trailing history
    start = stats_df["date"].iloc[50].date()
    new_player_game_id = "g_new_player_0"
    # Add NEWP to stats_df but NOT to targets_df, so the merge produces NaN trailing_air_yards
    # (no target history for NEWP means game_ay won't have NEWP, leading to NaN in the left merge)
    new_row_stats = pd.DataFrame([{
        "player_id": "NEWP", "player_name": "NEWP", "position_group": "WR", "team": "T0",
        "opponent_team": "T1", "season": 2024, "week": 6,
        "date": stats_df["date"].iloc[50], "home": True, "game_id": new_player_game_id,
        "targets": 1, "receiving_yards": 15.0,
        "passing_yards": 0.0, "attempts": 0.0, "rushing_yards": 0.0, "carries": 0.0,
    }])
    stats_df = pd.concat([new_row_stats, stats_df], ignore_index=True).sort_values(
        "date", kind="stable").reset_index(drop=True)

    preds = composition_backtest.walk_forward(
        stats_df, targets_df, start, halflife_days=100_000, reg=0.05,
        min_games=45, min_targets=45, min_catches=45, n_draws=2_000, seed=7)

    assert "NEWP" not in preds["player_id"].values
    expected_n = int((stats_df["date"] >= pd.Timestamp(start)).sum()) - 1
    assert len(preds) == expected_n


def test_moment_match_lognormal_on_zero_inflated_sample_does_not_collapse_sigma():
    """The real committed MC sample for many player-games (especially QBs, whose true
    receiving usage is near-zero) is heavily zero-inflated: any draw with zero catches
    gives total_yards == 0.0 exactly (a whole-branch review of this module found some
    real rows with >=99% of draws exactly zero). `_moment_match_lognormal` fits ONE
    smooth log-normal to that mixture by taking the sample mean/std of
    log(mc_sample + OFFSET) -- this must not silently collapse `sigma_hat` down near the
    `EPS` floor (which would make the fitted distribution absurdly overconfident and is
    exactly the failure mode a whole-branch review traced as the proximate cause of
    several catastrophic single-row NLL blowups).

    This constructs a synthetic MC-sample-like array that is 80% exact zeros and 20%
    drawn from a real log-normal-shaped nonzero-yardage generating process (mirroring
    simulate_rec_yds()'s own np.maximum(0.0, np.exp(log_yards) - OFFSET) idiom), then
    asserts the actual (mu_hat, sigma_hat) `_moment_match_lognormal` returns for it,
    computed by hand from the same real run below rather than a placeholder:

        rng = np.random.default_rng(42); n=100_000; 80% exact zeros, 20% drawn from
        max(0, exp(Normal(mu_ln=3.5, sigma_ln=0.6)) - OFFSET)
        -> mu_hat = 2.543718879029954, sigma_hat = 0.5498275870470118

    (verified reproducible across repeated runs with the same seed). sigma_hat sits
    close to the nonzero-generating process's own sigma_ln=0.6 and two orders of
    magnitude above EPS=1e-6 -- a real, un-collapsed fit for THIS specific zero
    fraction/generating process. If a future change to `_moment_match_lognormal`
    collapses sigma_hat toward EPS (or inflates it well past the nonzero generator's own
    sigma_ln) for this exact synthetic input, this test fails.
    """
    rng = np.random.default_rng(42)
    n = 100_000
    frac_zero = 0.8
    n_zero = int(n * frac_zero)
    n_nonzero = n - n_zero
    mu_ln, sigma_ln = 3.5, 0.6
    nonzero_draws = np.maximum(0.0, np.exp(rng.normal(mu_ln, sigma_ln, n_nonzero)) - model.OFFSET)
    sample = np.concatenate([np.zeros(n_zero), nonzero_draws])
    rng.shuffle(sample)

    mu_hat, sigma_hat = composition_backtest._moment_match_lognormal(sample)

    assert abs(mu_hat - 2.543718879029954) < 0.05
    assert abs(sigma_hat - 0.5498275870470118) < 0.05
    # The real regression this guards against: sigma_hat silently collapsing to (or near)
    # the EPS floor despite a meaningfully-varying nonzero component in the sample.
    assert sigma_hat > 1_000 * composition_backtest.EPS
    # sigma_hat should stay in the real, computed neighborhood of the nonzero
    # generating process's own sigma_ln=0.6 for this specific 80/20 zero/nonzero split
    # -- not collapsed near zero, and not blown up far past it either.
    assert 0.3 < sigma_hat < 0.6
