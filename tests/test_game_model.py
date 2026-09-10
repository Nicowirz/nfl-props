import numpy as np
import pandas as pd
import pytest

from nfl_props import game_model


def _synthetic_games(n_teams=16, rounds=4, seed=0, neutral_fraction=0.0):
    """Simulate completed games from a known power rating per team and a known home-field
    edge. home_score/away_score are constructed so their difference is EXACTLY the sampled
    margin (20 +/- margin/2 each) -- the model only ever looks at the difference, so the
    individual score values don't need to be game-realistic for this recovery test.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    power = dict(zip(teams, rng.normal(0, 7.0, n_teams)))
    home_field, sigma = 2.5, 13.5
    rows = []
    d = pd.Timestamp("2024-09-01")
    for _ in range(rounds):
        order = list(rng.permutation(teams))
        for i in range(0, n_teams - 1, 2):
            home, away = order[i], order[i + 1]
            neutral = bool(rng.random() < neutral_fraction)
            mu = (0.0 if neutral else home_field) + power[home] - power[away]
            margin = rng.normal(mu, sigma)
            rows.append({
                "game_id": f"g{len(rows)}", "season": 2024, "game_type": "REG", "week": 1,
                "gameday": d, "home_team": home, "away_team": away,
                "home_score": 20.0 + margin / 2, "away_score": 20.0 - margin / 2,
                "neutral": neutral,
            })
            d += pd.Timedelta(days=1)
    return pd.DataFrame(rows), teams, power, home_field, sigma


def test_fit_margin_recovers_power_and_home_field():
    df, teams, power, home_field, sigma = _synthetic_games(rounds=48, seed=0)
    r = game_model.fit_margin(df, reg=0.05, halflife_days=100_000, min_games=20)
    est_power = np.array([r.power[t] for t in teams])
    true_power = np.array([power[t] for t in teams])
    assert np.corrcoef(true_power, est_power)[0, 1] > 0.9
    assert abs(r.sigma - sigma) < 2.0


def test_fit_margin_respects_as_of():
    df, *_ = _synthetic_games(rounds=6)
    cutoff = df["gameday"].iloc[40].date()
    r = game_model.fit_margin(df, as_of=cutoff, min_games=20)
    assert r.n_games == (df["gameday"] < pd.Timestamp(cutoff)).sum()
    assert r.as_of == cutoff


def test_fit_margin_neutral_games_get_no_home_boost():
    df, teams, power, home_field, sigma = _synthetic_games(rounds=8, neutral_fraction=0.25, seed=1)
    r = game_model.fit_margin(df, reg=0.05, halflife_days=100_000, min_games=20)
    assert abs(r.home_field - home_field) < 1.5


def test_fit_margin_solves_with_zero_neutral_games():
    """With no neutral games in the fit window, the home_field column is identical to the
    intercept column (both are all-1s) -- HOME_FIELD_REG must keep the solve well-posed.
    """
    df, *_ = _synthetic_games(rounds=4, neutral_fraction=0.0, seed=2)
    r = game_model.fit_margin(df, reg=0.1, min_games=20)
    assert np.isfinite(r.home_field)
    assert all(np.isfinite(v) for v in r.power.values())


def test_predicted_margin_unknown_team_flagged():
    df, teams, *_ = _synthetic_games()
    r = game_model.fit_margin(df, reg=0.1, min_games=20)
    mu, sigma = game_model.predicted_margin(r, "NEWTEAM", teams[0])
    assert np.isfinite(mu) and sigma > 0
    assert "NEWTEAM" in r.prior_teams


def test_predicted_margin_neutral_excludes_home_field():
    df, teams, power, home_field, sigma = _synthetic_games(rounds=48, seed=0)
    r = game_model.fit_margin(df, reg=0.05, halflife_days=100_000, min_games=20)
    mu_home, _ = game_model.predicted_margin(r, teams[0], teams[1], neutral=False)
    mu_neutral, _ = game_model.predicted_margin(r, teams[0], teams[1], neutral=True)
    assert mu_home - mu_neutral == pytest.approx(r.home_field)


def test_table_sorted_by_power():
    df, *_ = _synthetic_games()
    r = game_model.fit_margin(df, reg=0.1, min_games=20)
    t = r.table()
    assert list(t["power"]) == sorted(t["power"], reverse=True)


def _synthetic_scores(n_teams=16, rounds=4, seed=0):
    """Simulate each team-side's own score from a known scoring_rate/allowed_rate per team."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    scoring = dict(zip(teams, rng.normal(0, 4.0, n_teams)))
    allowed = dict(zip(teams, rng.normal(0, 4.0, n_teams)))
    intercept, sigma = 22.0, 9.5
    rows = []
    d = pd.Timestamp("2024-09-01")
    for _ in range(rounds):
        order = list(rng.permutation(teams))
        for i in range(0, n_teams - 1, 2):
            home, away = order[i], order[i + 1]
            home_mu = intercept + scoring[home] + allowed[away]
            away_mu = intercept + scoring[away] + allowed[home]
            home_score = max(0.0, rng.normal(home_mu, sigma))
            away_score = max(0.0, rng.normal(away_mu, sigma))
            rows.append({
                "game_id": f"g{len(rows)}", "season": 2024, "game_type": "REG", "week": 1,
                "gameday": d, "home_team": home, "away_team": away,
                "home_score": home_score, "away_score": away_score, "neutral": False,
            })
            d += pd.Timedelta(days=1)
    return pd.DataFrame(rows), teams, scoring, allowed, intercept, sigma


def test_fit_score_recovers_scoring_and_allowed():
    df, teams, scoring, allowed, intercept, sigma = _synthetic_scores(rounds=60, seed=0)
    r = game_model.fit_score(df, reg=0.1, halflife_days=100_000, min_games=20)
    est_scoring = np.array([r.scoring[t] for t in teams])
    true_scoring = np.array([scoring[t] for t in teams])
    est_allowed = np.array([r.allowed[t] for t in teams])
    true_allowed = np.array([allowed[t] for t in teams])
    assert np.corrcoef(true_scoring, est_scoring)[0, 1] > 0.85
    assert np.corrcoef(true_allowed, est_allowed)[0, 1] > 0.85


def test_predicted_total_matches_sum_of_scores():
    df, teams, *_ = _synthetic_scores()
    r = game_model.fit_score(df, reg=0.1, min_games=20)
    home, away = teams[0], teams[1]
    home_mu, home_sigma = game_model.predicted_score(r, home, away)
    away_mu, away_sigma = game_model.predicted_score(r, away, home)
    total_mu, total_sigma = game_model.predicted_total(r, home, away)
    assert total_mu == pytest.approx(home_mu + away_mu)
    assert total_sigma == pytest.approx(np.sqrt(home_sigma ** 2 + away_sigma ** 2))


def test_predicted_score_unknown_team_flagged():
    df, teams, *_ = _synthetic_scores()
    r = game_model.fit_score(df, reg=0.1, min_games=20)
    mu, sigma = game_model.predicted_score(r, "NEWTEAM", teams[0])
    assert np.isfinite(mu) and sigma > 0
    assert "NEWTEAM" in r.prior_teams
