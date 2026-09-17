"""Same-game joint-outcome Monte Carlo simulation: draws correlated player-yardage
samples sharing one simulated game state, for computing JOINT (not marginal)
probabilities of same-game prop combinations.

model.py and game_model.py are unmodified and have zero knowledge this module exists --
callers fit both as usual and pass their already-validated outputs in here. No existing
marginal distribution changes; the shared shock only ever exists inside a simulation
draw, never written back to any Ratings object.

See docs/superpowers/specs/2026-09-17-nfl-props-joint-correlation-sim-design.md.
"""
from __future__ import annotations

import numpy as np

from .model import OFFSET

N_DRAWS = 10_000

# Calibrated via calibrate_joint_sensitivity() (Task 2) against real 3-season data --
# see that function and Tasks 4/5 of docs/superpowers/plans/2026-09-17-nfl-props-
# joint-correlation-sim.md for how these get populated. Starts at 0.0 (a true no-op) so
# this module's own tests are meaningful before calibration has run.
JOINT_SENSITIVITY: dict[str, float] = {"pass_yds": 0.0, "rush_yds": 0.0, "rec_yds": 0.0}


def simulate_player_yards(total_mu: float, total_sigma: float, league_avg_total: float,
                          players: list[tuple[float, float, str]],
                          sensitivity: dict[str, float] | None = None,
                          n: int = N_DRAWS, seed: int | None = None) -> np.ndarray:
    """Draw `n` joint Monte Carlo samples of every player's REAL yards in ONE game,
    correlated via one shared per-draw total deviation from a typical game.

    `players`: one (mu, sigma, stat) tuple per player being jointly priced -- mu/sigma
    are each player's OWN marginal from model.predicted_distribution(), completely
    unmodified; `stat` selects which sensitivity[stat] shock applies to that player for
    each draw. Returns shape (n, len(players)) of real yards (OFFSET already inverted,
    clamped non-negative), one row per draw, columns in `players`' order.
    """
    s = JOINT_SENSITIVITY if sensitivity is None else sensitivity
    rng = np.random.default_rng(seed)
    total_draws = rng.normal(total_mu, total_sigma, size=n)
    # A drawn total at or below 0 is a real (if rare) possibility under a Normal total
    # distribution; floor at 1.0 before the log so total_dev stays finite, mirroring
    # _safe_log_yards's own floor-guard philosophy in model.py.
    total_dev = np.log(np.maximum(total_draws, 1.0) / league_avg_total)
    out = np.zeros((n, len(players)))
    for i, (mu, sigma, stat) in enumerate(players):
        shocked_mu = mu + s[stat] * total_dev
        log_yards = rng.normal(shocked_mu, sigma)
        out[:, i] = np.maximum(0.0, np.exp(log_yards) - OFFSET)
    return out


def joint_prob(samples: np.ndarray, conditions: list[tuple[int, str, float]]) -> float:
    """Empirical P(every condition holds) across `samples` (n_draws x n_players).
    `conditions`: list of (player_column_index, 'over'|'under', line).
    """
    mask = np.ones(samples.shape[0], dtype=bool)
    for idx, selection, line in conditions:
        mask &= (samples[:, idx] > line) if selection == "over" else (samples[:, idx] < line)
    return float(mask.mean())
