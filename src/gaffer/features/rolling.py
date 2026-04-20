"""Exponential-moving-average rolling features with T+1 shift.

Ported from the rolling block inside `FantasyApiModel.prepare_train_test_sets`.
Keeps two artefacts:
- `roll_df`: shifted rolling averages — features known *before* each gameweek.
- `last_game_stats_df`: each player's most recent rolling snapshot, for test-set
  inference against upcoming fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gaffer.features.constants import ROLLING_FEATURES


@dataclass(frozen=True)
class RollingArtefacts:
    """Outputs of :func:`compute_rolling`."""

    rolling: pd.DataFrame
    last_game: pd.DataFrame


def compute_rolling(
    gwdf: pd.DataFrame,
    alpha: float = 0.95,
    min_periods: int = 5,
) -> RollingArtefacts:
    """Compute per-player EWMA rolling averages, shifted by one gameweek.

    Args:
        gwdf: Feature-engineered frame indexed by `kickoff_date`, with `name` column.
        alpha: EWMA smoothing factor. 1.0 uses only the latest observation; values
            closer to 0 weight older observations more heavily.
        min_periods: Minimum number of games before emitting a non-null row.

    Returns:
        RollingArtefacts with:
        - `rolling`: MultiIndex (name, kickoff_date), one row per player-GW, rolling
          averages of `ROLLING_FEATURES` from games *prior to* that GW.
        - `last_game`: one row per player holding the latest rolling snapshot,
          used as the feature vector for upcoming fixtures.
    """
    roll = (
        gwdf.groupby("name")[ROLLING_FEATURES]
        .ewm(alpha=alpha, min_periods=min_periods)
        .mean()
        .dropna(how="all")
    )
    last_game = roll.groupby("name").last()
    shifted = roll.groupby("name").shift(1).reset_index()
    shifted = shifted.set_index(["name", "kickoff_date"])
    return RollingArtefacts(rolling=shifted, last_game=last_game)
