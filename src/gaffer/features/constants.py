"""Column groupings used across feature engineering and modelling.

Ported from `FantasyApiModel.py` module-level constants, kept as lists (not tuples) so
they can be concatenated directly in DataFrame slicing.
"""

from __future__ import annotations

PLAYER_STATS_FEATURES: list[str] = [
    "minutes",
    "yellow_cards",
    "red_cards",
    "assists",
    "goals_scored",
    "penalties_missed",
    "goals_conceded",
    "clean_sheets",
    "own_goals",
    "saves",
    "penalties_saved",
    "creativity",
]

PLAYER_ID_COL: list[str] = ["name"]
PLAYER_METADATA_FEATURES: list[str] = ["position"]

TEAM_STATS_FEATURES: list[str] = ["team_points", "team_goals_scored", "team_goals_conceded"]
TEAM_METADATA_FEATURES: list[str] = ["team", "opponent_team", "team_fdr", "opponent_team_fdr"]

FIXTURE_FEATURES: list[str] = ["season", "kickoff_date", "was_home"]

FPL_FEATURES: list[str] = [
    "ict_index",
    "influence",
    "threat",
    "value",
    "selected_pct",
    "transfers_balance_pct",
]

TARGET_FEATURES: list[str] = ["total_points", "bonus", "bps"]

ROLLING_FEATURES: list[str] = (
    PLAYER_STATS_FEATURES + TARGET_FEATURES + FPL_FEATURES + TEAM_STATS_FEATURES
)
