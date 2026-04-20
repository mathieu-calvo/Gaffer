"""Row-level feature engineering transforms.

Ported from `FantasyApiModel._reformat_*` methods. These are pure dataframe
transforms — no I/O, no globals — applied to the combined historical + current
per-gameweek frame before rolling aggregation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gaffer.features.constants import (
    FIXTURE_FEATURES,
    FPL_FEATURES,
    PLAYER_ID_COL,
    PLAYER_METADATA_FEATURES,
    PLAYER_STATS_FEATURES,
    TARGET_FEATURES,
    TEAM_METADATA_FEATURES,
    TEAM_STATS_FEATURES,
)


def reformat_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert `kickoff_time` (ISO string) to `kickoff_date` (datetime, date-only)."""
    df = df.copy()
    df["kickoff_date"] = pd.to_datetime(df["kickoff_time"].str[:10])
    return df.drop(columns=["kickoff_time"])


def reformat_team_form(df: pd.DataFrame) -> pd.DataFrame:
    """Derive `team_points` (3/1/0), `team_goals_scored`, `team_goals_conceded`.

    Requires `was_home`, `team_h_score`, `team_a_score`. Drops the raw score columns.
    """
    df = df.copy()
    home_win = df["was_home"] & (df["team_h_score"] > df["team_a_score"])
    away_win = (~df["was_home"]) & (df["team_h_score"] < df["team_a_score"])
    draw = df["team_h_score"] == df["team_a_score"]
    home_loss = df["was_home"] & (df["team_h_score"] < df["team_a_score"])
    away_loss = (~df["was_home"]) & (df["team_h_score"] > df["team_a_score"])

    df["team_points"] = np.select(
        [home_win | away_win, draw, home_loss | away_loss], [3.0, 1.0, 0.0], default=np.nan
    )
    df["team_goals_scored"] = np.where(df["was_home"], df["team_h_score"], df["team_a_score"])
    df["team_goals_conceded"] = np.where(df["was_home"], df["team_a_score"], df["team_h_score"])
    return df.drop(columns=["team_a_score", "team_h_score"])


def reformat_fdr(df: pd.DataFrame) -> pd.DataFrame:
    """Convert home/away difficulty pair into `team_fdr` / `opponent_team_fdr`."""
    df = df.copy()
    df["team_fdr"] = np.where(df["was_home"], df["team_a_difficulty"], df["team_h_difficulty"])
    df["opponent_team_fdr"] = np.where(
        df["was_home"], df["team_h_difficulty"], df["team_a_difficulty"]
    )
    drop = [c for c in ["team_a", "team_h", "team_a_difficulty", "team_h_difficulty"] if c in df]
    return df.drop(columns=drop)


def reformat_fpl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize `transfers_balance` and `selected` to fractions of total managers.

    Drops the raw counts. Requires `nb_managers` column.
    """
    df = df.copy()
    df["transfers_balance_pct"] = df["transfers_balance"] / df["nb_managers"]
    df["selected_pct"] = df["selected"] / df["nb_managers"]
    return df.drop(
        columns=[
            c
            for c in [
                "transfers_balance",
                "transfers_in",
                "transfers_out",
                "selected",
                "nb_managers",
            ]
            if c in df.columns
        ]
    )


_ORDERED_COLUMNS = (
    FIXTURE_FEATURES
    + TEAM_METADATA_FEATURES
    + TEAM_STATS_FEATURES
    + PLAYER_ID_COL
    + PLAYER_METADATA_FEATURES
    + PLAYER_STATS_FEATURES
    + FPL_FEATURES
    + TARGET_FEATURES
)


def feature_engineer(gwdf: pd.DataFrame) -> pd.DataFrame:
    """Full per-row feature engineering pipeline.

    Tolerates both raw FPL-API rows (with `kickoff_time`, `team_h_difficulty`)
    and the bundled historical CSV (already cooked into `kickoff_date`,
    `team_fdr`). Applies FPL-feature normalization and team-form derivation,
    reorders to the canonical column layout, fixes the `GK` → `GKP` position
    inconsistency, and indexes by `kickoff_date` sorted per player.
    """
    if "kickoff_time" in gwdf.columns and "kickoff_date" not in gwdf.columns:
        gwdf = reformat_dates(gwdf)
    if "team_h_difficulty" in gwdf.columns and "team_fdr" not in gwdf.columns:
        gwdf = reformat_fdr(gwdf)
    gwdf = reformat_fpl_features(gwdf)
    gwdf = reformat_team_form(gwdf)
    gwdf = gwdf.loc[:, _ORDERED_COLUMNS]
    gwdf["position"] = gwdf["position"].replace("GK", "GKP")
    gwdf = gwdf.sort_values(["name", "season", "kickoff_date"])
    return gwdf.set_index("kickoff_date")
