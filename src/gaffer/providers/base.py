"""Data provider protocols and shared types.

Two providers serve different roles:
- `FplDataProvider` — live FPL API: current season snapshot + upcoming fixtures.
- `HistoricalDataProvider` — multi-season training data from bundled CSV.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class BootstrapData:
    """Snapshot of FPL `bootstrap-static/` response, reshaped for downstream use."""

    id_to_team_name: dict[int, str]
    id_to_team_short: dict[int, str]
    id_to_position_short: dict[int, str]
    id_to_player_name: dict[int, str]
    id_to_player_position: dict[int, int]
    player_id_to_team_name: dict[int, str]
    total_fpl_managers: int
    elements_df: pd.DataFrame = field(repr=False)


@runtime_checkable
class FplDataProvider(Protocol):
    """Provider of live FPL data for the current season."""

    @property
    def name(self) -> str:
        """Provider identifier (e.g., 'live', 'stub')."""
        ...

    @property
    def season(self) -> str:
        """Season label, e.g., '2024-25'."""
        ...

    def get_bootstrap(self) -> BootstrapData:
        """Return static FPL metadata (teams, positions, players, manager count)."""
        ...

    def get_fixtures(self) -> pd.DataFrame:
        """Return all fixtures for the current season.

        Columns: `id, event, kickoff_time, team_h, team_a, team_h_difficulty,
        team_a_difficulty, team_h_score, team_a_score, finished`.
        """
        ...

    def get_player_histories(self) -> pd.DataFrame:
        """Return per-gameweek stats for every player this season.

        One row per (player, fixture-played). Columns mirror the FPL
        `element-summary/{id}/history` payload plus mapped `name`, `team`,
        `opponent_team`, `season`, `position`.
        """
        ...

    def get_current_gw(self) -> int:
        """Return the next upcoming gameweek event number."""
        ...


@runtime_checkable
class HistoricalDataProvider(Protocol):
    """Provider of historical multi-season training data."""

    @property
    def name(self) -> str:
        ...

    def get_historical_gwdata(self) -> pd.DataFrame:
        """Return the combined historical per-gameweek DataFrame (multiple seasons)."""
        ...
