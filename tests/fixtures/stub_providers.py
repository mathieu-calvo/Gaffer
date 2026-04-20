"""In-memory stand-ins for the FPL data providers used in integration tests.

Lets the end-to-end flow run without network access or the bundled 31MB CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gaffer.providers.base import BootstrapData


def _player_history_row(
    *,
    name: str,
    team: str,
    position: str,
    season: str,
    gw: int,
    was_home: bool,
    rng: np.random.Generator,
) -> dict[str, object]:
    season_year = int(season.split("-")[0])
    return {
        "name": name,
        "season": season,
        "kickoff_time": f"{season_year}-08-{10 + gw:02d}T15:00:00Z",
        "team": team,
        "opponent_team": "OPP",
        "was_home": was_home,
        "team_h_score": int(rng.integers(0, 4)),
        "team_a_score": int(rng.integers(0, 4)),
        "team_h_difficulty": int(rng.integers(2, 5)),
        "team_a_difficulty": int(rng.integers(2, 5)),
        "team_h": team if was_home else "OPP",
        "team_a": "OPP" if was_home else team,
        "position": position,
        "minutes": 90,
        "yellow_cards": 0,
        "red_cards": 0,
        "assists": int(rng.integers(0, 2)),
        "goals_scored": int(rng.integers(0, 2)),
        "penalties_missed": 0,
        "goals_conceded": int(rng.integers(0, 3)),
        "clean_sheets": 0,
        "own_goals": 0,
        "saves": 0,
        "penalties_saved": 0,
        "creativity": float(rng.uniform(0, 50)),
        "ict_index": float(rng.uniform(0, 30)),
        "influence": float(rng.uniform(0, 50)),
        "threat": float(rng.uniform(0, 60)),
        "value": 70,
        "selected": int(rng.integers(1000, 100_000)),
        "transfers_balance": int(rng.integers(-5000, 5000)),
        "transfers_in": 0,
        "transfers_out": 0,
        "nb_managers": 1_000_000,
        "total_points": int(rng.integers(0, 12)),
        "bonus": 0,
        "bps": int(rng.integers(0, 50)),
    }


@dataclass
class StubFplProvider:
    """Synthetic in-memory provider implementing the FplDataProvider Protocol."""

    name: str = "stub"
    season: str = "2024-25"
    current_gw: int = 5

    def get_bootstrap(self) -> BootstrapData:
        elements_df = pd.DataFrame(
            [
                {"id": 1, "first_name": "Alice", "second_name": "GK1",
                 "team": 1, "element_type": 1, "now_cost": 50},
                {"id": 2, "first_name": "Bob", "second_name": "DEF1",
                 "team": 2, "element_type": 2, "now_cost": 55},
            ]
        )
        elements_df["name"] = elements_df["first_name"] + " " + elements_df["second_name"]
        return BootstrapData(
            id_to_team_name={1: "ARS", 2: "MCI"},
            id_to_team_short={1: "ARS", 2: "MCI"},
            id_to_position_short={1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"},
            id_to_player_name={1: "Alice GK1", 2: "Bob DEF1"},
            id_to_player_position={1: 1, 2: 2},
            player_id_to_team_name={1: "ARS", 2: "MCI"},
            total_fpl_managers=1_000_000,
            elements_df=elements_df,
        )

    def get_fixtures(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"id": 1, "event": 4, "kickoff_time": "2024-08-15T15:00:00Z",
             "team_h": "ARS", "team_a": "MCI", "team_h_difficulty": 3,
             "team_a_difficulty": 4, "team_h_score": 1, "team_a_score": 0, "finished": True},
            {"id": 2, "event": 5, "kickoff_time": "2024-08-22T15:00:00Z",
             "team_h": "ARS", "team_a": "MCI", "team_h_difficulty": 3,
             "team_a_difficulty": 4, "team_h_score": None, "team_a_score": None, "finished": False},
        ])

    def get_player_histories(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        rows = [
            _player_history_row(
                name="Alice GK1", team="ARS", position="GKP",
                season=self.season, gw=g, was_home=bool(g % 2), rng=rng,
            )
            for g in range(1, 4)
        ]
        return pd.DataFrame(rows)

    def get_current_gw(self) -> int:
        return self.current_gw


@dataclass
class StubHistoricalProvider:
    """Synthetic historical provider — emits a tiny multi-season frame."""

    name: str = "stub-historical"
    n_players: int = 4
    n_seasons: int = 3

    def get_historical_gwdata(self) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        positions = ["GKP", "DEF", "MID", "FWD"]
        rows: list[dict[str, object]] = []
        for season_idx in range(self.n_seasons):
            season = f"20{20 + season_idx}-{21 + season_idx}"
            for pid in range(self.n_players):
                name = f"Player{pid}"
                pos = positions[pid % len(positions)]
                team = "ARS" if pid % 2 == 0 else "MCI"
                for gw in range(1, 8):
                    rows.append(
                        _player_history_row(
                            name=name, team=team, position=pos, season=season,
                            gw=gw, was_home=bool(gw % 2), rng=rng,
                        )
                    )
        return pd.DataFrame(rows)
