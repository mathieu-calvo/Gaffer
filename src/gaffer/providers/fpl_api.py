"""Live FPL API provider.

Wraps https://fantasy.premierleague.com/api/ endpoints. No authentication required.
Responses are cached through a two-tier (memory LRU + SQLite) cache keyed by URL.

Ported from `FantasyApiModel._get_static_mapping` and `get_latest_season_data` with
concerns split: raw HTTP + caching here; feature engineering lives in `features/`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import requests

from gaffer.cache.memory_cache import TTLMemoryCache
from gaffer.cache.sqlite_cache import SqliteCache
from gaffer.config import settings
from gaffer.providers.base import BootstrapData

ROOT_URL = "https://fantasy.premierleague.com/api/"


class LiveFplApiProvider:
    """Provider hitting the public FPL API, with LRU + SQLite caching."""

    name = "live"

    def __init__(self, season: str = "2024-25") -> None:
        self._season = season
        self._session = requests.Session()
        self._mem_cache = TTLMemoryCache(maxsize=64)
        self._disk_cache = SqliteCache(settings.cache_dir / "fpl_api.sqlite")

    @property
    def season(self) -> str:
        return self._season

    # --- HTTP with caching ----------------------------------------------------

    def _get_json(self, path: str, ttl_hours: int) -> Any:
        url = ROOT_URL + path
        cached = self._mem_cache.get(url) or self._disk_cache.get(url)
        if cached is not None:
            self._mem_cache.set(url, cached, ttl_hours * 3600)
            return cached
        resp = self._session.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        self._mem_cache.set(url, payload, ttl_hours * 3600)
        self._disk_cache.set(url, payload, ttl_hours * 3600)
        return payload

    # --- Provider interface ---------------------------------------------------

    def get_bootstrap(self) -> BootstrapData:
        data = self._get_json("bootstrap-static/", ttl_hours=settings.bootstrap_ttl_hours)
        teams_df = pd.DataFrame(data["teams"]).set_index("id")[["name", "short_name"]]
        positions = pd.DataFrame(data["element_types"]).set_index("id")["singular_name_short"]
        elements_df = pd.DataFrame(data["elements"])
        elements_df["name"] = elements_df["first_name"] + " " + elements_df["second_name"]

        id_to_team_name = teams_df["name"].to_dict()
        return BootstrapData(
            id_to_team_name=id_to_team_name,
            id_to_team_short=teams_df["short_name"].to_dict(),
            id_to_position_short=positions.to_dict(),
            id_to_player_name=elements_df.set_index("id")["name"].to_dict(),
            id_to_player_position=elements_df.set_index("id")["element_type"].to_dict(),
            player_id_to_team_name=elements_df.set_index("id")["team"].map(id_to_team_name).to_dict(),
            total_fpl_managers=int(data["total_players"]),
            elements_df=elements_df,
        )

    def get_fixtures(self) -> pd.DataFrame:
        data = self._get_json("fixtures/", ttl_hours=settings.fixtures_ttl_hours)
        fixtures = pd.DataFrame(data)
        bootstrap = self.get_bootstrap()
        fixtures["team_h"] = fixtures["team_h"].map(bootstrap.id_to_team_name)
        fixtures["team_a"] = fixtures["team_a"].map(bootstrap.id_to_team_name)
        return fixtures

    def get_current_gw(self) -> int:
        fixtures = self.get_fixtures()
        fixtures = fixtures.assign(
            gw_start=fixtures.groupby("event")["kickoff_time"].transform("min")
        )
        today = datetime.today().strftime("%Y-%m-%d")
        upcoming = fixtures[fixtures["gw_start"] > today]
        if upcoming.empty:
            return int(fixtures["event"].max())
        return int(upcoming["event"].min())

    def get_manager_entry(self, manager_id: int) -> dict[str, Any]:
        """Fetch public entry info for an FPL manager.

        Useful fields: `last_deadline_bank` (£m × 10), `current_event`, `name`,
        `player_first_name`, `player_last_name`. Raises if the id is unknown.
        """
        return self._get_json(f"entry/{int(manager_id)}/", ttl_hours=1)

    def get_manager_picks(self, manager_id: int, event: int) -> list[int]:
        """Return the 15 FPL player ids picked by `manager_id` for the given GW."""
        payload = self._get_json(
            f"entry/{int(manager_id)}/event/{int(event)}/picks/", ttl_hours=1
        )
        picks = payload.get("picks", [])
        return [int(p["element"]) for p in picks]

    def get_player_histories(self) -> pd.DataFrame:
        """Per-player GW history for the current season.

        Loops over every element id from bootstrap-static — ~600 HTTP calls. Each
        response is cached on disk, so subsequent runs within TTL are instant.
        """
        bootstrap = self.get_bootstrap()
        fixtures = self.get_fixtures()
        past_fixtures = fixtures[fixtures["finished"]][
            ["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
        ]

        frames: list[pd.DataFrame] = []
        for player_id in bootstrap.elements_df["id"].unique():
            payload = self._get_json(
                f"element-summary/{player_id}/", ttl_hours=settings.fixtures_ttl_hours
            )
            history = pd.DataFrame(payload.get("history", []))
            if history.empty:
                continue
            history["name"] = history["element"].map(bootstrap.id_to_player_name)
            history["team"] = history["element"].map(bootstrap.player_id_to_team_name)
            history["opponent_team"] = history["opponent_team"].map(bootstrap.id_to_team_name)
            history["position"] = (
                history["element"]
                .map(bootstrap.id_to_player_position)
                .map(bootstrap.id_to_position_short)
            )
            frames.append(history)

        if not frames:
            return pd.DataFrame()

        histories = pd.concat(frames, ignore_index=True)
        histories["season"] = self._season
        histories["nb_managers"] = bootstrap.total_fpl_managers
        histories = histories.merge(past_fixtures, left_on="fixture", right_on="id", how="left")
        histories = histories.drop(
            columns=[
                c
                for c in [
                    "element",
                    "id",
                    "round",
                    "fixture",
                    "expected_assists",
                    "expected_goal_involvements",
                    "expected_goals",
                    "expected_goals_conceded",
                    "starts",
                ]
                if c in histories.columns
            ]
        )
        return histories
