"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gaffer.domain.enums import Position
from gaffer.domain.player import Player


def _player(
    pid: int,
    position: Position,
    team: str = "ARS",
    price: float = 5.0,
    name: str | None = None,
) -> Player:
    return Player(
        id=pid,
        name=name or f"P{pid}",
        team=team,
        position=position,
        price=price,
    )


@pytest.fixture
def player_factory():
    """Returns a callable that builds Player instances with sensible defaults."""
    return _player


@pytest.fixture
def valid_squad_players() -> list[Player]:
    """Fifteen players forming a legal 2/5/5/3 squad under the £100m cap.

    Distributed across 6 clubs so the max-3-per-club rule isn't tripped.
    """
    teams = ["ARS", "MCI", "LIV", "CHE", "TOT", "NEW"]
    players: list[Player] = []
    pid = 1
    quota = {Position.GKP: 2, Position.DEF: 5, Position.MID: 5, Position.FWD: 3}
    for pos, count in quota.items():
        for _ in range(count):
            players.append(
                _player(
                    pid=pid,
                    position=pos,
                    team=teams[(pid - 1) % len(teams)],
                    price=5.0,
                )
            )
            pid += 1
    return players


@pytest.fixture
def toy_player_pool() -> pd.DataFrame:
    """40-player pool indexed by id for use in MILP tests.

    Position breakdown: 4 GKP / 12 DEF / 14 MID / 10 FWD across 6 clubs,
    with a small price spread so the optimiser has real choices to make.
    """
    teams = ["ARS", "MCI", "LIV", "CHE", "TOT", "NEW"]
    rows: list[dict[str, object]] = []
    pid = 1
    breakdown = {Position.GKP: 4, Position.DEF: 12, Position.MID: 14, Position.FWD: 10}
    for pos, count in breakdown.items():
        for i in range(count):
            rows.append(
                {
                    "id": pid,
                    "name": f"{pos.value}{i + 1}",
                    "team": teams[pid % len(teams)],
                    "position": pos.value,
                    "price": 4.0 + (i % 5),  # 4.0, 5.0, 6.0, 7.0, 8.0
                }
            )
            pid += 1
    return pd.DataFrame(rows).set_index("id")


@pytest.fixture
def toy_projections(toy_player_pool: pd.DataFrame) -> pd.DataFrame:
    """Projections for `toy_player_pool` over 3 gameweeks (1, 2, 3).

    Per-position mean expected points (FWD high, MID mid, DEF low, GKP lowest)
    plus a deterministic per-player shift so the optimiser has clear preferences.
    """
    base = {
        Position.GKP.value: 3.5,
        Position.DEF.value: 4.0,
        Position.MID.value: 5.0,
        Position.FWD.value: 5.5,
    }
    rows: list[dict[str, object]] = []
    for pid, row in toy_player_pool.iterrows():
        for gw in (1, 2, 3):
            ep = base[row["position"]] + (pid % 7) * 0.3
            rows.append(
                {
                    "player_id": int(pid),
                    "gameweek": gw,
                    "expected_points": float(ep),
                }
            )
    return pd.DataFrame(rows).set_index(["player_id", "gameweek"])


@pytest.fixture
def synthetic_gwdf() -> pd.DataFrame:
    """Tiny historical-style frame for feature-engineering tests.

    Two players, two seasons, four matches each. Columns mirror the inputs the
    `features.engineering` module expects to find.
    """
    rng = np.random.default_rng(0)
    rows: list[dict[str, object]] = []
    for season in ("2022-23", "2023-24"):
        for player_idx, name in enumerate(["Smith", "Jones"]):
            for gw in range(1, 5):
                was_home = bool(gw % 2)
                rows.append(
                    {
                        "name": name,
                        "season": season,
                        "kickoff_time": f"2023-08-{10 + gw:02d}T15:00:00Z",
                        "team": "ARS" if player_idx == 0 else "MCI",
                        "opponent_team": "LIV",
                        "was_home": was_home,
                        "team_h_score": int(rng.integers(0, 4)),
                        "team_a_score": int(rng.integers(0, 4)),
                        "team_h_difficulty": 3,
                        "team_a_difficulty": 2,
                        "team_h": "ARS",
                        "team_a": "LIV",
                        "position": "MID",
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
                )
    return pd.DataFrame(rows)
