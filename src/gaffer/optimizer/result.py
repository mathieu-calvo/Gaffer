"""Result containers for the squad optimiser."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameweekPlan:
    """Optimiser output for one gameweek of the planning horizon."""

    gameweek: int
    squad_ids: list[int]
    xi_ids: list[int]
    bench_ids: list[int]  # ordered: [bench GK, outfield sub 1, 2, 3]
    captain_id: int
    vice_captain_id: int
    transfers_in: list[int]
    transfers_out: list[int]
    hit_cost: int
    expected_points: float  # net of hit cost; includes captain doubling + bench weight


@dataclass(frozen=True)
class OptimizerResult:
    """Full multi-gameweek optimisation result."""

    plans: list[GameweekPlan]
    total_expected_points: float
    solver_status: str
    objective_value: float
