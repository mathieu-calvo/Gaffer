"""Orchestrates MILP optimisation against a projection DataFrame.

Takes raw predictions and player metadata, normalises the shape expected by the
solver, and returns an :class:`OptimizerResult`.
"""

from __future__ import annotations

import pandas as pd

from gaffer.config import settings
from gaffer.optimizer.milp import OptimizerInputs, solve
from gaffer.optimizer.result import OptimizerResult


def optimize_squad(
    projections: pd.DataFrame,
    players: pd.DataFrame,
    start_gw: int,
    horizon: int | None = None,
    initial_squad_ids: list[int] | None = None,
    bank: float = 0.0,
    free_transfers: int = 1,
    bench_weight: float = 0.1,
    time_limit: int | None = None,
) -> OptimizerResult:
    """Run the MILP on a projection frame and return the multi-GW plan.

    Args:
        projections: MultiIndex (player_id, gameweek) with an `expected_points` column.
        players: Index=player_id, columns `name, team, position, price`.
        start_gw: First gameweek to plan from.
        horizon: Number of gameweeks; defaults to `settings.horizon`.
        initial_squad_ids: Your current 15-man squad (for transfer planning).
            Omit for a fresh / wildcard selection.
        bank: Money in the bank (£m).
        free_transfers: Free transfers available in the first gameweek.
        bench_weight: Weight applied to bench expected points in the objective.
        time_limit: CBC time limit in seconds; defaults to `settings.solver_time_limit`.
    """
    inputs = OptimizerInputs(
        projections=projections,
        players=players,
        start_gw=start_gw,
        horizon=horizon if horizon is not None else settings.horizon,
        initial_squad_ids=initial_squad_ids,
        bank=bank,
        free_transfers=free_transfers,
        bench_weight=bench_weight,
        time_limit=time_limit if time_limit is not None else settings.solver_time_limit,
    )
    return solve(inputs)
