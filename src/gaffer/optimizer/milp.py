"""Multi-gameweek MILP for FPL squad selection.

Built on PuLP + CBC (bundled, free, runs on Streamlit Cloud). Given per-player
per-gameweek expected-points projections, the solver picks:

- a 15-man squad for each gameweek that satisfies the position quotas (2/5/5/3),
  £100m + bank budget, and max-3-per-club rule;
- a starting XI in a legal FPL formation, plus a captain (2× points);
- transfers moving one GW's squad into the next, charged at 4 points per
  transfer above the free-transfer allowance.

Objective maximised:
    Σ_gw Σ_p  E[pts(p, gw)] · (starting + captain + bench_weight · bench)
          − 4 · max(0, extra_transfers_gw − free_transfers_gw)

Vice-captain is picked post-hoc (second-best starter by projected points) since
its expected value is marginal and including it bloats the model without
meaningfully changing squad choices.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

from gaffer.domain.constraints import FPL_RULES
from gaffer.domain.enums import Position
from gaffer.optimizer.result import GameweekPlan, OptimizerResult


@dataclass(frozen=True)
class OptimizerInputs:
    """Structured inputs to :func:`solve`.

    `projections` must have a MultiIndex of (player_id, gameweek) and at minimum
    an `expected_points` column. `players` is indexed by player_id with columns
    `name`, `team`, `position` (str matching Position enum), and `price`.
    """

    projections: pd.DataFrame
    players: pd.DataFrame
    start_gw: int
    horizon: int = 3
    initial_squad_ids: list[int] | None = None
    bank: float = 0.0
    free_transfers: int = 1
    bench_weight: float = 0.1
    max_transfers_per_gw: int = 15  # effectively unbounded; 15 = full wildcard
    time_limit: int = 60


def _build_projection_matrix(
    projections: pd.DataFrame, player_ids: list[int], gws: list[int]
) -> pd.DataFrame:
    """Reshape projections to player × gameweek expected-points matrix."""
    ep = (
        projections["expected_points"]
        .unstack(level="gameweek")
        .reindex(index=player_ids, columns=gws)
    )
    return ep.fillna(0.0)


def _add_squad_constraints(
    prob: pulp.LpProblem,
    squad: dict[tuple[int, int], pulp.LpVariable],
    starting: dict[tuple[int, int], pulp.LpVariable],
    captain: dict[tuple[int, int], pulp.LpVariable],
    players: pd.DataFrame,
    gws: list[int],
    budget_cap: float,
) -> None:
    """All rules that hold within each gameweek independently."""
    rules = FPL_RULES
    by_position = {pos: players[players["position"] == pos.value].index for pos in Position}
    by_club = {club: players[players["team"] == club].index for club in players["team"].unique()}
    player_ids = players.index.tolist()

    for gw in gws:
        # Squad size + position quotas
        prob += (
            pulp.lpSum(squad[(p, gw)] for p in player_ids) == rules.squad_size,
            f"squad_size_{gw}",
        )
        for pos, quota in rules.squad_quota.items():
            prob += (
                pulp.lpSum(squad[(p, gw)] for p in by_position[pos]) == quota,
                f"squad_{pos.value}_{gw}",
            )

        # XI size + formation bounds
        prob += (
            pulp.lpSum(starting[(p, gw)] for p in player_ids) == rules.xi_size,
            f"xi_size_{gw}",
        )
        for pos in Position:
            ids_pos = by_position[pos]
            prob += (
                pulp.lpSum(starting[(p, gw)] for p in ids_pos) >= rules.formation_min[pos],
                f"xi_min_{pos.value}_{gw}",
            )
            prob += (
                pulp.lpSum(starting[(p, gw)] for p in ids_pos) <= rules.formation_max[pos],
                f"xi_max_{pos.value}_{gw}",
            )

        # Captain: exactly one, must be in the XI
        prob += (
            pulp.lpSum(captain[(p, gw)] for p in player_ids) == 1,
            f"captain_count_{gw}",
        )
        for p in player_ids:
            prob += (
                starting[(p, gw)] <= squad[(p, gw)],
                f"start_subset_squad_{p}_{gw}",
            )
            prob += (
                captain[(p, gw)] <= starting[(p, gw)],
                f"captain_starts_{p}_{gw}",
            )

        # Club cap
        for club, club_ids in by_club.items():
            prob += (
                pulp.lpSum(squad[(p, gw)] for p in club_ids) <= rules.max_per_club,
                f"club_cap_{club}_{gw}",
            )

        # Budget
        prob += (
            pulp.lpSum(players.loc[p, "price"] * squad[(p, gw)] for p in player_ids)
            <= budget_cap,
            f"budget_{gw}",
        )


def _add_transfer_constraints(
    prob: pulp.LpProblem,
    squad: dict[tuple[int, int], pulp.LpVariable],
    xfers_in: dict[tuple[int, int], pulp.LpVariable],
    xfers_out: dict[tuple[int, int], pulp.LpVariable],
    extra_hits: dict[int, pulp.LpVariable],
    player_ids: list[int],
    gws: list[int],
    initial_squad_ids: list[int],
    free_transfers: int,
    max_transfers_per_gw: int,
) -> None:
    initial_set = set(initial_squad_ids)
    for idx, gw in enumerate(gws):
        for p in player_ids:
            prev = 1 if (idx == 0 and p in initial_set) else (
                0 if idx == 0 else squad[(p, gws[idx - 1])]
            )
            prob += (
                squad[(p, gw)] - prev == xfers_in[(p, gw)] - xfers_out[(p, gw)],
                f"xfer_balance_{p}_{gw}",
            )
            prob += (
                xfers_in[(p, gw)] + xfers_out[(p, gw)] <= 1,
                f"xfer_no_bounce_{p}_{gw}",
            )

        gw_free = free_transfers if idx == 0 else FPL_RULES.default_free_transfers
        prob += (
            extra_hits[gw]
            >= pulp.lpSum(xfers_in[(p, gw)] for p in player_ids) - gw_free,
            f"hits_{gw}",
        )
        prob += (
            pulp.lpSum(xfers_in[(p, gw)] for p in player_ids) <= max_transfers_per_gw,
            f"xfer_cap_{gw}",
        )


def solve(inputs: OptimizerInputs) -> OptimizerResult:
    """Run the multi-gameweek MILP and return the full plan."""
    players = inputs.players.copy()
    player_ids: list[int] = players.index.tolist()
    gws: list[int] = list(range(inputs.start_gw, inputs.start_gw + inputs.horizon))

    ep = _build_projection_matrix(inputs.projections, player_ids, gws)

    prob = pulp.LpProblem("gaffer_squad", pulp.LpMaximize)

    squad = {
        (p, gw): pulp.LpVariable(f"squad_{p}_{gw}", cat="Binary")
        for p in player_ids
        for gw in gws
    }
    starting = {
        (p, gw): pulp.LpVariable(f"start_{p}_{gw}", cat="Binary")
        for p in player_ids
        for gw in gws
    }
    captain = {
        (p, gw): pulp.LpVariable(f"cap_{p}_{gw}", cat="Binary")
        for p in player_ids
        for gw in gws
    }

    budget_cap = FPL_RULES.budget + inputs.bank
    _add_squad_constraints(prob, squad, starting, captain, players, gws, budget_cap)

    has_initial = inputs.initial_squad_ids is not None
    xfers_in: dict[tuple[int, int], pulp.LpVariable] = {}
    xfers_out: dict[tuple[int, int], pulp.LpVariable] = {}
    extra_hits: dict[int, pulp.LpVariable] = {}
    if has_initial:
        for p in player_ids:
            for gw in gws:
                xfers_in[(p, gw)] = pulp.LpVariable(f"in_{p}_{gw}", cat="Binary")
                xfers_out[(p, gw)] = pulp.LpVariable(f"out_{p}_{gw}", cat="Binary")
        for gw in gws:
            extra_hits[gw] = pulp.LpVariable(
                f"hits_{gw}", lowBound=0, cat="Integer"
            )
        _add_transfer_constraints(
            prob,
            squad,
            xfers_in,
            xfers_out,
            extra_hits,
            player_ids,
            gws,
            inputs.initial_squad_ids,  # type: ignore[arg-type]
            inputs.free_transfers,
            inputs.max_transfers_per_gw,
        )

    # Objective
    obj: list[pulp.LpAffineExpression] = []
    for gw in gws:
        for p in player_ids:
            pts = float(ep.loc[p, gw])
            if pts == 0:
                continue
            obj.append(pts * starting[(p, gw)])
            obj.append(pts * captain[(p, gw)])  # captain adds +1× on top of starting
            obj.append(
                pts * inputs.bench_weight * (squad[(p, gw)] - starting[(p, gw)])
            )
    if has_initial:
        for gw in gws:
            obj.append(-FPL_RULES.transfer_hit_cost * extra_hits[gw])

    prob += pulp.lpSum(obj)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=inputs.time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if prob.status != pulp.LpStatusOptimal and prob.status != pulp.LpStatusNotSolved:
        # Not strictly fatal (time-limited may be feasible suboptimal), but surface it.
        pass

    plans = _extract_plans(
        prob=prob,
        squad=squad,
        starting=starting,
        captain=captain,
        xfers_in=xfers_in,
        xfers_out=xfers_out,
        extra_hits=extra_hits,
        ep=ep,
        gws=gws,
        player_ids=player_ids,
        players=players,
        has_initial=has_initial,
        bench_weight=inputs.bench_weight,
    )
    total_ep = sum(p.expected_points for p in plans)
    return OptimizerResult(
        plans=plans,
        total_expected_points=total_ep,
        solver_status=status,
        objective_value=float(pulp.value(prob.objective) or 0.0),
    )


def _extract_plans(
    prob: pulp.LpProblem,
    squad: dict[tuple[int, int], pulp.LpVariable],
    starting: dict[tuple[int, int], pulp.LpVariable],
    captain: dict[tuple[int, int], pulp.LpVariable],
    xfers_in: dict[tuple[int, int], pulp.LpVariable],
    xfers_out: dict[tuple[int, int], pulp.LpVariable],
    extra_hits: dict[int, pulp.LpVariable],
    ep: pd.DataFrame,
    gws: list[int],
    player_ids: list[int],
    players: pd.DataFrame,
    has_initial: bool,
    bench_weight: float,
) -> list[GameweekPlan]:
    """Pull integer solution values and assemble the per-GW plans."""
    plans: list[GameweekPlan] = []
    for gw in gws:
        chosen = [p for p in player_ids if _is_one(squad[(p, gw)])]
        xi = [p for p in chosen if _is_one(starting[(p, gw)])]
        bench = [p for p in chosen if p not in xi]
        cap = next((p for p in player_ids if _is_one(captain[(p, gw)])), xi[0])

        # Vice-captain: second-best XI member by expected points.
        xi_sorted = sorted(xi, key=lambda p: float(ep.loc[p, gw]), reverse=True)
        vice = next((p for p in xi_sorted if p != cap), xi_sorted[0])

        bench_ordered = _order_bench(bench, ep, gw, players)

        t_in = (
            [p for p in player_ids if _is_one(xfers_in[(p, gw)])] if has_initial else []
        )
        t_out = (
            [p for p in player_ids if _is_one(xfers_out[(p, gw)])] if has_initial else []
        )
        hits = int(round(extra_hits[gw].value() or 0)) if has_initial else 0

        ep_gross = sum(
            float(ep.loc[p, gw])
            * (
                (1.0 if p in xi else bench_weight)
                + (1.0 if p == cap else 0.0)
            )
            for p in chosen
        )
        ep_net = ep_gross - hits * FPL_RULES.transfer_hit_cost

        plans.append(
            GameweekPlan(
                gameweek=gw,
                squad_ids=chosen,
                xi_ids=xi,
                bench_ids=bench_ordered,
                captain_id=cap,
                vice_captain_id=vice,
                transfers_in=t_in,
                transfers_out=t_out,
                hit_cost=hits * FPL_RULES.transfer_hit_cost,
                expected_points=ep_net,
            )
        )
    return plans


def _is_one(var: pulp.LpVariable) -> bool:
    value = var.value()
    return value is not None and value > 0.5


def _order_bench(
    bench: list[int], ep: pd.DataFrame, gw: int, players: pd.DataFrame
) -> list[int]:
    """Return bench ordered as [GK, outfield1, outfield2, outfield3] by desc xp."""
    gk = [p for p in bench if players.loc[p, "position"] == Position.GKP.value]
    outfield = sorted(
        (p for p in bench if players.loc[p, "position"] != Position.GKP.value),
        key=lambda p: float(ep.loc[p, gw]),
        reverse=True,
    )
    return gk + outfield
