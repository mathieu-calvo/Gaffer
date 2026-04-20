"""MILP optimizer tests."""

from __future__ import annotations

import pandas as pd
import pytest

from gaffer.domain.constraints import FPL_RULES
from gaffer.optimizer.milp import OptimizerInputs, solve
from gaffer.optimizer.result import GameweekPlan, OptimizerResult
from gaffer.services.optimization_service import optimize_squad


@pytest.fixture
def solved_result(toy_player_pool, toy_projections) -> OptimizerResult:
    inputs = OptimizerInputs(
        projections=toy_projections,
        players=toy_player_pool,
        start_gw=1,
        horizon=3,
        time_limit=30,
    )
    return solve(inputs)


class TestSolveBasic:
    def test_returns_result_with_one_plan_per_gw(self, solved_result):
        assert len(solved_result.plans) == 3
        assert {p.gameweek for p in solved_result.plans} == {1, 2, 3}

    def test_solver_status_optimal(self, solved_result):
        assert solved_result.solver_status == "Optimal"

    def test_total_expected_points_matches_plan_sum(self, solved_result):
        total = sum(p.expected_points for p in solved_result.plans)
        assert solved_result.total_expected_points == pytest.approx(total)


class TestSquadConstraints:
    def test_squad_size_15(self, solved_result):
        for plan in solved_result.plans:
            assert len(plan.squad_ids) == FPL_RULES.squad_size

    def test_position_quotas(self, solved_result, toy_player_pool):
        for plan in solved_result.plans:
            counts = toy_player_pool.loc[plan.squad_ids, "position"].value_counts()
            assert counts.get("GKP", 0) == 2
            assert counts.get("DEF", 0) == 5
            assert counts.get("MID", 0) == 5
            assert counts.get("FWD", 0) == 3

    def test_budget_respected(self, solved_result, toy_player_pool):
        for plan in solved_result.plans:
            total = toy_player_pool.loc[plan.squad_ids, "price"].sum()
            assert total <= FPL_RULES.budget + 1e-6

    def test_club_cap_respected(self, solved_result, toy_player_pool):
        for plan in solved_result.plans:
            counts = toy_player_pool.loc[plan.squad_ids, "team"].value_counts()
            assert (counts <= FPL_RULES.max_per_club).all()


class TestXIAndCaptain:
    def test_xi_is_eleven(self, solved_result):
        for plan in solved_result.plans:
            assert len(plan.xi_ids) == FPL_RULES.xi_size

    def test_xi_subset_of_squad(self, solved_result):
        for plan in solved_result.plans:
            assert set(plan.xi_ids).issubset(set(plan.squad_ids))

    def test_bench_is_four(self, solved_result):
        for plan in solved_result.plans:
            assert len(plan.bench_ids) == FPL_RULES.bench_size

    def test_bench_first_slot_is_gk(self, solved_result, toy_player_pool):
        for plan in solved_result.plans:
            first_bench = plan.bench_ids[0]
            assert toy_player_pool.loc[first_bench, "position"] == "GKP"

    def test_xi_and_bench_partition_squad(self, solved_result):
        for plan in solved_result.plans:
            assert set(plan.xi_ids) | set(plan.bench_ids) == set(plan.squad_ids)
            assert not (set(plan.xi_ids) & set(plan.bench_ids))

    def test_captain_in_xi(self, solved_result):
        for plan in solved_result.plans:
            assert plan.captain_id in plan.xi_ids

    def test_vice_captain_in_xi_and_distinct(self, solved_result):
        for plan in solved_result.plans:
            assert plan.vice_captain_id in plan.xi_ids
            assert plan.vice_captain_id != plan.captain_id

    def test_legal_formation(self, solved_result, toy_player_pool):
        for plan in solved_result.plans:
            counts = toy_player_pool.loc[plan.xi_ids, "position"].value_counts()
            assert counts.get("GKP", 0) == 1
            assert 3 <= counts.get("DEF", 0) <= 5
            assert 2 <= counts.get("MID", 0) <= 5
            assert 1 <= counts.get("FWD", 0) <= 3


class TestTransfers:
    def test_no_transfers_when_no_initial_squad(self, solved_result):
        for plan in solved_result.plans:
            assert plan.transfers_in == []
            assert plan.transfers_out == []
            assert plan.hit_cost == 0

    def test_transfer_planning_with_initial_squad(
        self, toy_player_pool, toy_projections
    ):
        # Build a plausible starting 15-man squad: pick the cheapest valid lineup.
        starting_ids: list[int] = []
        quota = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
        teams_used: dict[str, int] = {}
        for pos, n in quota.items():
            pool = toy_player_pool[toy_player_pool["position"] == pos].sort_values("price")
            picked = 0
            for pid, row in pool.iterrows():
                if teams_used.get(row["team"], 0) >= 3:
                    continue
                starting_ids.append(int(pid))
                teams_used[row["team"]] = teams_used.get(row["team"], 0) + 1
                picked += 1
                if picked == n:
                    break
        assert len(starting_ids) == 15

        inputs = OptimizerInputs(
            projections=toy_projections,
            players=toy_player_pool,
            start_gw=1,
            horizon=2,
            initial_squad_ids=starting_ids,
            free_transfers=1,
            time_limit=30,
        )
        result = solve(inputs)
        assert result.solver_status == "Optimal"
        # Transfer counts should be balanced (in == out per gw).
        for plan in result.plans:
            assert len(plan.transfers_in) == len(plan.transfers_out)


class TestOptimizationService:
    def test_service_wraps_solver(self, toy_player_pool, toy_projections):
        result = optimize_squad(
            projections=toy_projections,
            players=toy_player_pool,
            start_gw=1,
            horizon=1,
            time_limit=30,
        )
        assert isinstance(result, OptimizerResult)
        assert len(result.plans) == 1
        assert isinstance(result.plans[0], GameweekPlan)


class TestObjective:
    def test_captain_doubles_points(self, solved_result, toy_projections):
        # Captain's expected points should be the highest (or tied) of any
        # starter, since the objective weights captain by 2× and starters by 1×.
        for plan in solved_result.plans:
            xi_eps = {
                pid: float(toy_projections.loc[(pid, plan.gameweek), "expected_points"])
                for pid in plan.xi_ids
            }
            cap_ep = xi_eps[plan.captain_id]
            assert cap_ep == max(xi_eps.values())

    def test_high_ep_players_preferred(self, toy_player_pool):
        # Build projections where MID id 17 (first MID) gets a huge boost — it
        # should land in the squad in every gw.
        rows = []
        gks = toy_player_pool[toy_player_pool["position"] == "GKP"].index.tolist()
        for pid in toy_player_pool.index:
            for gw in (1, 2):
                ep = 10.0 if pid == gks[0] else 1.0
                rows.append({"player_id": int(pid), "gameweek": gw, "expected_points": ep})
        projections = pd.DataFrame(rows).set_index(["player_id", "gameweek"])
        result = solve(OptimizerInputs(
            projections=projections,
            players=toy_player_pool,
            start_gw=1,
            horizon=2,
            time_limit=30,
        ))
        for plan in result.plans:
            assert gks[0] in plan.squad_ids
