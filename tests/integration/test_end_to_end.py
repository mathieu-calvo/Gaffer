"""End-to-end: stub providers → training matrix → optimizer plan."""

from __future__ import annotations

import pandas as pd
import pytest

from gaffer.optimizer.milp import OptimizerInputs, solve
from gaffer.services.prediction_service import build_training_set
from tests.fixtures.stub_providers import StubFplProvider, StubHistoricalProvider


class TestBuildTrainingSet:
    def test_yields_aligned_X_y_seasons(self):
        td = build_training_set(
            fpl=StubFplProvider(),
            historical=StubHistoricalProvider(n_players=4, n_seasons=3),
            alpha=0.5,
        )
        assert len(td.X) == len(td.y) == len(td.seasons)
        assert td.X.index.equals(td.y.index)

    def test_target_is_numeric(self):
        td = build_training_set(
            fpl=StubFplProvider(),
            historical=StubHistoricalProvider(n_players=4, n_seasons=3),
            alpha=0.5,
        )
        assert pd.api.types.is_numeric_dtype(td.y)


class TestPredictThenOptimize:
    def test_synthetic_projection_to_squad_plan(self, toy_player_pool, toy_projections):
        # Skip the predictor step (covered in test_models) and feed projections
        # straight into the solver — exactly what the prediction → optimization
        # service handoff does in production.
        result = solve(OptimizerInputs(
            projections=toy_projections,
            players=toy_player_pool,
            start_gw=1,
            horizon=2,
            time_limit=30,
        ))
        assert result.solver_status == "Optimal"
        assert len(result.plans) == 2
        for plan in result.plans:
            assert len(plan.squad_ids) == 15
            assert len(plan.xi_ids) == 11
            assert plan.captain_id in plan.xi_ids

    def test_transfer_planning_keeps_initial_squad_intact_when_no_gain(
        self, toy_player_pool, toy_projections
    ):
        # If we hand the optimizer a strong starting squad (same as it would
        # pick from scratch), there should be no transfers — the hit cost
        # would just destroy value.
        from_scratch = solve(OptimizerInputs(
            projections=toy_projections, players=toy_player_pool,
            start_gw=1, horizon=1, time_limit=30,
        ))
        gw1_squad = from_scratch.plans[0].squad_ids
        with_initial = solve(OptimizerInputs(
            projections=toy_projections, players=toy_player_pool,
            start_gw=1, horizon=1, time_limit=30,
            initial_squad_ids=gw1_squad, free_transfers=1,
        ))
        # 0 transfers in, 0 out — the optimal squad is already the initial one.
        assert with_initial.plans[0].transfers_in == []
        assert with_initial.plans[0].transfers_out == []
        assert with_initial.plans[0].hit_cost == 0


@pytest.mark.slow
class TestRealCsvProvider:
    """Smoke tests against the bundled historical CSV — only run with `-m slow`."""

    def test_csv_loads(self):
        from gaffer.providers.historical_csv import HistoricalCsvProvider
        df = HistoricalCsvProvider().get_historical_gwdata()
        assert len(df) > 0
        assert "name" in df.columns
        assert "total_points" in df.columns
